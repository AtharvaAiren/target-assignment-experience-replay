"""Provenance and checkout-safety tests for the reviewer add-on overlay."""

from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from experiments.gradient_reuse_reviewer_extension import mammoth_bootstrap, overlay
from experiments.gradient_reuse_mechanisms.protocol import canonical_sha256, sha256_file


def _fake_git(root: Path, *arguments: str) -> str:
    del root
    if arguments == ("rev-parse", "HEAD"):
        return overlay.UPSTREAM_COMMIT
    if arguments == ("status", "--porcelain", "--untracked-files=all"):
        return ""
    raise AssertionError(f"unexpected git invocation: {arguments}")


def _fake_loggers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, ModuleType, list[tuple[object, ...]], list[dict[str, object]]]:
    mammoth = tmp_path / "mammoth"
    logger_path = mammoth / "utils/loggers.py"
    logger_path.parent.mkdir(parents=True)
    logger_path.write_text("# authenticated test logger\n", encoding="utf-8")

    original_calls: list[tuple[object, ...]] = []
    wandb_calls: list[dict[str, object]] = []
    module = ModuleType("utils.loggers")
    module.__file__ = str(logger_path)

    def original(*arguments: object) -> str:
        original_calls.append(arguments)
        return "upstream-result"

    module.log_extra_metrics = original
    module.wandb = SimpleNamespace(log=wandb_calls.append)
    monkeypatch.setattr(
        mammoth_bootstrap,
        "MAMMOTH_LOGGERS_SHA256",
        sha256_file(logger_path),
    )
    return mammoth, module, original_calls, wandb_calls


def test_historical_overlay_and_external_addon_have_exact_closed_inventories() -> None:
    base = overlay.validate_base_sources()
    external = overlay.external_files()

    assert base == overlay.BASE_FILES
    assert len(base) == 19
    assert canonical_sha256(base) == overlay.BASE_BUNDLE_SHA256
    assert set(external) == overlay.EXTERNAL_FILES
    assert overlay.extension_bundle_sha256() == canonical_sha256(external)
    assert not any("NOTICE" in name or "__pycache__" in name for name in external)


def test_install_writes_only_historical_base_overlay_into_mammoth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "mammoth"
    checkout.mkdir()
    monkeypatch.setattr(overlay, "_git", _fake_git)
    monkeypatch.setattr(overlay, "overlay_checkout_lock", lambda root: nullcontext())

    binding = overlay.install_overlay(checkout)

    assert binding["base_overlay_bundle_sha256"] == overlay.BASE_BUNDLE_SHA256
    assert binding["base_file_count"] == 19
    assert binding["supplemental_files_written_into_mammoth"] is False
    assert set(binding["external_files"]) == overlay.EXTERNAL_FILES
    for relative, expected in overlay.BASE_FILES.items():
        installed = checkout / relative
        assert installed.is_file() and not installed.is_symlink()
        assert sha256_file(installed) == expected
    for relative in overlay.EXTERNAL_FILES:
        assert not (checkout / relative).exists()
        assert not (checkout / relative).is_symlink()
    assert (checkout / overlay.BASE_MANIFEST_NAME).is_file()


def test_verifier_rejects_one_changed_installed_base_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "mammoth"
    checkout.mkdir()
    monkeypatch.setattr(overlay, "_git", _fake_git)
    monkeypatch.setattr(overlay, "overlay_checkout_lock", lambda root: nullcontext())
    overlay.install_overlay(checkout)
    relative = sorted(overlay.BASE_FILES)[0]
    (checkout / relative).write_text("changed\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="installed base-overlay file changed"):
        overlay.verify_installed_overlay(checkout)


def test_copy_exact_rejects_a_dangling_destination_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("authenticated bytes\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    destination = tmp_path / "checkout" / "model.py"
    destination.parent.mkdir()
    destination.symlink_to(outside)

    with pytest.raises(RuntimeError, match="symlink|mismatched|refusing"):
        overlay._copy_exact(source, destination, sha256_file(source))

    assert destination.is_symlink()
    assert not outside.exists()


def test_copy_exact_ignores_an_old_predictable_symbolic_temporary_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("authenticated bytes\n", encoding="utf-8")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    destination = checkout / "model.py"
    outside = tmp_path / "outside.py"
    outside.write_text("must remain unchanged\n", encoding="utf-8")
    temporary = checkout / f".model.py.{os.getpid()}.tmp"
    temporary.symlink_to(outside)

    overlay._copy_exact(source, destination, sha256_file(source))

    assert outside.read_text(encoding="utf-8") == "must remain unchanged\n"
    assert destination.read_text(encoding="utf-8") == "authenticated bytes\n"
    assert destination.is_file() and not destination.is_symlink()
    assert temporary.is_symlink()


def test_manifest_writer_ignores_an_old_predictable_symbolic_temporary_path(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "overlay.json"
    outside = tmp_path / "outside.json"
    outside.write_text("must remain unchanged\n", encoding="utf-8")
    temporary = tmp_path / f".overlay.json.{os.getpid()}.tmp"
    temporary.symlink_to(outside)

    overlay._write_manifest(destination, {"status": "PASS"})

    assert outside.read_text(encoding="utf-8") == "must remain unchanged\n"
    assert destination.read_text(encoding="utf-8") == ('{\n  "status": "PASS"\n}\n')
    assert destination.is_file() and not destination.is_symlink()
    assert temporary.is_symlink()


def test_external_overlay_rejects_an_unlisted_regular_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = tmp_path / "models" / "expected.py"
    expected.parent.mkdir(parents=True)
    expected.write_text("expected\n", encoding="utf-8")
    extra = tmp_path / "models" / "extra.py"
    extra.write_text("extra\n", encoding="utf-8")
    monkeypatch.setattr(overlay, "EXTERNAL_OVERLAY", tmp_path)
    monkeypatch.setattr(overlay, "EXTERNAL_FILES", frozenset({"models/expected.py"}))

    with pytest.raises(RuntimeError, match="unexpected reviewer add-on files"):
        overlay.external_files()


def test_domain_logger_preserves_upstream_task_il_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mammoth, module, original_calls, wandb_calls = _fake_loggers(tmp_path, monkeypatch)
    mammoth_bootstrap._patch_domain_il_logger(module, mammoth)
    arguments = SimpleNamespace(dataset="seq-cifar100", nowand=False)

    observed = module.log_extra_metrics(
        arguments, 7.25, 0.0, "Backward Transfer", 20, "CUSTOM"
    )

    assert observed == "upstream-result"
    assert original_calls == [(arguments, 7.25, 0.0, "Backward Transfer", 20, "CUSTOM")]
    assert wandb_calls == []


def test_domain_logger_handles_only_finite_permuted_mnist_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mammoth, module, original_calls, wandb_calls = _fake_loggers(tmp_path, monkeypatch)
    mammoth_bootstrap._patch_domain_il_logger(module, mammoth)

    module.log_extra_metrics(
        SimpleNamespace(dataset="perm-mnist", nowand=False),
        12.5,
        None,
        "Backward Transfer",
        20,
        "CUSTOM",
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Backward Transfer: [Domain-IL]: 12.50\n"
        "\tRaw Backward Transfer values: Domain-IL 12.5\n"
    )
    assert original_calls == []
    assert wandb_calls == [{"CUSTOM_domain_Backward Transfer": 12.5, "Task": 20}]


def test_domain_logger_rejects_missing_task_il_outside_permuted_mnist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mammoth, module, original_calls, wandb_calls = _fake_loggers(tmp_path, monkeypatch)
    mammoth_bootstrap._patch_domain_il_logger(module, mammoth)

    with pytest.raises(RuntimeError, match="outside the authenticated Domain-IL"):
        module.log_extra_metrics(
            SimpleNamespace(dataset="seq-cifar100", nowand=True),
            12.5,
            None,
            "Backward Transfer",
            20,
        )

    assert original_calls == []
    assert wandb_calls == []


@pytest.mark.parametrize("metric", [float("nan"), float("inf"), float("-inf")])
def test_domain_logger_rejects_nonfinite_permuted_mnist_metric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metric: float
) -> None:
    mammoth, module, original_calls, wandb_calls = _fake_loggers(tmp_path, monkeypatch)
    mammoth_bootstrap._patch_domain_il_logger(module, mammoth)

    with pytest.raises(RuntimeError, match="non-finite Domain-IL"):
        module.log_extra_metrics(
            SimpleNamespace(dataset="perm-mnist", nowand=True),
            metric,
            None,
            "Backward Transfer",
            20,
        )

    assert original_calls == []
    assert wandb_calls == []


def test_domain_logger_rejects_changed_mammoth_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mammoth, module, _, _ = _fake_loggers(tmp_path, monkeypatch)
    logger_path = Path(module.__file__)
    logger_path.write_text("# changed after authentication\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="logger source changed"):
        mammoth_bootstrap._patch_domain_il_logger(module, mammoth)
