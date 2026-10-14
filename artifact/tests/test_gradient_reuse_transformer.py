"""Scientific-contract tests for the fixed-conditioning Transformer audit."""

from __future__ import annotations

import copy
import gzip
import itertools
import json
import os
import socket
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments.gradient_reuse_transformer.assets import (
    AssetContractError,
    sha256_file,
    sha256_tree,
    validate_asset_manifest,
)
from experiments.gradient_reuse_transformer.acquire import acquire_assets
from experiments.gradient_reuse_transformer.audit import (
    NextTokenPacket,
    _gradient,
    next_token_metric_sums,
    run_next_token_audit,
)
from experiments.gradient_reuse_transformer.data import (
    _is_allowed_license,
    build_chronological_stream,
    iter_json_records,
    packetize,
)
from experiments.gradient_reuse_transformer.model import (
    PythiaNextTokenModel,
    load_local_pythia,
)
from experiments.gradient_reuse_transformer.run import (
    ACTUATOR_NAME,
    DEFAULT_CONFIG,
    PREFLIGHT_SCHEMA,
    _artifact_inventory,
    _optimizer,
    _validate_asset_source_binding,
    build_environment_manifest,
    build_validated_input_bundle,
    capture_runtime_environment,
    main,
    read_config,
    run_tiny_smoke,
    validate_config,
    validate_cuda_preflight_report,
    validate_embedded_input_bundle,
    validate_environment_manifest,
    validate_transformer_source_manifest,
)
from experiments.source_provenance import build_source_manifest


class IntegerTokenizer:
    vocab_size = 41

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[int]]:
        self.calls.append(dict(kwargs))
        return {"input_ids": [int(value) for value in text.split()]}


def _write_transformer_source_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "transformer-source.json"
    path.write_text(
        json.dumps(build_source_manifest("transformer"), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_cli_identity_files(tmp_path: Path) -> dict[str, Path]:
    model_root = tmp_path / "model"
    dataset_root = tmp_path / "dataset"
    model_root.mkdir(exist_ok=True)
    dataset_root.mkdir(exist_ok=True)
    paths = {
        "asset_manifest": tmp_path / "assets.json",
        "source_manifest": tmp_path / "source.json",
        "environment_manifest": tmp_path / "environment.json",
        "preflight_report": tmp_path / "preflight-input.json",
    }
    paths["asset_manifest"].write_text(
        json.dumps(
            {
                "schema_version": "fmmt.transformer_assets.v1",
                "assets": {
                    "model": {"kind": "directory", "path": "model"},
                    "dataset": {"kind": "directory", "path": "dataset"},
                },
            }
        ),
        encoding="utf-8",
    )
    paths["source_manifest"].write_text(
        json.dumps({"schema_version": "test.source.v1"}), encoding="utf-8"
    )
    paths["environment_manifest"].write_text(
        json.dumps({"schema_version": "test.environment.v1"}), encoding="utf-8"
    )
    paths["preflight_report"].write_text(
        json.dumps({"schema_version": PREFLIGHT_SCHEMA}), encoding="utf-8"
    )
    return paths


def _record(
    index: int,
    date: str,
    *,
    license_name: str = "CC BY 4.0",
    tokens: list[int] | None = None,
    trailing_spaces: int = 0,
) -> dict[str, Any]:
    values = tokens or [
        (index + step * 3 + step**2) % IntegerTokenizer.vocab_size
        for step in range(14)
    ]
    return {
        "id": index,
        "source": f"source-{index % 2}",
        "created": date,
        "text": " ".join(str(value) for value in values) + " " * trailing_spaces,
        "metadata": {
            "license": license_name,
            "url": f"https://invalid.example/{index}",
        },
    }


def test_locked_config_is_full_scale_and_fixed_conditioning() -> None:
    config = read_config(DEFAULT_CONFIG)
    assert config["model"]["architecture"] == "Pythia-160M-deduped"
    assert config["stream"]["context_length"] == 128
    assert config["stream"]["warmup_examples"] == 4096
    assert config["stream"]["online_examples"] == 8192
    assert config["stream"]["conditioning_policy"] == "fixed_context_predicted_token_only"
    assert config["audit"]["n_permutations"] == 4
    assert config["audit"]["independent_model_trajectories"] == 1
    assert config["audit"]["seed_level_replication"] is False
    assert "seeds" not in config["audit"]
    assert config["audit"]["conditioning_tokens_permuted"] is False
    assert config["optimization"]["counterfactual_actuator"] == ACTUATOR_NAME
    assert config["optimization"]["adamw_foreach"] is False
    assert config["optimization"]["adamw_fused"] is False
    assert config["audit"]["fail_on_zero_norm_actuator_branch"] is True
    assert config["cuda_preflight"] == {
        "required_for_paper_eligible_full_run": True,
        "minimum_warmup_packets": 2,
        "minimum_online_packets": 3,
        "minimum_total_device_memory_bytes": 32000000000,
        "maximum_peak_reserved_bytes": 27917287424,
    }
    assert all(
        len(config["assets"][role]["resolved_revision"]) == 40
        for role in ("model", "dataset")
    )


def test_config_rejects_a_mutable_asset_revision() -> None:
    config = read_config(DEFAULT_CONFIG)
    changed = copy.deepcopy(config)
    changed["assets"]["dataset"]["resolved_revision"] = "main"
    with pytest.raises(RuntimeError, match="not pinned"):
        validate_config(changed)


def test_config_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-config.json"
    path.write_text(
        '{"schema_version":"first","schema_version":"second"}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="strict JSON"):
        read_config(path)


def test_config_rejects_permutation_seeds_as_model_replications() -> None:
    config = read_config(DEFAULT_CONFIG)
    changed = copy.deepcopy(config)
    changed["audit"]["seeds"] = [300, 301, 302, 303]
    with pytest.raises(RuntimeError, match="may not be presented as trajectory replicas"):
        validate_config(changed)


def test_config_and_optimizer_fix_raw_actuator_and_adamw_implementation() -> None:
    config = read_config(DEFAULT_CONFIG)
    model = torch.nn.Linear(3, 2)
    optimizer = _optimizer(model, config["optimization"])
    assert optimizer.defaults["foreach"] is False
    assert optimizer.defaults["fused"] is False

    changed = copy.deepcopy(config)
    changed["optimization"]["counterfactual_actuator"] = "adamw_counterfactual"
    with pytest.raises(RuntimeError, match="raw-gradient rule"):
        validate_config(changed)

    changed = copy.deepcopy(config)
    changed["optimization"]["adamw_foreach"] = True
    with pytest.raises(RuntimeError, match="foreach"):
        validate_config(changed)


def test_asset_manifest_authenticates_local_bytes_and_rejects_drift(
    tmp_path: Path,
) -> None:
    model = tmp_path / "model"
    dataset = tmp_path / "dataset"
    model.mkdir()
    dataset.mkdir()
    (model / "config.json").write_text('{"model_type":"gpt_neox"}\n')
    (model / "model.safetensors").write_bytes(b"locked-model")
    (dataset / "news.jsonl").write_text('{"id":1}\n')
    requirements = {
        "model": {
            "asset_name": "model",
            "repository_id": "org/model",
            "resolved_revision": "a" * 40,
            "kind": "directory",
        },
        "dataset": {
            "asset_name": "dataset",
            "repository_id": "org/dataset",
            "resolved_revision": "b" * 40,
            "kind": "directory",
        },
    }
    manifest = tmp_path / "assets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "fmmt.transformer_assets.v1",
                "assets": {
                    "model": {
                        "path": "model",
                        "kind": "directory",
                        "repository_id": "org/model",
                        "requested_revision": "a" * 40,
                        "resolved_revision": "a" * 40,
                        "sha256": sha256_tree(model),
                    },
                    "dataset": {
                        "path": "dataset",
                        "kind": "directory",
                        "repository_id": "org/dataset",
                        "requested_revision": "b" * 40,
                        "resolved_revision": "b" * 40,
                        "sha256": sha256_tree(dataset),
                    },
                },
            }
        )
    )
    validated = validate_asset_manifest(manifest, requirements)
    assert validated["model"].path == model
    assert validated["dataset"].sha256 == sha256_tree(dataset)

    (dataset / "news.jsonl").write_text('{"id":2}\n')
    with pytest.raises(AssetContractError, match="content hash mismatch"):
        validate_asset_manifest(manifest, requirements)


def test_asset_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    manifest = tmp_path / "duplicate-assets.json"
    manifest.write_text(
        '{"schema_version":"fmmt.transformer_assets.v1",'
        '"assets":{},"assets":{}}\n',
        encoding="utf-8",
    )
    with pytest.raises(AssetContractError, match="duplicate JSON key"):
        validate_asset_manifest(manifest, {})


def test_acquisition_materializes_canonical_symlink_free_assets_without_network(
    tmp_path: Path,
) -> None:
    source_manifest = _write_transformer_source_manifest(tmp_path)
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(dict(kwargs))
        local = Path(kwargs["local_dir"])
        local.mkdir(parents=True)
        if kwargs["repo_type"] == "model":
            (local / "config.json").write_text('{"model_type":"gpt_neox"}\n')
            blobs = local / "blobs"
            blobs.mkdir()
            (blobs / "weights.bin").write_bytes(b"immutable weights")
            (local / "pytorch_model.bin").symlink_to(blobs / "weights.bin")
            (local / "tokenizer.json").write_text('{"version":"1.0"}\n')
        else:
            shard = local / "v0/documents/news.json.gz"
            shard.parent.mkdir(parents=True)
            rows = [
                _record(2, "2022-01-02"),
                _record(1, "2022-01-01"),
            ]
            payload = "".join(json.dumps(row) + "\n" for row in rows)
            shard.write_bytes(gzip.compress(payload.encode("utf-8"), mtime=0))
        return str(local)

    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_manifest = acquire_assets(
        config_path=DEFAULT_CONFIG,
        output_root=first_root,
        source_manifest=source_manifest,
        snapshot_download_fn=fake_snapshot_download,
    )
    second_manifest = acquire_assets(
        config_path=DEFAULT_CONFIG,
        output_root=second_root,
        source_manifest=source_manifest,
        snapshot_download_fn=fake_snapshot_download,
    )
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    acquisition_payload = json.loads(first_manifest.read_text())
    assert acquisition_payload["source_provenance"]["experiment"] == "transformer"
    assert (
        acquisition_payload["source_provenance"]["validation_status"] == "PASS"
    )
    assert acquisition_payload["source_provenance"]["source_manifest_sha256"] == (
        sha256_file(source_manifest)
    )
    assert len(
        acquisition_payload["source_provenance"]["source_bundle_sha256"]
    ) == 64
    assert (first_root / "dataset/news.jsonl").read_bytes() == (
        second_root / "dataset/news.jsonl"
    ).read_bytes()
    assert not any(path.is_symlink() for path in first_root.rglob("*"))
    assert not (first_root / "_downloads").exists()
    assert not (first_root / "_dataset_source").exists()
    config = read_config(DEFAULT_CONFIG)
    validated = validate_asset_manifest(
        first_manifest,
        config["assets"],
        expected_config_sha256=sha256_file(DEFAULT_CONFIG),
    )
    assert set(validated) == {"model", "dataset"}
    derivation = json.loads((first_root / "dataset/DERIVATION.json").read_text())
    assert derivation["rows"] == 2
    assert derivation["source_format"] == "json_lines"
    assert derivation["source_resolved_revision"] == config["assets"]["dataset"][
        "resolved_revision"
    ]
    assert all(call["revision"] in {
        config["assets"]["model"]["resolved_revision"],
        config["assets"]["dataset"]["resolved_revision"],
    } for call in calls)
    assert all(call["max_workers"] == 4 for call in calls)
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        acquire_assets(
            config_path=DEFAULT_CONFIG,
            output_root=first_root,
            source_manifest=source_manifest,
            snapshot_download_fn=fake_snapshot_download,
        )


def test_acquisition_canonicalizes_a_pinned_parquet_snapshot(tmp_path: Path) -> None:
    source_manifest = _write_transformer_source_manifest(tmp_path)
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")

    def fake_snapshot_download(**kwargs: Any) -> str:
        local = Path(kwargs["local_dir"])
        local.mkdir(parents=True)
        if kwargs["repo_type"] == "model":
            (local / "config.json").write_text("{}\n")
            (local / "model.safetensors").write_bytes(b"weights")
            (local / "tokenizer.json").write_text("{}\n")
        else:
            rows = [_record(4, "2022-04-04"), _record(3, "2022-04-03")]
            parquet.write_table(pyarrow.Table.from_pylist(rows), local / "train.parquet")
        return str(local)

    root = tmp_path / "parquet-assets"
    acquire_assets(
        config_path=DEFAULT_CONFIG,
        output_root=root,
        source_manifest=source_manifest,
        snapshot_download_fn=fake_snapshot_download,
    )
    derivation = json.loads((root / "dataset/DERIVATION.json").read_text())
    assert derivation["source_format"] == "parquet"
    assert derivation["rows"] == 2
    lines = (root / "dataset/news.jsonl").read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == [4, 3]


@pytest.mark.parametrize(
    "payload",
    [
        '{"id":1,"id":2}',
        '{"id":1,"metadata":{"license":"CC BY 4.0","license":"CC BY-SA 4.0"}}',
        '{"id":1,"score":NaN}',
    ],
)
def test_json_record_reader_rejects_ambiguous_or_nonfinite_json(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(payload + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid strict JSON"):
        list(iter_json_records(path))


def test_stream_is_chronological_deduplicated_and_target_leakage_free() -> None:
    tokenizer = IntegerTokenizer()
    original_tokens = {
        index: [(index + step * 3 + step**2) % tokenizer.vocab_size for step in range(14)]
        for index in range(10)
    }
    records = [
        _record(6, "2022-01-07", tokens=original_tokens[6]),
        _record(1, "Published on January 2, 2022", tokens=original_tokens[1]),
        _record(5, "2022-01-06", tokens=original_tokens[5]),
        _record(0, "2020-01-01", tokens=original_tokens[0]),  # too old
        _record(2, "2022-01-03", license_name="CC BY-NC 4.0", tokens=original_tokens[2]),
        _record(3, "2022-01-04", tokens=original_tokens[3]),
        _record(4, "2022-01-05", tokens=original_tokens[4]),
        _record(7, "2022-01-08", license_name="CC BY-SA 4.0", tokens=original_tokens[7]),
        _record(8, "2022-01-09", tokens=original_tokens[8]),
        _record(9, "2022-01-10", tokens=original_tokens[8]),  # duplicate text
    ]
    stream = build_chronological_stream(
        records,
        tokenizer,
        context_length=5,
        warmup_examples=2,
        online_examples=4,
        earliest_utc="2021-01-01T00:00:00+00:00",
        offset_salt="unit-test-offset",
    )
    examples = stream.warmup + stream.online
    assert [item.timestamp_utc for item in examples] == sorted(
        item.timestamp_utc for item in examples
    )
    assert stream.manifest["rejected_timestamp"] == 1
    assert stream.manifest["rejected_license"] == 1
    assert stream.manifest["rejected_duplicate"] == 1
    assert stream.manifest["target_position_strictly_after_context"] is True
    assert stream.manifest["selection_depends_on_model_seed"] is False
    for item in examples:
        assert item.target_position == item.window_start + 5
        assert len(item.input_ids) == 5
        # The target occupies the next source position, never an input position.
        matched = next(
            values
            for values in original_tokens.values()
            if item.text_sha256
            == __import__("hashlib").sha256(
                " ".join(str(value) for value in values).encode()
            ).hexdigest()
        )
        assert list(item.input_ids) == matched[item.window_start : item.target_position]
        assert item.target_id == matched[item.target_position]
    assert all(
        call
        == {
            "add_special_tokens": False,
            "padding": False,
            "truncation": False,
            "return_attention_mask": False,
        }
        for call in tokenizer.calls
    )
    second = build_chronological_stream(
        records,
        IntegerTokenizer(),
        context_length=5,
        warmup_examples=2,
        online_examples=4,
        earliest_utc="2021-01-01T00:00:00+00:00",
        offset_salt="unit-test-offset",
    )
    assert second == stream


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("CC BY 4.0", True),
        ("CC-BY-SA-3.0", True),
        (
            "Creative Commons - Attribution - "
            "https://creativecommons.org/licenses/by/4.0/",
            True,
        ),
        ("CC BY NC 4.0", False),
        ("not CC BY 4.0", False),
        ("CC BY-ND 4.0", False),
        ("Creative Commons Attribution", False),
    ],
)
def test_license_filter_uses_an_exact_normalized_allowlist(
    label: str, expected: bool
) -> None:
    assert _is_allowed_license(label) is expected


def test_packetizer_preserves_fixed_contexts_and_targets_separately() -> None:
    stream = build_chronological_stream(
        [_record(index, f"2022-02-{index + 1:02d}", trailing_spaces=index) for index in range(8)],
        IntegerTokenizer(),
        context_length=4,
        warmup_examples=2,
        online_examples=6,
        earliest_utc="2021-01-01T00:00:00+00:00",
        offset_salt="packet-test",
    )
    packets = list(packetize(stream.online, 2))
    assert len(packets) == 3
    assert all(packet.input_ids.shape == (2, 4) for packet in packets)
    assert all(packet.targets.dtype == np.int64 for packet in packets)
    assert [packet.packet_number for packet in packets] == [0, 1, 2]
    assert not any(
        np.shares_memory(packet.targets, packet.input_ids.numpy()) for packet in packets
    )


def test_window_offset_does_not_depend_on_article_token_values() -> None:
    first_record = _record(17, "2022-05-01", tokens=list(range(14)))
    changed_record = copy.deepcopy(first_record)
    changed_record["text"] = " ".join(str((value + 9) % 41) for value in range(14))
    first = build_chronological_stream(
        [first_record, _record(18, "2022-05-02")],
        IntegerTokenizer(),
        context_length=5,
        warmup_examples=1,
        online_examples=1,
        earliest_utc="2021-01-01T00:00:00+00:00",
        offset_salt="content-independent-offset",
    )
    changed = build_chronological_stream(
        [changed_record, _record(18, "2022-05-02")],
        IntegerTokenizer(),
        context_length=5,
        warmup_examples=1,
        online_examples=1,
        earliest_utc="2021-01-01T00:00:00+00:00",
        offset_salt="content-independent-offset",
    )
    assert first.warmup[0].example_id == changed.warmup[0].example_id
    assert first.warmup[0].window_start == changed.warmup[0].window_start
    assert first.warmup[0].text_sha256 != changed.warmup[0].text_sha256


class FakeBackbone(torch.nn.Module):
    def __init__(self, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            vocab_size=7,
            is_encoder_decoder=False,
            use_cache=True,
            hidden_dropout=dropout,
            attention_dropout=0.0,
        )
        self.embedding = torch.nn.Embedding(7, 5)
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(5, 7)
        self.calls: list[tuple[bool, bool, int, torch.Tensor]] = []

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        logits_to_keep: int,
    ) -> Any:
        self.calls.append(
            (use_cache, return_dict, logits_to_keep, attention_mask.detach().clone())
        )
        logits = self.head(self.dropout(self.embedding(input_ids)))
        return SimpleNamespace(logits=logits[:, -logits_to_keep:, :])


def test_model_wrapper_scores_only_final_position_and_disables_cache() -> None:
    backbone = FakeBackbone()
    model = PythiaNextTokenModel(backbone).eval()
    inputs = torch.tensor([[1, 2, 3], [4, 5, 6]])
    output = model(inputs)
    expected = backbone.head(backbone.embedding(inputs))[:, -1, :]
    torch.testing.assert_close(output, expected)
    assert output.shape == (2, 7)
    assert backbone.config.use_cache is False
    use_cache, return_dict, logits_to_keep, attention = backbone.calls[-1]
    assert use_cache is False and return_dict is True and logits_to_keep == 1
    assert torch.equal(attention, torch.ones_like(inputs))


def test_model_wrapper_rejects_active_dropout() -> None:
    with pytest.raises(RuntimeError, match="active dropout|nonzero hidden_dropout"):
        PythiaNextTokenModel(FakeBackbone(dropout=0.2))


def test_local_model_loader_forces_offline_eager_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, dict[str, Any]] = {}

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> Any:
            calls["tokenizer"] = {"path": path, **kwargs}
            return SimpleNamespace(vocab_size=7)

    class AutoModel:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> Any:
            calls["model"] = {"path": path, **kwargs}
            return FakeBackbone(), {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }

    module = types.ModuleType("transformers")
    module.AutoTokenizer = AutoTokenizer
    module.AutoModelForCausalLM = AutoModel
    monkeypatch.setitem(sys.modules, "transformers", module)
    model, tokenizer = load_local_pythia(tmp_path, dtype="float32")
    assert isinstance(model, PythiaNextTokenModel)
    assert tokenizer.vocab_size == 7
    assert calls["tokenizer"]["local_files_only"] is True
    assert calls["model"]["local_files_only"] is True
    assert calls["model"]["attn_implementation"] == "eager"
    assert calls["model"]["torch_dtype"] is torch.float32
    assert calls["model"]["output_loading_info"] is True


def test_next_token_metrics_report_nll_perplexity_and_topk() -> None:
    logits = torch.tensor(
        [[5.0, 1.0, 0.0, -1.0, -2.0, -3.0], [0.0, 2.0, 3.0, 1.0, -1.0, -2.0]]
    )
    targets = torch.tensor([0, 1])
    sums = next_token_metric_sums(logits, targets)
    assert sums["count"] == 2
    assert sums["top1_correct"] == 1
    assert sums["top5_correct"] == 2
    assert sums["nll_sum"] == pytest.approx(
        float(F.cross_entropy(logits, targets, reduction="sum"))
    )


def test_permutation_gradient_expectation_preserves_token_marginal() -> None:
    torch.manual_seed(5)
    model = torch.nn.Linear(3, 3, bias=False).double().eval()
    inputs = torch.tensor(
        [[1.0, 0.0, 0.2], [0.0, 1.0, -0.3], [0.5, 0.4, 1.0], [-0.2, 0.3, 0.7]],
        dtype=torch.float64,
    )
    targets = torch.tensor([0, 1, 1, 2])
    gradients = []
    for permutation in itertools.permutations(range(len(targets))):
        permuted = targets[torch.tensor(permutation)]
        gradients.append(_gradient(model, inputs, permuted)[0]["weight"])
    empirical = torch.stack(gradients).mean(dim=0)
    logits = model(inputs)
    marginal = F.one_hot(targets, num_classes=3).double().mean(dim=0)
    marginal_loss = -(marginal[None, :] * F.log_softmax(logits, dim=-1)).sum(-1).mean()
    expected = torch.autograd.grad(marginal_loss, model.weight)[0]
    torch.testing.assert_close(empirical, expected, rtol=1e-12, atol=1e-12)


class MutatingBufferModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(3, 5))
        self.register_buffer("calls", torch.zeros((), dtype=torch.long))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.calls.add_(1)
        return inputs.float() @ self.weight


def test_audit_fails_closed_if_a_virtual_branch_mutates_state() -> None:
    model = MutatingBufferModel().eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    packets = [
        NextTokenPacket(
            input_ids=torch.randint(0, 3, (4, 3)),
            targets=np.asarray([0, 1, 2, 3]),
            ids=tuple(f"{packet}-{row}" for row in range(4)),
            timestamps=tuple(f"2022-01-{packet * 4 + row + 1:02d}" for row in range(4)),
            packet_number=packet,
        )
        for packet in range(3)
    ]
    with pytest.raises(RuntimeError, match="mutated the live model"):
        run_next_token_audit(
            model=model,
            optimizer=optimizer,
            packets=packets,
            n_permutations=2,
            permutation_salt=3,
            seed=4,
            device="cpu",
            radius_history=[0.01],
            radius_window=2,
            fallback_radius=0.01,
            gradient_clip=1.0,
            bootstrap_draws=32,
            bootstrap_block_length=1,
        )


class TinyTokenPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 6)
        self.head = torch.nn.Linear(6, 11)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(inputs).mean(dim=1))


class CountingTokenPredictor(TinyTokenPredictor):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return super().forward(inputs)


class TargetAccessSentinelPacket:
    def __init__(
        self,
        *,
        model: CountingTokenPredictor,
        minimum_forward_calls: int,
        packet_number: int,
    ) -> None:
        self.input_ids = (torch.arange(20).reshape(4, 5) + packet_number).remainder(11)
        self.ids = tuple(f"sentinel-{packet_number}-{row}" for row in range(4))
        self.timestamps = tuple(
            f"2022-06-{packet_number * 4 + row + 1:02d}" for row in range(4)
        )
        self.packet_number = packet_number
        self._model = model
        self._minimum_forward_calls = minimum_forward_calls
        self._targets = np.asarray(
            [(packet_number + row * 3) % 11 for row in range(4)], dtype=np.int64
        )
        self.reveal_count = 0

    @property
    def targets(self) -> np.ndarray:
        if self._model.forward_calls < self._minimum_forward_calls:
            raise AssertionError("current targets were accessed before branch logits")
        self.reveal_count += 1
        return self._targets


def test_current_targets_are_revealed_only_after_all_virtual_branch_logits() -> None:
    torch.manual_seed(41)
    model = CountingTokenPredictor().eval()
    packets = [
        TargetAccessSentinelPacket(
            model=model,
            minimum_forward_calls=minimum,
            packet_number=packet,
        )
        for packet, minimum in enumerate((1, 8, 15))
    ]
    result = run_next_token_audit(
        model=model,
        optimizer=torch.optim.AdamW(
            model.parameters(), lr=0.002, foreach=False, fused=False
        ),
        packets=packets,
        n_permutations=2,
        permutation_salt=19,
        seed=7,
        device="cpu",
        radius_history=[0.01],
        radius_window=2,
        fallback_radius=0.01,
        gradient_clip=1.0,
        bootstrap_draws=32,
        bootstrap_block_length=1,
    )
    assert [packet.reveal_count for packet in packets] == [1, 1, 1]
    assert result["audit"]["strict_prelabel_branch_order"] is True
    assert result["audit"]["observed_target_reveal_events"] == 3


def test_four_replica_audit_reports_nested_prefixes_and_all_contrast_intervals() -> None:
    torch.manual_seed(43)
    model = TinyTokenPredictor().eval()
    packets = [
        NextTokenPacket(
            input_ids=(torch.arange(20).reshape(4, 5) + packet).remainder(11),
            targets=np.asarray([(packet + row * 2) % 11 for row in range(4)]),
            ids=tuple(f"prefix-{packet}-{row}" for row in range(4)),
            timestamps=tuple(
                f"2022-07-{packet * 4 + row + 1:02d}" for row in range(4)
            ),
            packet_number=packet,
        )
        for packet in range(3)
    ]
    result = run_next_token_audit(
        model=model,
        optimizer=torch.optim.AdamW(
            model.parameters(), lr=0.002, foreach=False, fused=False
        ),
        packets=packets,
        n_permutations=4,
        permutation_salt=23,
        seed=11,
        device="cpu",
        radius_history=[0.01],
        radius_window=2,
        fallback_radius=0.01,
        gradient_clip=1.0,
        bootstrap_draws=32,
        bootstrap_block_length=1,
    )
    sensitivity = result["permutation_prefix_sensitivity"]
    assert sensitivity["nested_prefix_sizes"] == [1, 2, 4]
    assert sensitivity["task_metrics"]["4"] == result["task_metrics"]["K4"]
    intervals = result["inference"]["paired_circular_block_intervals"]
    assert intervals["K1_vs_K0"]["nll"]["estimate"] == pytest.approx(
        result["effects"]["gross_nll_reduction"]
    )
    assert intervals["K1_vs_K4"]["top1_accuracy"]["estimate"] == pytest.approx(
        result["effects"]["pairing_top1_accuracy"]
    )
    alignment = result["distribution_alignment"]
    assert alignment["generated_packets"] == 3
    assert alignment["scored_source_packets"] == 2
    assert alignment["excluded_terminal_unscored_packets"] == 1
    assert [
        value["used_by_subsequent_scored_branch"]
        for value in result["distribution_records"]
    ] == [True, True, False]


def test_live_clipping_cannot_mutate_the_stored_raw_gradient() -> None:
    torch.manual_seed(47)
    model = TinyTokenPredictor().eval()
    packets = [
        NextTokenPacket(
            input_ids=(torch.arange(20).reshape(4, 5) + packet).remainder(11),
            targets=np.asarray([(packet + row * 2) % 11 for row in range(4)]),
            ids=tuple(f"clip-{packet}-{row}" for row in range(4)),
            timestamps=tuple(
                f"2022-08-{packet * 4 + row + 1:02d}" for row in range(4)
            ),
            packet_number=packet,
        )
        for packet in range(3)
    ]
    result = run_next_token_audit(
        model=model,
        optimizer=torch.optim.AdamW(
            model.parameters(), lr=0.002, foreach=False, fused=False
        ),
        packets=packets,
        n_permutations=2,
        permutation_salt=29,
        seed=13,
        device="cpu",
        radius_history=[0.01],
        radius_window=2,
        fallback_radius=0.01,
        gradient_clip=1.0e-12,
        bootstrap_draws=32,
        bootstrap_block_length=1,
    )
    records = result["distribution_records"]
    assert all(value["raw_true_gradient_preserved_exactly"] for value in records)
    assert all(
        value["stored_true_gradient_norm_after_live_clip"]
        == value["live_true_gradient_norm_before_clip"]
        for value in records
    )
    assert all(
        value["live_true_gradient_norm_after_clip"]
        < value["live_true_gradient_norm_before_clip"]
        for value in records
    )
    assert result["audit"][
        "raw_true_loss_gradient_preserved_before_live_clipping"
    ] is True


class ZeroGradientPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(11))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return (self.anchor * 0.0).expand(inputs.shape[0], -1)


def test_full_protocol_fails_closed_on_a_zero_norm_actuator_branch() -> None:
    model = ZeroGradientPredictor().eval()
    packets = [
        NextTokenPacket(
            input_ids=torch.zeros((4, 5), dtype=torch.long),
            targets=np.asarray([0, 1, 2, 3]),
            ids=tuple(f"zero-{packet}-{row}" for row in range(4)),
            timestamps=tuple(
                f"2022-09-{packet * 4 + row + 1:02d}" for row in range(4)
            ),
            packet_number=packet,
        )
        for packet in range(3)
    ]
    with pytest.raises(RuntimeError, match="zero-norm actuator branch"):
        run_next_token_audit(
            model=model,
            optimizer=torch.optim.AdamW(
                model.parameters(), lr=0.002, foreach=False, fused=False
            ),
            packets=packets,
            n_permutations=2,
            permutation_salt=31,
            seed=17,
            device="cpu",
            radius_history=[0.01],
            radius_window=2,
            fallback_radius=0.01,
            gradient_clip=1.0,
            bootstrap_draws=32,
            bootstrap_block_length=1,
            fail_on_zero_norm_actuator_branch=True,
        )


def test_permutation_seed_cannot_change_the_true_live_trajectory() -> None:
    torch.manual_seed(29)
    initial = TinyTokenPredictor().eval()
    packets = [
        NextTokenPacket(
            input_ids=(torch.arange(20).reshape(4, 5) + packet).remainder(11),
            targets=np.asarray([(packet + row * 2) % 11 for row in range(4)]),
            ids=tuple(f"unit-{packet}-{row}" for row in range(4)),
            timestamps=tuple(
                f"2022-03-{packet * 4 + row + 1:02d}" for row in range(4)
            ),
            packet_number=packet,
        )
        for packet in range(4)
    ]

    def run(seed: int) -> dict[str, Any]:
        model = copy.deepcopy(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
        return run_next_token_audit(
            model=model,
            optimizer=optimizer,
            packets=packets,
            n_permutations=2,
            permutation_salt=701,
            seed=seed,
            device="cpu",
            radius_history=[0.01],
            radius_window=2,
            fallback_radius=0.01,
            gradient_clip=1.0,
            bootstrap_draws=32,
            bootstrap_block_length=1,
        )

    first = run(3)
    second = run(97)
    assert first["final_live_model_sha256"] == second["final_live_model_sha256"]
    assert first["task_metrics"]["K0"] == second["task_metrics"]["K0"]
    assert first["task_metrics"]["K1"] == second["task_metrics"]["K1"]


def test_cpu_tiny_smoke_is_offline_deterministic_and_paper_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_network(*_: Any, **__: Any) -> Any:
        raise AssertionError("CPU smoke attempted network access")

    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    first = run_tiny_smoke(17)
    second = run_tiny_smoke(17)
    assert first == second
    assert first["status"] == "PASS"
    assert first["network_access"] is False
    assert first["paper_evidence_eligible"] is False
    assert first["source_provenance"]["validation_status"] == "NOT_PROVIDED"
    assert first["source_provenance"]["source_manifest_sha256"] is None
    assert first["unit"]["source_provenance"] == first["source_provenance"]
    assert first["stream"]["conditioning_tokens_are_never_permuted"] is True
    unit = first["unit"]
    assert unit["audit"]["all_pass"] is True
    assert unit["audit"]["live_trajectory_uses_true_targets_only"] is True
    assert unit["audit"]["predicted_targets_only_are_permuted"] is True
    assert unit["distribution_alignment"]["target_multiset_preserved_all_packets"] is True
    assert set(unit["task_metrics"]) == {"K0", "K1", "K4"}
    assert all("perplexity" in value for value in unit["task_metrics"].values())
    intervals = unit["inference"]["paired_circular_block_intervals"]
    assert set(intervals) == {"K1_vs_K0", "K4_vs_K0", "K1_vs_K4"}
    assert all(
        set(values) == {"nll", "top1_accuracy", "top5_accuracy"}
        for values in intervals.values()
    )
    assert "power" not in unit
    assert unit["permutation_prefix_sensitivity"]["nested_prefix_sizes"] == [1, 2]
    assert unit["actuator"]["name"] == ACTUATOR_NAME
    assert unit["actuator"]["not_an_adamw_counterfactual"] is True
    assert unit["optimizer_implementation"]["foreach"] is False
    assert unit["optimizer_implementation"]["fused"] is False
    mc = unit["inference"]["permutation_outcome_mc_standard_error"]
    assert set(mc) >= {"nll", "top1_accuracy", "top5_accuracy"}
    assert unit["distribution_alignment"]["minimum_unique_index_assignments"] >= 1


def test_smoke_cli_writes_one_atomic_result_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    output = tmp_path / "smoke.json"
    assert main(["--smoke-tiny", "--seed", "23", "--out", str(output)]) == 0
    value = json.loads(output.read_text())
    assert value["status"] == "PASS"
    assert value["paper_evidence_eligible"] is False
    assert value["source_provenance"]["validation_status"] == "NOT_PROVIDED"
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        main(["--smoke-tiny", "--seed", "23", "--out", str(output)])


def test_full_cli_rejects_an_ad_hoc_seed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="seed is smoke-only"):
        main(["--full", "--seed", "301", "--out", str(tmp_path / "full.json")])


def test_cli_requires_an_explicit_mode_and_advertises_full(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        main(["--out", str(tmp_path / "implicit-full.json")])
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    assert "--full" in help_text
    assert "--replace-existing-heartbeat" not in help_text


def test_transformer_source_freeze_is_required_and_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="source-manifest is required"):
        validate_transformer_source_manifest(
            None, config_path=DEFAULT_CONFIG, required=True
        )

    source_manifest = _write_transformer_source_manifest(tmp_path)
    binding = validate_transformer_source_manifest(
        source_manifest, config_path=DEFAULT_CONFIG, required=True
    )
    assert binding["experiment"] == "transformer"
    assert binding["validation_status"] == "PASS"
    assert binding["source_manifest_sha256"] == sha256_file(source_manifest)
    assert len(binding["source_bundle_sha256"]) == 64
    assert binding["configuration_sha256"] == sha256_file(DEFAULT_CONFIG)

    payload = json.loads(source_manifest.read_text())
    payload["files"][0]["sha256"] = "0" * 64
    source_manifest.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="failed validation"):
        validate_transformer_source_manifest(
            source_manifest, config_path=DEFAULT_CONFIG, required=True
        )


def test_environment_manifest_binds_patch_packages_numerics_and_optimizer(
    tmp_path: Path,
) -> None:
    config = read_config(DEFAULT_CONFIG)
    source = {
        "schema_version": "gradient_reuse.transformer_source_binding.v1",
        "experiment": "transformer",
        "validation_status": "PASS",
        "source_manifest_sha256": "1" * 64,
        "source_bundle_sha256": "2" * 64,
        "configuration_sha256": sha256_file(DEFAULT_CONFIG),
    }
    observed = capture_runtime_environment(
        device="cpu", optimizer_specification=config["optimization"]
    )
    manifest = build_environment_manifest(
        config_sha256=sha256_file(DEFAULT_CONFIG),
        source_provenance=source,
        observed=observed,
    )
    path = tmp_path / "environment.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    binding = validate_environment_manifest(
        path,
        expected_config_sha256=sha256_file(DEFAULT_CONFIG),
        expected_source_provenance=source,
        observed=observed,
    )
    assert binding["manifest_sha256"] == sha256_file(path)
    assert binding["observed"]["python"]["version"].count(".") == 2
    assert set(binding["observed"]["packages"]) >= {
        "transformers",
        "tokenizers",
        "safetensors",
        "huggingface_hub",
        "pyarrow",
    }
    assert binding["observed"]["optimizer"] == {
        "name": "AdamW",
        "foreach": False,
        "fused": False,
        "counterfactual_actuator": ACTUATOR_NAME,
    }

    changed = copy.deepcopy(observed)
    changed["numerics"]["float32_matmul_precision"] = "drifted"
    with pytest.raises(RuntimeError, match="current runtime differs"):
        validate_environment_manifest(
            path,
            expected_config_sha256=sha256_file(DEFAULT_CONFIG),
            expected_source_provenance=source,
            observed=changed,
        )


def test_cuda_preflight_report_is_bound_and_memory_gated(tmp_path: Path) -> None:
    config = read_config(DEFAULT_CONFIG)
    source = {
        "schema_version": "gradient_reuse.transformer_source_binding.v1",
        "experiment": "transformer",
        "validation_status": "PASS",
        "source_manifest_sha256": "1" * 64,
        "source_bundle_sha256": "2" * 64,
        "configuration_sha256": sha256_file(DEFAULT_CONFIG),
    }
    environment = {
        "schema_version": "fmmt.gradient_reuse_transformer.environment_binding.v1",
        "manifest_sha256": "3" * 64,
        "environment_fingerprint_sha256": "4" * 64,
        "observed": {"accelerator": {"gpu": {"total_memory_bytes": 34_000_000_000}}},
    }
    assets = {
        "model": {
            "repository_id": "org/model",
            "resolved_revision": "a" * 40,
            "sha256": "5" * 64,
        },
        "dataset": {
            "repository_id": "org/data",
            "resolved_revision": "b" * 40,
            "sha256": "6" * 64,
        },
    }
    source_path = tmp_path / "source.json"
    environment_path = tmp_path / "environment.json"
    asset_path = tmp_path / "assets.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    environment_path.write_text(json.dumps(environment), encoding="utf-8")
    asset_path.write_text(json.dumps({"assets": assets}), encoding="utf-8")
    input_bundle = build_validated_input_bundle(
        {
            "configuration": DEFAULT_CONFIG,
            "source_manifest": source_path,
            "environment_manifest": environment_path,
            "asset_manifest": asset_path,
        }
    )
    validate_embedded_input_bundle(input_bundle)
    payload = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "PASS",
        "paper_evidence_eligible": False,
        "purpose": "resource and integration validation only",
        "config_sha256": sha256_file(DEFAULT_CONFIG),
        "source_provenance": source,
        "environment_provenance": environment,
        "assets": assets,
        "validated_input_bundle": input_bundle,
        "workload": {
            "warmup_packets": 2,
            "online_packets": 3,
            "packet_size": 16,
            "context_length": 128,
            "n_permutations": 4,
            "full_model_and_assets": True,
        },
        "resources": {
            "wall_time_seconds": 12.0,
            "peak_allocated_bytes": 20_000_000_000,
            "peak_reserved_bytes": 22_000_000_000,
            "total_device_memory_bytes": 34_000_000_000,
            "maximum_peak_reserved_bytes": config["cuda_preflight"][
                "maximum_peak_reserved_bytes"
            ],
        },
        "stream": {"schema_version": "fmmt.fixed_conditioning_stream.v1"},
        "resume_supported": False,
        "unit": {
            "audit": {
                "all_pass": True,
                "zero_norm_actuator_branches": 0,
                "zero_norm_actuator_branch_fail_closed": True,
            }
        },
    }

    def seal(value: dict[str, Any]) -> dict[str, Any]:
        value.pop("artifact_inventory", None)
        value["artifact_inventory"] = _artifact_inventory(
            {
                "result_payload_without_inventory": value,
                "validated_input_bundle": value["validated_input_bundle"],
                "stream": value["stream"],
                "resources": value["resources"],
                "unit": value["unit"],
            }
        )
        return value

    seal(payload)
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    binding = validate_cuda_preflight_report(
        path,
        config=config,
        expected_config_sha256=sha256_file(DEFAULT_CONFIG),
        expected_source_provenance=source,
        expected_environment_provenance=environment,
        expected_assets=assets,
        expected_input_bundle=input_bundle,
    )
    assert binding["report_sha256"] == sha256_file(path)
    assert binding["reserved_memory_headroom_bytes"] > 0

    over = copy.deepcopy(payload)
    over["resources"]["peak_reserved_bytes"] = (
        config["cuda_preflight"]["maximum_peak_reserved_bytes"] + 1
    )
    seal(over)
    path.write_text(json.dumps(over))
    with pytest.raises(RuntimeError, match="memory ceiling"):
        validate_cuda_preflight_report(
            path,
            config=config,
            expected_config_sha256=sha256_file(DEFAULT_CONFIG),
            expected_source_provenance=source,
            expected_environment_provenance=environment,
            expected_assets=assets,
            expected_input_bundle=input_bundle,
        )

    mismatched = copy.deepcopy(payload)
    mismatched["assets"]["model"]["sha256"] = "0" * 64
    seal(mismatched)
    path.write_text(json.dumps(mismatched))
    with pytest.raises(RuntimeError, match="different external assets"):
        validate_cuda_preflight_report(
            path,
            config=config,
            expected_config_sha256=sha256_file(DEFAULT_CONFIG),
            expected_source_provenance=source,
            expected_environment_provenance=environment,
            expected_assets=assets,
            expected_input_bundle=input_bundle,
        )

    wrong_bundle = copy.deepcopy(input_bundle)
    wrong_bundle["payloads"]["source_manifest"]["drift"] = True
    wrong_bundle["inventory"]["source_manifest"][
        "canonical_payload_sha256"
    ] = "0" * 64
    mismatched_bundle = copy.deepcopy(payload)
    mismatched_bundle["validated_input_bundle"] = wrong_bundle
    seal(mismatched_bundle)
    path.write_text(json.dumps(mismatched_bundle))
    with pytest.raises(RuntimeError, match="input-bundle hash is invalid"):
        validate_cuda_preflight_report(
            path,
            config=config,
            expected_config_sha256=sha256_file(DEFAULT_CONFIG),
            expected_source_provenance=source,
            expected_environment_provenance=environment,
            expected_assets=assets,
            expected_input_bundle=input_bundle,
        )


def test_full_run_rejects_assets_from_another_source_freeze(tmp_path: Path) -> None:
    source_manifest = _write_transformer_source_manifest(tmp_path)
    binding = validate_transformer_source_manifest(
        source_manifest, config_path=DEFAULT_CONFIG, required=True
    )
    asset_manifest = tmp_path / "assets.json"
    asset_manifest.write_text(
        json.dumps({"source_provenance": binding}), encoding="utf-8"
    )
    _validate_asset_source_binding(asset_manifest, binding)

    mismatched = copy.deepcopy(binding)
    mismatched["source_bundle_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="source provenance differs"):
        _validate_asset_source_binding(asset_manifest, mismatched)


def test_smoke_records_a_validated_source_freeze_but_stays_ineligible(
    tmp_path: Path,
) -> None:
    source_manifest = _write_transformer_source_manifest(tmp_path)
    result = run_tiny_smoke(19, source_manifest=source_manifest)
    assert result["paper_evidence_eligible"] is False
    assert result["source_provenance"]["validation_status"] == "PASS"
    assert result["source_provenance"]["source_manifest_sha256"] == sha256_file(
        source_manifest
    )
    assert result["unit"]["source_provenance"] == result["source_provenance"]


def test_full_cli_requires_source_freeze_before_execution(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="source-manifest is required"):
        main(
            [
                "--full",
                "--asset-manifest",
                str(tmp_path / "assets.json"),
                "--out",
                str(tmp_path / "full.json"),
            ]
        )


def test_real_asset_cli_requires_environment_manifest(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="environment-manifest is required"):
        main(
            [
                "--full",
                "--asset-manifest",
                str(tmp_path / "assets.json"),
                "--source-manifest",
                str(tmp_path / "source.json"),
                "--out",
                str(tmp_path / "full.json"),
            ]
        )


def test_full_cli_requires_and_wires_preflight_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.gradient_reuse_transformer.run as transformer_run

    paths = _write_cli_identity_files(tmp_path)
    common = [
        "--full",
        "--asset-manifest",
        str(paths["asset_manifest"]),
        "--source-manifest",
        str(paths["source_manifest"]),
        "--environment-manifest",
        str(paths["environment_manifest"]),
    ]
    with pytest.raises(RuntimeError, match="preflight-report is required"):
        main([*common, "--out", str(tmp_path / "missing-preflight.json")])

    captured: dict[str, Any] = {}

    def fake_full(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        kwargs["progress_callback"]({"phase": "online_audit", "completed_packets": 4})
        bundle = build_validated_input_bundle(
            {
                "configuration": kwargs["config_path"],
                "source_manifest": kwargs["source_manifest"],
                "environment_manifest": kwargs["environment_manifest"],
                "asset_manifest": kwargs["asset_manifest"],
                "cuda_preflight_report": kwargs["preflight_report"],
            }
        )
        return {
            "status": "PASS",
            "paper_evidence_eligible": True,
            "validated_input_bundle": bundle,
        }

    monkeypatch.setattr(transformer_run, "run_full", fake_full)
    report = paths["preflight_report"]
    output = tmp_path / "full.json"
    assert main(
        [
            *common,
            "--preflight-report",
            str(report),
            "--out",
            str(output),
        ]
    ) == 0
    assert captured["preflight_report"] == str(report)
    assert json.loads(output.read_text())["paper_evidence_eligible"] is True


def test_cli_paths_are_disjoint_and_existing_heartbeat_is_never_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.gradient_reuse_transformer.run as transformer_run

    with pytest.raises(RuntimeError, match="path roles must be disjoint"):
        main(
            [
                "--smoke-tiny",
                "--config",
                str(tmp_path / "same.json"),
                "--out",
                str(tmp_path / "same.json"),
            ]
        )

    paths = _write_cli_identity_files(tmp_path)

    def fake_preflight(**kwargs: Any) -> dict[str, Any]:
        kwargs["progress_callback"]({"phase": "online_audit", "completed_packets": 3})
        bundle = build_validated_input_bundle(
            {
                "configuration": kwargs["config_path"],
                "source_manifest": kwargs["source_manifest"],
                "environment_manifest": kwargs["environment_manifest"],
                "asset_manifest": kwargs["asset_manifest"],
            }
        )
        return {
            "schema_version": PREFLIGHT_SCHEMA,
            "status": "PASS",
            "paper_evidence_eligible": False,
            "validated_input_bundle": bundle,
        }

    monkeypatch.setattr(transformer_run, "run_cuda_preflight", fake_preflight)
    common = [
        "--cuda-preflight",
        "--asset-manifest",
        str(paths["asset_manifest"]),
        "--source-manifest",
        str(paths["source_manifest"]),
        "--environment-manifest",
        str(paths["environment_manifest"]),
    ]
    heartbeat = tmp_path / "shared.heartbeat.json"
    assert main(
        [
            *common,
            "--heartbeat",
            str(heartbeat),
            "--out",
            str(tmp_path / "preflight-one.json"),
        ]
    ) == 0
    completed_heartbeat = heartbeat.read_bytes()
    with pytest.raises(RuntimeError, match="refusing to overwrite existing heartbeat"):
        main(
            [
                *common,
                "--heartbeat",
                str(heartbeat),
                "--out",
                str(tmp_path / "preflight-two.json"),
            ]
        )
    assert heartbeat.read_bytes() == completed_heartbeat
    assert not (tmp_path / "preflight-two.json").exists()


def test_cli_rejects_output_and_heartbeat_inside_hashed_asset_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.gradient_reuse_transformer.run as transformer_run

    paths = _write_cli_identity_files(tmp_path)

    def should_not_run(**_: Any) -> dict[str, Any]:
        raise AssertionError("path rejection must occur before the preflight")

    monkeypatch.setattr(transformer_run, "run_cuda_preflight", should_not_run)
    common = [
        "--cuda-preflight",
        "--asset-manifest",
        str(paths["asset_manifest"]),
        "--source-manifest",
        str(paths["source_manifest"]),
        "--environment-manifest",
        str(paths["environment_manifest"]),
    ]
    with pytest.raises(RuntimeError, match="output must not equal or descend"):
        main(
            [
                *common,
                "--out",
                str(tmp_path / "model" / "result.json"),
            ]
        )
    with pytest.raises(RuntimeError, match="heartbeat must not equal or descend"):
        main(
            [
                *common,
                "--heartbeat",
                str(tmp_path / "dataset" / "progress.json"),
                "--out",
                str(tmp_path / "safe-result.json"),
            ]
        )


def test_cuda_preflight_cli_writes_nonresumable_heartbeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.gradient_reuse_transformer.run as transformer_run

    paths = _write_cli_identity_files(tmp_path)

    def fake_preflight(**kwargs: Any) -> dict[str, Any]:
        kwargs["progress_callback"](
            {"phase": "online_audit", "completed_packets": 3}
        )
        bundle = build_validated_input_bundle(
            {
                "configuration": kwargs["config_path"],
                "source_manifest": kwargs["source_manifest"],
                "environment_manifest": kwargs["environment_manifest"],
                "asset_manifest": kwargs["asset_manifest"],
            }
        )
        return {
            "schema_version": "fmmt.gradient_reuse_transformer.cuda_preflight.v1",
            "status": "PASS",
            "paper_evidence_eligible": False,
            "validated_input_bundle": bundle,
        }

    monkeypatch.setattr(transformer_run, "run_cuda_preflight", fake_preflight)
    output = tmp_path / "preflight.json"
    heartbeat = tmp_path / "preflight.heartbeat.json"
    assert main(
        [
            "--cuda-preflight",
            "--asset-manifest",
            str(paths["asset_manifest"]),
            "--source-manifest",
            str(paths["source_manifest"]),
            "--environment-manifest",
            str(paths["environment_manifest"]),
            "--heartbeat",
            str(heartbeat),
            "--out",
            str(output),
        ]
    ) == 0
    assert json.loads(output.read_text())["paper_evidence_eligible"] is False
    progress = json.loads(heartbeat.read_text())
    assert progress["status"] == "COMPLETE"
    assert progress["resume_supported"] is False
    assert progress["completed_packets"] == 3
    assert progress["output_file_sha256"] == sha256_file(output)
    assert len(progress["result_payload_sha256"]) == 64
    invocation = progress["invocation"]
    assert invocation["mode"] == "cuda_preflight"
    assert invocation["input_bundle_sha256"] == json.loads(output.read_text())[
        "validated_input_bundle"
    ]["bundle_sha256"]
    assert set(invocation["input_inventory"]) == {
        "configuration",
        "source_manifest",
        "environment_manifest",
        "asset_manifest",
    }


def test_cli_fails_closed_if_an_input_manifest_changes_during_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import experiments.gradient_reuse_transformer.run as transformer_run

    paths = _write_cli_identity_files(tmp_path)

    def mutating_preflight(**kwargs: Any) -> dict[str, Any]:
        bundle = build_validated_input_bundle(
            {
                "configuration": kwargs["config_path"],
                "source_manifest": kwargs["source_manifest"],
                "environment_manifest": kwargs["environment_manifest"],
                "asset_manifest": kwargs["asset_manifest"],
            }
        )
        Path(kwargs["source_manifest"]).write_text(
            json.dumps({"schema_version": "test.source.changed.v1"}),
            encoding="utf-8",
        )
        return {
            "schema_version": PREFLIGHT_SCHEMA,
            "status": "PASS",
            "paper_evidence_eligible": False,
            "validated_input_bundle": bundle,
        }

    monkeypatch.setattr(transformer_run, "run_cuda_preflight", mutating_preflight)
    output = tmp_path / "mutated-result.json"
    heartbeat = tmp_path / "mutated-result.heartbeat.json"
    with pytest.raises(RuntimeError, match="input changed during"):
        main(
            [
                "--cuda-preflight",
                "--asset-manifest",
                str(paths["asset_manifest"]),
                "--source-manifest",
                str(paths["source_manifest"]),
                "--environment-manifest",
                str(paths["environment_manifest"]),
                "--heartbeat",
                str(heartbeat),
                "--out",
                str(output),
            ]
        )
    assert not output.exists()
    failed = json.loads(heartbeat.read_text())
    assert failed["status"] == "FAILED"
    assert failed["invocation"]["input_inventory"]["source_manifest"][
        "file_sha256"
    ] != sha256_file(paths["source_manifest"])
