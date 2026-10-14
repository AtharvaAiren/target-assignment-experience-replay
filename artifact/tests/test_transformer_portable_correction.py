"""Behavioral tests for the portable Transformer rule implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments.gradient_reuse_reviewer_extension import transformer_bootstrap
from experiments.gradient_reuse_reviewer_extension import transformer_correction as fix
from experiments.gradient_reuse_reviewer_extension import transformer_stream2 as e7


class _AddedToken:
    def __init__(self, content: str, *, special: bool) -> None:
        self.content = content
        self.special = special
        self.normalized = not special
        self.single_word = False
        self.lstrip = False
        self.rstrip = False


class _Backend:
    def get_vocab_size(self, *, with_added_tokens: bool) -> int:
        return fix.EFFECTIVE_VOCAB_SIZE if with_added_tokens else fix.BASE_VOCAB_SIZE


class _Tokenizer:
    vocab_size = fix.BASE_VOCAB_SIZE
    bos_token_id = 0
    eos_token_id = 0
    pad_token_id = 1
    unk_token_id = 0

    def __init__(self, root: Path, emitted: list[int]) -> None:
        self.name_or_path = str(root)
        self.emitted = emitted
        self.backend_tokenizer = _Backend()
        self._added = dict(fix.EXPECTED_ADDED_VOCAB)
        self.added_tokens_decoder = {
            index: _AddedToken(token, special=index in {0, 1})
            for token, index in self._added.items()
        }
        self._vocab = {
            "<|endoftext|>": 0,
            "<|padding|>": 1,
            **{f"base_{index}": index for index in range(2, fix.BASE_VOCAB_SIZE)},
            **dict(fix.EXPECTED_WHITESPACE_ADDED),
        }

    def __len__(self) -> int:
        return fix.EFFECTIVE_VOCAB_SIZE

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added)

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": list(self.emitted)}


def _tokenizer(tmp_path: Path, emitted: list[int]) -> tuple[_Tokenizer, str]:
    content = b'{"portable-test":true}\n'
    (tmp_path / "tokenizer.json").write_bytes(content)
    return _Tokenizer(tmp_path, emitted), hashlib.sha256(content).hexdigest()


def _gpt_neox() -> torch.nn.Module:
    cls = type("GPTNeoXForCausalLM", (torch.nn.Module,), {})
    model = cls()
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        model_type="gpt_neox",
        architectures=["GPTNeoXForCausalLM"],
        classifier_dropout=0.1,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    model.projection = torch.nn.Linear(3, 3)
    return model


def _historical_validator(_: torch.nn.Module) -> None:
    raise RuntimeError("model config has nonzero classifier_dropout=0.1")


def test_classifier_dropout_exception_is_narrow_and_checks_live_modules() -> None:
    model = _gpt_neox()
    fix.validate_classifier_dropout_compatibility(model, _historical_validator)

    model.dropout = torch.nn.Dropout(0.2)
    with pytest.raises(RuntimeError, match="active dropout"):
        fix.validate_classifier_dropout_compatibility(model, _historical_validator)

    model = _gpt_neox()
    model.classifier = torch.nn.Linear(3, 2)
    with pytest.raises(RuntimeError, match="classifier module"):
        fix.validate_classifier_dropout_compatibility(model, _historical_validator)


def test_padded_vocabulary_identity_preserves_full_model_softmax() -> None:
    fix.validate_padded_vocabulary_identity(
        tokenizer_vocab=50_254,
        tokenizer_length=50_277,
        model_vocab=50_304,
        input_rows=50_304,
        output_rows=50_304,
        special_ids={"bos": 0, "eos": 0, "pad": 1, "unk": 0},
    )
    with pytest.raises(RuntimeError, match="padded-vocabulary identity changed"):
        fix.validate_padded_vocabulary_identity(
            tokenizer_vocab=50_254,
            tokenizer_length=50_277,
            model_vocab=50_304,
            input_rows=50_304,
            output_rows=50_277,
            special_ids={"bos": 0, "eos": 0, "pad": 1, "unk": 0},
        )


def test_effective_vocabulary_accepts_added_ids_and_rejects_padding(
    tmp_path: Path,
) -> None:
    tokenizer, tokenizer_hash = _tokenizer(tmp_path, [0, 50_254, 50_276])
    effective = fix.validate_effective_tokenizer_identity(
        tokenizer, tokenizer_json_sha256=tokenizer_hash
    )
    assert effective == frozenset(range(50_277))

    # The successful explicit validation populates the same per-object cache
    # used by the runtime tokenization helper.
    assert fix.token_ids_with_effective_vocab(tokenizer, "probe") == [
        0,
        50_254,
        50_276,
    ]
    tokenizer.emitted = [50_277]
    with pytest.raises(RuntimeError, match="outside its authenticated"):
        fix.token_ids_with_effective_vocab(tokenizer, "probe")
    fix._VALIDATED_TOKENIZERS.clear()


def test_public_manifest_authenticates_portable_bytes_and_records_history() -> None:
    provenance = e7.authenticate_correction_files(
        e7.PORTABLE_CORRECTION_MANIFEST,
        e7.PORTABLE_CORRECTION_PATCH,
    )
    assert provenance["schema_version"] == (
        e7.PORTABLE_CORRECTION_PROVENANCE_SCHEMA
    )
    assert provenance["status"] == "PORTABLE_BYTES_AUTHENTICATED"
    assert provenance["executed_manifest_sha256"] == (
        e7.EXPECTED_PORTABLE_CORRECTION_MANIFEST_SHA256
    )
    assert provenance["executed_patch_sha256"] == (
        e7.EXPECTED_PORTABLE_CORRECTION_PATCH_SHA256
    )
    assert provenance["historical_effective_vocab_manifest_sha256"] == (
        e7.EXPECTED_CORRECTION_MANIFEST_SHA256
    )
    assert provenance["historical_effective_vocab_patch_sha256"] == (
        e7.EXPECTED_CORRECTION_PATCH_SHA256
    )
    assert provenance["historical_patch_bytes_distributed"] is False
    assert provenance["semantic_equivalence_verified"] is False

    activated = e7.activated_correction_provenance(provenance)
    e7.validate_activated_correction_provenance(activated)
    altered = dict(activated)
    altered["semantic_equivalence_verified"] = True
    with pytest.raises(RuntimeError, match="correction"):
        e7.validate_activated_correction_provenance(altered)

    bootstrap = transformer_bootstrap.validate_correction(
        transformer_bootstrap.PORTABLE_CORRECTION_MANIFEST,
        transformer_bootstrap.PORTABLE_CORRECTION_PATCH,
    )
    assert bootstrap["manifest_sha256"] == (
        transformer_bootstrap.EXPECTED_PORTABLE_CORRECTION_MANIFEST_SHA256
    )
    assert bootstrap["patch_sha256"] == (
        transformer_bootstrap.EXPECTED_PORTABLE_CORRECTION_PATCH_SHA256
    )


def test_install_wires_all_three_runtime_hooks() -> None:
    import experiments.gradient_reuse_transformer.audit as audit_module
    import experiments.gradient_reuse_transformer.data as data_module
    import experiments.gradient_reuse_transformer.model as model_module

    original_model_validator = model_module.validate_mode_stable_model
    original_audit_validator = audit_module.validate_mode_stable_model
    original_loader = model_module.load_local_pythia
    original_token_ids = data_module._token_ids
    fix._INSTALLED = False
    try:
        fix.install()
        assert model_module.load_local_pythia is fix.load_local_pythia_compatible
        assert data_module._token_ids is fix.token_ids_with_effective_vocab
        assert model_module.validate_mode_stable_model is (
            audit_module.validate_mode_stable_model
        )
    finally:
        model_module.validate_mode_stable_model = original_model_validator
        audit_module.validate_mode_stable_model = original_audit_validator
        model_module.load_local_pythia = original_loader
        data_module._token_ids = original_token_ids
        fix._INSTALLED = False


def test_public_correction_files_contain_no_private_execution_identity() -> None:
    paths = (
        e7.PORTABLE_CORRECTION_MANIFEST,
        e7.PORTABLE_CORRECTION_PATCH,
        e7.PORTABLE_CORRECTION_HISTORY,
    )
    forbidden = (
        "/" + "home",
        "/" + "Users" + "/",
        "example-author",
        "research-host",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        assert all(token.lower() not in text for token in forbidden)

    bridge = json.loads(e7.PORTABLE_CORRECTION_HISTORY.read_text(encoding="utf-8"))
    assert bridge["historical_patch_bytes_distributed"] is False
    assert bridge["portable_source_is_byte_identical_to_historical_patch"] is False
