"""Portable reimplementation of documented Pythia compatibility rules.

The execution-time correction files contained machine-local paths and are not
distributed. This module implements three rules documented for that historical
execution without those paths:

* tolerate the unused ``classifier_dropout=0.1`` configuration field only for
  ``GPTNeoXForCausalLM`` after proving that no active dropout module exists;
* accept the checkpoint's authenticated 50,254/50,277/50,304 base,
  effective, and padded vocabulary dimensions while retaining all 50,304
  output logits; and
* validate token IDs against the tokenizer's complete effective vocabulary,
  including its 23 authenticated added tokens.

The accompanying public manifest authenticates these portable bytes and keeps
the author-recorded historical digests separate. Neither byte identity nor
semantic equivalence to the unavailable historical patches has been verified.
"""

from __future__ import annotations

import hashlib
import runpy
from pathlib import Path
from typing import Any, Callable

import torch


BASE_VOCAB_SIZE = 50_254
EFFECTIVE_VOCAB_SIZE = 50_277
MODEL_VOCAB_SIZE = 50_304
EXPECTED_TOKENIZER_JSON_SHA256 = (
    "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624"
)
EXPECTED_SPECIAL_IDS = {"bos": 0, "eos": 0, "pad": 1, "unk": 0}
EXPECTED_SPECIAL_ADDED = {"<|endoftext|>": 0, "<|padding|>": 1}
EXPECTED_WHITESPACE_ADDED = {
    " " * width: 50_278 - width for width in range(2, 25)
}
EXPECTED_ADDED_VOCAB = {**EXPECTED_SPECIAL_ADDED, **EXPECTED_WHITESPACE_ADDED}

_DROPOUT_TYPES = (
    torch.nn.Dropout,
    torch.nn.Dropout1d,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
    torch.nn.AlphaDropout,
    torch.nn.FeatureAlphaDropout,
)
_CAUSAL_DROPOUT_FIELDS = (
    "attention_dropout",
    "hidden_dropout",
    "attn_pdrop",
    "embd_pdrop",
    "resid_pdrop",
)
_VALIDATED_TOKENIZERS: dict[int, tuple[Any, frozenset[int]]] = {}
_INSTALLED = False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_classifier_dropout_compatibility(
    model: torch.nn.Module,
    original_validator: Callable[[torch.nn.Module], None],
) -> None:
    """Apply the narrow execution-time classifier-dropout exception."""

    try:
        original_validator(model)
        return
    except RuntimeError as error:
        if str(error) != "model config has nonzero classifier_dropout=0.1":
            raise

    backbone = getattr(model, "backbone", model)
    config = getattr(backbone, "config", None)
    architectures = getattr(config, "architectures", None)
    if not (
        type(backbone).__name__ == "GPTNeoXForCausalLM"
        and getattr(config, "model_type", None) == "gpt_neox"
        and isinstance(architectures, (list, tuple))
        and "GPTNeoXForCausalLM" in architectures
        and float(getattr(config, "classifier_dropout")) == 0.1
    ):
        raise RuntimeError(
            "classifier-dropout compatibility is restricted to the "
            "authenticated GPTNeoXForCausalLM architecture"
        )

    for name, module in model.named_modules():
        label = name or "<root>"
        if isinstance(module, _DROPOUT_TYPES) and float(module.p) != 0.0:
            raise RuntimeError(f"audit model contains active dropout at {label}")
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            raise RuntimeError(f"audit model contains BatchNorm at {label}")
    for field in _CAUSAL_DROPOUT_FIELDS:
        value = getattr(config, field, None)
        if value is not None and float(value) != 0.0:
            raise RuntimeError(f"model config has nonzero {field}={value}")
    if any("classifier" in name.lower() for name, _ in backbone.named_modules()):
        raise RuntimeError("causal LM unexpectedly contains a classifier module")


def validate_padded_vocabulary_identity(
    *,
    tokenizer_vocab: int,
    tokenizer_length: int,
    model_vocab: int,
    input_rows: int,
    output_rows: int,
    special_ids: dict[str, int],
) -> None:
    """Validate the exact checkpoint/tokenizer padding relationship."""

    if not (
        tokenizer_vocab == BASE_VOCAB_SIZE
        and tokenizer_length == EFFECTIVE_VOCAB_SIZE
        and model_vocab == MODEL_VOCAB_SIZE
        and input_rows == MODEL_VOCAB_SIZE
        and output_rows == MODEL_VOCAB_SIZE
        and special_ids == EXPECTED_SPECIAL_IDS
        and model_vocab % 128 == 0
        and model_vocab - tokenizer_vocab == 50
    ):
        raise RuntimeError(
            "authenticated Pythia padded-vocabulary identity changed: "
            f"tokenizer_vocab={tokenizer_vocab}, "
            f"tokenizer_length={tokenizer_length}, model_vocab={model_vocab}, "
            f"input_rows={input_rows}, output_rows={output_rows}, "
            f"special_ids={special_ids}"
        )


def _token_content(value: Any) -> str:
    content = getattr(value, "content", None)
    return str(content) if content is not None else str(value)


def validate_effective_tokenizer_identity(
    tokenizer: Any,
    *,
    tokenizer_json_sha256: str = EXPECTED_TOKENIZER_JSON_SHA256,
) -> frozenset[int]:
    """Return the exact effective ID set after authenticating the tokenizer."""

    cache_key = id(tokenizer)
    cached = _VALIDATED_TOKENIZERS.get(cache_key)
    if cached is not None and cached[0] is tokenizer:
        return cached[1]

    root = Path(str(getattr(tokenizer, "name_or_path", ""))).resolve()
    tokenizer_json = root / "tokenizer.json"
    if not tokenizer_json.is_file() or _sha256(tokenizer_json) != tokenizer_json_sha256:
        raise RuntimeError("authenticated tokenizer.json identity changed")
    if int(getattr(tokenizer, "vocab_size")) != BASE_VOCAB_SIZE:
        raise RuntimeError("authenticated tokenizer base vocabulary size changed")
    if int(len(tokenizer)) != EFFECTIVE_VOCAB_SIZE:
        raise RuntimeError("authenticated tokenizer effective vocabulary size changed")

    raw_vocab = tokenizer.get_vocab()
    if not isinstance(raw_vocab, dict):
        raise RuntimeError("tokenizer get_vocab() did not return a mapping")
    full_vocab = {str(token): int(index) for token, index in raw_vocab.items()}
    effective_ids = frozenset(full_vocab.values())
    if len(full_vocab) != EFFECTIVE_VOCAB_SIZE or effective_ids != frozenset(
        range(EFFECTIVE_VOCAB_SIZE)
    ):
        raise RuntimeError("authenticated tokenizer effective ID inventory changed")

    raw_added = tokenizer.get_added_vocab()
    if not isinstance(raw_added, dict):
        raise RuntimeError("tokenizer get_added_vocab() did not return a mapping")
    added = {str(token): int(index) for token, index in raw_added.items()}
    if added != EXPECTED_ADDED_VOCAB:
        raise RuntimeError("authenticated tokenizer added-token mapping changed")

    decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if not isinstance(decoder, dict) or set(map(int, decoder)) != set(
        EXPECTED_ADDED_VOCAB.values()
    ):
        raise RuntimeError("authenticated tokenizer added-token decoder changed")
    for token, index in EXPECTED_ADDED_VOCAB.items():
        value = decoder[index]
        if _token_content(value) != token:
            raise RuntimeError(f"added-token content changed at ID {index}")
        expected_special = index in {0, 1}
        if (
            bool(getattr(value, "special", False)) != expected_special
            or bool(getattr(value, "normalized", False)) == expected_special
            or bool(getattr(value, "single_word", False))
            or bool(getattr(value, "lstrip", False))
            or bool(getattr(value, "rstrip", False))
        ):
            raise RuntimeError(f"added-token flags changed at ID {index}")

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None or not (
        int(backend.get_vocab_size(with_added_tokens=False)) == BASE_VOCAB_SIZE
        and int(backend.get_vocab_size(with_added_tokens=True)) == EFFECTIVE_VOCAB_SIZE
    ):
        raise RuntimeError("authenticated backend tokenizer sizes changed")
    special_ids = {
        "bos": tokenizer.bos_token_id,
        "eos": tokenizer.eos_token_id,
        "pad": tokenizer.pad_token_id,
        "unk": tokenizer.unk_token_id,
    }
    if special_ids != EXPECTED_SPECIAL_IDS:
        raise RuntimeError(f"authenticated special-token IDs changed: {special_ids}")
    if max(effective_ids) >= MODEL_VOCAB_SIZE:
        raise RuntimeError("effective tokenizer IDs exceed model input/output rows")

    _VALIDATED_TOKENIZERS[cache_key] = (tokenizer, effective_ids)
    return effective_ids


def token_ids_with_effective_vocab(tokenizer: Any, text: str) -> list[int]:
    """Tokenize while accepting all and only authenticated effective IDs."""

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    values = encoded.get("input_ids")
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    if values and isinstance(values[0], list):
        if len(values) != 1:
            raise RuntimeError("tokenizer returned more than one sequence")
        values = values[0]
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise RuntimeError("tokenizer did not return a flat integer input_ids list")
    effective_ids = validate_effective_tokenizer_identity(tokenizer)
    if any(value not in effective_ids for value in values):
        raise RuntimeError(
            "tokenizer emitted an ID outside its authenticated effective vocabulary"
        )
    return values


def load_local_pythia_compatible(
    model_path: str | Path,
    *,
    dtype: str = "float32",
):
    """Load the authenticated local model with the padded-vocabulary rules."""

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - GPU environment only.
        raise RuntimeError(
            "full Transformer runs require the pinned transformers dependency"
        ) from error

    from experiments.gradient_reuse_transformer import model as model_module

    path = str(Path(model_path).resolve())
    dtype_map = {"float32": torch.float32, "bfloat16": torch.bfloat16}
    if dtype not in dtype_map:
        raise ValueError("dtype must be float32 or bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, use_fast=True
    )
    loaded = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        torch_dtype=dtype_map[dtype],
        attn_implementation="eager",
        output_loading_info=True,
    )
    if not isinstance(loaded, tuple) or len(loaded) != 2:
        raise RuntimeError("transformers did not return model loading diagnostics")
    backbone, loading_info = loaded
    if not isinstance(loading_info, dict):
        raise RuntimeError("invalid model loading diagnostics")
    failures = {
        key: loading_info.get(key, [])
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        if loading_info.get(key)
    }
    if failures:
        raise RuntimeError(
            f"local checkpoint is not an exact architecture match: {failures}"
        )

    model = model_module.PythiaNextTokenModel(backbone)
    config = backbone.config
    if not (
        type(backbone).__name__ == "GPTNeoXForCausalLM"
        and getattr(config, "model_type", None) == "gpt_neox"
        and "GPTNeoXForCausalLM" in (getattr(config, "architectures", None) or [])
    ):
        raise RuntimeError("padded-vocabulary correction loaded an unexpected model")

    special_ids = {
        key: int(value)
        for key, value in {
            "bos": tokenizer.bos_token_id,
            "eos": tokenizer.eos_token_id,
            "pad": tokenizer.pad_token_id,
            "unk": tokenizer.unk_token_id,
        }.items()
        if value is not None
    }
    validate_padded_vocabulary_identity(
        tokenizer_vocab=int(tokenizer.vocab_size),
        tokenizer_length=int(len(tokenizer)),
        model_vocab=model.vocab_size,
        input_rows=int(backbone.get_input_embeddings().weight.shape[0]),
        output_rows=int(backbone.get_output_embeddings().weight.shape[0]),
        special_ids=special_ids,
    )
    validate_effective_tokenizer_identity(tokenizer)
    model.eval()
    return model, tokenizer


def install() -> None:
    """Install all three compatibility rules exactly once in this process."""

    global _INSTALLED
    if _INSTALLED:
        return

    import experiments.gradient_reuse_transformer.audit as audit_module
    import experiments.gradient_reuse_transformer.data as data_module
    import experiments.gradient_reuse_transformer.model as model_module

    original_validator = model_module.validate_mode_stable_model

    def corrected_validator(model: torch.nn.Module) -> None:
        validate_classifier_dropout_compatibility(model, original_validator)

    model_module.validate_mode_stable_model = corrected_validator
    audit_module.validate_mode_stable_model = corrected_validator
    model_module.load_local_pythia = load_local_pythia_compatible
    data_module._token_ids = token_ids_with_effective_vocab
    _INSTALLED = True


if __name__ in {"_reviewer_extension_effective_vocab_correction", "__main__"}:
    install()


if __name__ == "__main__":  # pragma: no cover - operational entry point.
    runpy.run_module(
        "experiments.gradient_reuse_transformer.run",
        run_name="__main__",
        alter_sys=True,
    )
