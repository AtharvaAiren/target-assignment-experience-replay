"""Local-only causal-language-model wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


_DROPOUT_FIELDS = (
    "attention_dropout",
    "hidden_dropout",
    "classifier_dropout",
    "attn_pdrop",
    "embd_pdrop",
    "resid_pdrop",
)


def validate_mode_stable_model(model: torch.nn.Module) -> None:
    dropout_types = (
        torch.nn.Dropout,
        torch.nn.Dropout1d,
        torch.nn.Dropout2d,
        torch.nn.Dropout3d,
        torch.nn.AlphaDropout,
        torch.nn.FeatureAlphaDropout,
    )
    for name, module in model.named_modules():
        label = name or "<root>"
        if isinstance(module, dropout_types) and float(module.p) != 0.0:
            raise RuntimeError(f"audit model contains active dropout at {label}")
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            raise RuntimeError(f"audit model contains BatchNorm at {label}")
    config = getattr(getattr(model, "backbone", model), "config", None)
    for field in _DROPOUT_FIELDS:
        value = getattr(config, field, None) if config is not None else None
        if value is not None and float(value) != 0.0:
            raise RuntimeError(f"model config has nonzero {field}={value}")


class PythiaNextTokenModel(torch.nn.Module):
    """Return only logits for the token strictly after a fixed context."""

    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        config = getattr(backbone, "config", None)
        if config is None:
            raise RuntimeError("causal LM has no config")
        if getattr(config, "is_encoder_decoder", False):
            raise RuntimeError("expected a decoder-only causal language model")
        if hasattr(config, "use_cache"):
            config.use_cache = False
        validate_mode_stable_model(self)

    @property
    def vocab_size(self) -> int:
        return int(self.backbone.config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] < 2:
            raise ValueError("input_ids must be [batch, context>=2]")
        attention_mask = torch.ones_like(input_ids)
        output = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
        logits = output.logits
        if (
            logits.ndim != 3
            or logits.shape[0] != input_ids.shape[0]
            or logits.shape[1] not in {1, input_ids.shape[1]}
        ):
            raise RuntimeError("causal LM returned invalid logits")
        return logits[:, -1, :]


def load_local_pythia(
    model_path: str | Path,
    *,
    dtype: str = "float32",
) -> tuple[PythiaNextTokenModel, Any]:
    """Load an already authenticated local snapshot; never contact the Hub."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - exercised on GPU server.
        raise RuntimeError(
            "full Transformer runs require the pinned transformers dependency"
        ) from error
    path = str(Path(model_path).resolve())
    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if dtype not in dtype_map:
        raise ValueError("dtype must be float32 or bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        use_fast=True,
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
        raise RuntimeError(f"local checkpoint is not an exact architecture match: {failures}")
    model = PythiaNextTokenModel(backbone)
    if int(tokenizer.vocab_size) != model.vocab_size:
        raise RuntimeError("tokenizer/model vocabulary mismatch")
    model.eval()
    return model, tokenizer
