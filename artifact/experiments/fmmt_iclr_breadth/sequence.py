"""Sequential regression breadth arm with a two-layer GRU predictor."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.fmmt_workshop.datasets import load_workshop_stream

from .core import AuditPacket, run_fixed_trajectory_audit, seed_everything


class SequentialGRURegressor(nn.Module):
    """A multi-layer sequence model with deterministic normalization."""

    def __init__(
        self, input_dim: int, *, hidden_size: int, layers: int, mlp_width: int
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(int(input_dim))
        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_size),
            num_layers=int(layers),
            batch_first=True,
            dropout=0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(int(hidden_size)),
            nn.Linear(int(hidden_size), int(mlp_width)),
            nn.GELU(),
            nn.Linear(int(mlp_width), 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(inputs)
        hidden, _ = self.gru(normalized)
        return self.head(hidden[:, -1]).squeeze(-1)


def _windows(features: np.ndarray, indices: np.ndarray, context: int) -> torch.Tensor:
    features = np.asarray(features, dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    if int(context) < 2:
        raise ValueError("sequence context must contain at least two time steps")
    output = np.zeros((len(indices), int(context), features.shape[1]), dtype=np.float32)
    for row, index in enumerate(indices):
        start = max(0, int(index) - int(context) + 1)
        available = features[start : int(index) + 1]
        output[row, -len(available) :] = available
        if start == 0 and len(available) < int(context):
            output[row, : int(context) - len(available)] = available[0]
    return torch.from_numpy(output)


def _synthetic_sequence(seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed) + 1709)
    rows = 512
    dimension = 8
    features = rng.normal(size=(rows, dimension)).astype(np.float32)
    drift = np.sin(np.arange(rows, dtype=np.float64) / 35.0).astype(np.float32)
    features[:, 0] += drift
    signal = np.zeros(rows, dtype=np.float32)
    for index in range(rows):
        start = max(0, index - 7)
        signal[index] = (
            0.7 * features[index, 0]
            + 0.4 * features[start : index + 1, 1].mean()
            - 0.25 * features[index, 2]
            + 0.2 * drift[index]
        )
    signal += rng.normal(0.0, 0.08, size=rows).astype(np.float32)
    return {
        "name": "SYNTHETIC_SEQUENCE_SMOKE",
        "X": features,
        "y": signal,
        "cal_n": 128,
        "scored_mask": np.arange(rows) >= 128,
        "month_id": np.arange(rows, dtype=np.int64) // 64,
        "event_id": np.arange(rows, dtype=np.int64),
        "chronology_preserved": True,
        "all_preprocessed_rows_retained": True,
        "chronology_audit": {
            "n_rows": rows,
            "timestamps_nondecreasing": True,
            "synthetic": True,
        },
        "data_manifest": {
            "dataset": "synthetic_sequence_smoke",
            "paper_evidence_eligible": False,
        },
    }


def load_sequence_data(
    config: Mapping[str, Any], *, smoke: bool, seed: int
) -> dict[str, Any]:
    if smoke:
        return _synthetic_sequence(seed)
    stream = load_workshop_stream(str(config["dataset"]), smoke=False, seed=seed)
    if not bool(stream.get("chronology_preserved")):
        raise RuntimeError("sequence loader did not certify chronology")
    manifest = stream.get("data_manifest")
    if not isinstance(manifest, Mapping):
        raise RuntimeError("sequence data manifest is missing")
    expected_sha = str(config["archive_sha256"])
    if manifest.get("sha256") != expected_sha:
        raise RuntimeError(
            f"sequence archive hash mismatch: expected {expected_sha}, observed {manifest.get('sha256')}"
        )
    return stream


def sequence_packets(
    stream: Mapping[str, Any], config: Mapping[str, Any], *, smoke: bool
) -> Iterator[AuditPacket]:
    features = np.asarray(stream["X"], dtype=np.float32)
    targets = np.asarray(stream["y"], dtype=np.float32)
    scored = np.flatnonzero(np.asarray(stream["scored_mask"], dtype=bool))
    context = int(config["context_length"] if not smoke else 8)
    packet_size = int(config["packet_size"] if not smoke else 16)
    if len(scored) < 2 * packet_size:
        raise RuntimeError("sequence scoring stream is too short")
    boundaries = list(range(0, len(scored), packet_size)) + [len(scored)]
    if boundaries[-1] - boundaries[-2] == 1:
        boundaries.pop(-2)
    quartile_edges = np.linspace(0, len(scored), 5)
    clusters = np.asarray(stream["month_id"])
    for packet_number, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        index = scored[start:end]
        if len(index) < 2:
            raise RuntimeError("sequence packet has fewer than two rows")
        position = 0.5 * (start + end)
        quartile = min(4, int(np.searchsorted(quartile_edges[1:], position, side="right")) + 1)
        cluster_values = clusters[index]
        unique, counts = np.unique(cluster_values, return_counts=True)
        cluster = unique[int(np.argmax(counts))]
        yield AuditPacket(
            inputs=_windows(features, index, context),
            targets=targets[index].copy(),
            ids=index.astype(np.int64),
            cluster=str(cluster),
            domain=f"chronological_quartile_{quartile}",
        )


def _train_warmup(
    model: nn.Module,
    stream: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    seed: int,
    device: str,
    smoke: bool,
) -> tuple[torch.optim.Optimizer, list[float], dict[str, Any]]:
    seed_everything(seed)
    torch_device = torch.device(device)
    model.to(torch_device)
    features = np.asarray(stream["X"], dtype=np.float32)
    targets = np.asarray(stream["y"], dtype=np.float32)
    cal_n = int(stream["cal_n"])
    context = int(config["context_length"] if not smoke else 8)
    index = np.arange(cal_n, dtype=np.int64)
    dataset = TensorDataset(
        _windows(features, index, context), torch.from_numpy(targets[index].copy())
    )
    generator = torch.Generator().manual_seed(int(seed) + 7741)
    batch_size = int(config["warmup_batch_size"] if not smoke else 32)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=torch_device.type == "cuda",
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["warmup_learning_rate"] if not smoke else 0.01),
        betas=tuple(float(value) for value in config["adamw_betas"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["warmup_epochs"] if not smoke else 4)
    radius_window = int(config["radius_window"])
    update_norms: list[float] = []
    final_loss = float("nan")
    model.train()
    for epoch in range(epochs):
        if epoch == epochs - 1:
            for group in optimizer.param_groups:
                group["lr"] = float(
                    config["online_learning_rate"] if not smoke else 0.003
                )
        for batch_number, (inputs, target) in enumerate(loader):
            inputs = inputs.to(torch_device, non_blocking=True)
            target = target.to(torch_device, non_blocking=True)
            capture = epoch == epochs - 1 and batch_number >= max(0, len(loader) - radius_window)
            before = (
                {name: value.detach().clone() for name, value in model.named_parameters()}
                if capture
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(inputs)
            loss = F.mse_loss(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip"]))
            optimizer.step()
            final_loss = float(loss.detach().item())
            if before is not None:
                squared = 0.0
                for name, value in model.named_parameters():
                    squared += float((value.detach().double() - before[name].double()).square().sum().item())
                update_norms.append(math.sqrt(max(squared, 0.0)))
        print(
            f"sequence warmup seed={seed} epoch={epoch + 1}/{epochs} mse={final_loss:.6f}",
            flush=True,
        )
    if not update_norms or not np.isfinite(final_loss):
        raise RuntimeError("sequence warmup did not produce a valid radius")
    model.eval()
    return optimizer, update_norms, {
        "epochs": epochs,
        "updates": epochs * len(loader),
        "final_mse": final_loss,
        "radius_history": update_norms,
        "optimizer": "AdamW",
    }


def run_sequence_unit(
    config: Mapping[str, Any], *, seed: int, device: str, smoke: bool
) -> dict[str, Any]:
    # Model parameters must be initialized from the declared unit seed, not
    # from whatever RNG state the launching process happens to inherit.
    seed_everything(seed)
    stream = load_sequence_data(config, smoke=smoke, seed=seed)
    input_dim = int(np.asarray(stream["X"]).shape[1])
    model = SequentialGRURegressor(
        input_dim,
        hidden_size=int(config["model"]["hidden_size"] if not smoke else 16),
        layers=int(config["model"]["layers"] if not smoke else 2),
        mlp_width=int(config["model"]["mlp_width"] if not smoke else 16),
    )
    optimizer, radius_history, warmup = _train_warmup(
        model, stream, config, seed=seed, device=device, smoke=smoke
    )
    core = run_fixed_trajectory_audit(
        model=model,
        optimizer=optimizer,
        packets=sequence_packets(stream, config, smoke=smoke),
        task="regression",
        actuators=list(config["actuators"]),
        n_permutations=(2 if smoke else int(config["n_permutations"])),
        permutation_salt=int(config["permutation_salt"]),
        seed=seed,
        device=device,
        radius_history=radius_history,
        radius_window=int(config["radius_window"]),
        fallback_radius=float(config["fallback_radius"]),
        gradient_clip=float(config["gradient_clip"]),
        bootstrap_draws=(100 if smoke else int(config["bootstrap_draws"])),
        bootstrap_block_length=int(config["bootstrap_block_length"]),
    )
    core.update(
        {
            "schema_version": "fmmt.iclr_breadth.sequence_unit.v1",
            "setting": "sequence_regression",
            "dataset_manifest": dict(stream.get("data_manifest", {})),
            "chronology_audit": dict(stream.get("chronology_audit", {})),
            "warmup": warmup,
            "model": {
                "name": "SequentialGRURegressor",
                "input_dim": input_dim,
                "hidden_size": int(config["model"]["hidden_size"] if not smoke else 16),
                "layers": int(config["model"]["layers"] if not smoke else 2),
                "context_length": int(config["context_length"] if not smoke else 8),
                "stochastic_layers": False,
            },
            "smoke": bool(smoke),
        }
    )
    return core
