"""Cross-entropy breadth arm on a deterministic CIFAR-100 corruption stream."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .core import AuditPacket, run_fixed_trajectory_audit, seed_everything, sha256_file


REPO = Path(__file__).resolve().parents[2]


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


class _CIFARArrays(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("CIFAR images must be NHWC")
        self.images = np.asarray(images, dtype=np.uint8)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.images[index].copy()).permute(2, 0, 1),
            torch.tensor(int(self.labels[index]), dtype=torch.long),
        )


class _ResidualBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int) -> None:
        super().__init__()
        groups = max(1, min(16, output_channels // 8))
        while output_channels % groups:
            groups -= 1
        self.conv1 = nn.Conv2d(
            input_channels, output_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(groups, output_channels)
        self.conv2 = nn.Conv2d(
            output_channels, output_channels, 3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(groups, output_channels)
        if stride != 1 or input_channels != output_channels:
            self.skip = nn.Conv2d(
                input_channels, output_channels, 1, stride=stride, bias=False
            )
        else:
            self.skip = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        hidden = F.relu(self.norm1(self.conv1(inputs)), inplace=False)
        hidden = self.norm2(self.conv2(hidden))
        return F.relu(hidden + residual, inplace=False)


class DeepResidualClassifier(nn.Module):
    """A GroupNorm residual CNN with no mutable running-statistic buffers."""

    def __init__(
        self,
        *,
        classes: int,
        widths: list[int],
        blocks_per_stage: list[int],
    ) -> None:
        super().__init__()
        if len(widths) != len(blocks_per_stage) or len(widths) < 2:
            raise ValueError("residual stage widths/blocks are inconsistent")
        first = int(widths[0])
        self.stem = nn.Sequential(
            nn.Conv2d(3, first, 3, padding=1, bias=False),
            nn.GroupNorm(max(1, min(8, first // 8)), first),
            nn.ReLU(inplace=False),
        )
        stages: list[nn.Module] = []
        current = first
        for stage_number, (width, count) in enumerate(zip(widths, blocks_per_stage)):
            width = int(width)
            for block_number in range(int(count)):
                stride = 2 if stage_number > 0 and block_number == 0 else 1
                stages.append(_ResidualBlock(current, width, stride))
                current = width
        self.body = nn.Sequential(*stages)
        self.head = nn.Linear(current, int(classes))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.body(self.stem(inputs))
        hidden = F.adaptive_avg_pool2d(hidden, 1).flatten(1)
        return self.head(hidden)


def _normalize(
    images: torch.Tensor, mean: list[float], std: list[float]
) -> torch.Tensor:
    value = images.float() / 255.0 if images.dtype == torch.uint8 else images.float()
    center = torch.tensor(mean, dtype=value.dtype, device=value.device)[None, :, None, None]
    scale = torch.tensor(std, dtype=value.dtype, device=value.device)[None, :, None, None]
    return (value - center) / scale


def _synthetic_cifar(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed) + 991)
    classes = 4

    def create(count: int) -> tuple[np.ndarray, np.ndarray]:
        labels = np.arange(count, dtype=np.int64) % classes
        rng.shuffle(labels)
        image = rng.normal(35.0, 14.0, size=(count, 16, 16, 3))
        for index, label in enumerate(labels):
            channel = int(label % 3)
            quadrant = int(label)
            row = 1 if quadrant < 2 else 8
            column = 1 if quadrant % 2 == 0 else 8
            image[index, row : row + 7, column : column + 7, channel] += 170.0
        return np.clip(image, 0, 255).astype(np.uint8), labels

    train_x, train_y = create(256)
    stream_x, stream_y = create(192)
    return train_x, train_y, stream_x, stream_y


def load_classification_data(
    config: Mapping[str, Any], *, smoke: bool, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if smoke:
        train_x, train_y, stream_x, stream_y = _synthetic_cifar(seed)
        manifest = {
            "dataset": "synthetic_cifar_smoke",
            "paper_evidence_eligible": False,
            "train_shape": list(train_x.shape),
            "stream_shape": list(stream_x.shape),
            "train_sha256": _array_sha256(train_x),
            "stream_sha256": _array_sha256(stream_x),
            "labels_sha256": _array_sha256(stream_y),
        }
        return train_x, train_y, stream_x, stream_y, manifest

    if str(config["dataset"]) != "CIFAR100":
        raise ValueError("the locked classification dataset must be CIFAR100")
    try:
        import torchvision
        from torchvision.datasets import CIFAR100
    except Exception as error:  # pragma: no cover - server preflight path.
        raise RuntimeError(
            "full classification requires a working torchvision installation"
        ) from error
    configured_root = Path(str(config["data_root"])).expanduser()
    root = (
        configured_root.resolve()
        if configured_root.is_absolute()
        else (REPO / configured_root).resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    allow_download = bool(config.get("allow_download", False))
    train = CIFAR100(root=str(root), train=True, download=allow_download)
    test = CIFAR100(root=str(root), train=False, download=allow_download)
    train_x = np.asarray(train.data, dtype=np.uint8)
    train_y = np.asarray(train.targets, dtype=np.int64)
    stream_x = np.asarray(test.data, dtype=np.uint8)
    stream_y = np.asarray(test.targets, dtype=np.int64)
    if train_x.shape != (50_000, 32, 32, 3) or stream_x.shape != (10_000, 32, 32, 3):
        raise RuntimeError("unexpected CIFAR-100 canonical array shapes")
    if set(np.unique(train_y)) != set(range(100)) or set(np.unique(stream_y)) != set(range(100)):
        raise RuntimeError("CIFAR-100 labels are incomplete")
    archive = root / "cifar-100-python.tar.gz"
    if not archive.is_file():
        raise RuntimeError(
            "paper-eligible CIFAR-100 requires the canonical source archive; "
            "an extracted-only tree is insufficient provenance"
        )
    import hashlib as _hashlib

    md5_digest = _hashlib.md5()
    with archive.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5_digest.update(block)
    md5 = md5_digest.hexdigest()
    if md5 != str(config["archive_md5"]):
        raise RuntimeError(
            f"CIFAR-100 archive MD5 mismatch: expected {config['archive_md5']}, observed {md5}"
        )
    archive_record: dict[str, Any] = {
        "path": str(archive),
        "bytes": archive.stat().st_size,
        "md5": md5,
        "sha256": sha256_file(archive),
    }
    manifest = {
        "dataset": "CIFAR100",
        "source": str(config["source_url"]),
        "license": "terms not asserted by this runner; author must verify from the authoritative dataset source before release",
        "torchvision": str(torchvision.__version__),
        "archive": archive_record,
        "train_shape": list(train_x.shape),
        "stream_shape": list(stream_x.shape),
        "train_array_sha256": _array_sha256(train_x),
        "train_labels_sha256": _array_sha256(train_y),
        "stream_array_sha256": _array_sha256(stream_x),
        "stream_labels_sha256": _array_sha256(stream_y),
        "canonical_split_reuse": False,
        "stream_semantics": "deterministic nonstationary online stream; not natural chronology",
    }
    return train_x, train_y, stream_x, stream_y, manifest


def _corrupt(
    images: np.ndarray,
    specification: Mapping[str, Any],
    *,
    seed: int,
    block_number: int,
) -> torch.Tensor:
    value = torch.from_numpy(np.asarray(images, dtype=np.uint8).copy()).permute(0, 3, 1, 2).float() / 255.0
    kind = str(specification["kind"])
    severity = float(specification.get("severity", 0.0))
    if kind == "clean":
        pass
    elif kind == "brightness":
        value = value * max(0.0, 1.0 - severity)
    elif kind == "contrast":
        center = value.mean(dim=(-2, -1), keepdim=True)
        value = center + (value - center) * max(0.0, 1.0 - severity)
    elif kind == "gaussian_noise":
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(block_number), 44017])
        )
        noise = torch.from_numpy(
            rng.normal(0.0, severity, size=tuple(value.shape)).astype(np.float32)
        )
        value = value + noise
    elif kind == "blur":
        kernel = int(specification.get("kernel", 3))
        if kernel < 3 or kernel % 2 != 1:
            raise ValueError("blur kernel must be odd and at least three")
        value = F.avg_pool2d(value, kernel, stride=1, padding=kernel // 2)
    elif kind == "quantize":
        levels = int(specification.get("levels", 16))
        if levels < 2:
            raise ValueError("quantization requires at least two levels")
        value = torch.round(value * (levels - 1)) / float(levels - 1)
    else:
        raise ValueError(f"unknown locked corruption {kind!r}")
    return torch.clamp(value, 0.0, 1.0)


def classification_packets(
    images: np.ndarray,
    labels: np.ndarray,
    config: Mapping[str, Any],
    *,
    seed: int,
    smoke: bool,
) -> Iterator[AuditPacket]:
    stream_seed = int(config["stream_order_seed"])
    order = np.random.default_rng(stream_seed).permutation(len(labels))
    if smoke:
        order = order[: min(len(order), 192)]
    block_size = int(config["domain_block_size"] if not smoke else 48)
    packet_size = int(config["packet_size"] if not smoke else 16)
    domains = list(config["domains"])
    mean = list(config["normalization_mean"] if not smoke else [0.5, 0.5, 0.5])
    std = list(config["normalization_std"] if not smoke else [0.25, 0.25, 0.25])
    if len(order) < 2 * packet_size:
        raise RuntimeError("classification stream is too short")
    for block_number, start in enumerate(range(0, len(order), block_size)):
        block_indices = order[start : start + block_size]
        if len(block_indices) < 2:
            raise RuntimeError("a classification domain block contains one example")
        domain = domains[block_number % len(domains)]
        corrupted = _corrupt(
            images[block_indices], domain, seed=seed, block_number=block_number
        )
        normalized = _normalize(corrupted, mean, std)
        for local_start in range(0, len(block_indices), packet_size):
            local_end = min(local_start + packet_size, len(block_indices))
            if local_end - local_start < 2:
                raise RuntimeError("a classification packet contains one example")
            chosen = block_indices[local_start:local_end]
            yield AuditPacket(
                inputs=normalized[local_start:local_end].contiguous(),
                targets=np.asarray(labels[chosen], dtype=np.int64),
                ids=np.asarray(chosen, dtype=np.int64),
                cluster=f"block-{block_number:03d}",
                domain=str(domain["name"]),
            )


def _train_warmup(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    config: Mapping[str, Any],
    *,
    device: str,
    seed: int,
    smoke: bool,
) -> tuple[torch.optim.Optimizer, list[float], dict[str, Any]]:
    seed_everything(seed)
    torch_device = torch.device(device)
    model.to(torch_device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config["warmup_learning_rate"] if not smoke else 0.08),
        momentum=float(config["momentum"]),
        weight_decay=float(config["weight_decay"]),
    )
    dataset = _CIFARArrays(train_x, train_y)
    generator = torch.Generator().manual_seed(int(seed) + 1129)
    batch_size = int(config["warmup_batch_size"] if not smoke else 32)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=int(config["data_workers"] if not smoke else 0),
        pin_memory=torch_device.type == "cuda",
        drop_last=False,
    )
    epochs = int(config["warmup_epochs"] if not smoke else 3)
    mean = list(config["normalization_mean"] if not smoke else [0.5, 0.5, 0.5])
    std = list(config["normalization_std"] if not smoke else [0.25, 0.25, 0.25])
    radius_window = int(config["radius_window"])
    update_norms: list[float] = []
    final_loss = float("nan")
    model.train()
    for epoch in range(epochs):
        if epoch == epochs - 1:
            for group in optimizer.param_groups:
                group["lr"] = float(
                    config["online_learning_rate"] if not smoke else 0.02
                )
        for batch_number, (images, labels) in enumerate(loader):
            images = _normalize(images.to(torch_device, non_blocking=True), mean, std)
            labels = labels.to(torch_device, non_blocking=True)
            capture = epoch == epochs - 1 and batch_number >= max(0, len(loader) - radius_window)
            before = (
                {name: value.detach().clone() for name, value in model.named_parameters()}
                if capture
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = F.cross_entropy(output, labels)
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
            f"classification warmup seed={seed} epoch={epoch + 1}/{epochs} "
            f"cross_entropy={final_loss:.6f}",
            flush=True,
        )
    if not update_norms or not np.isfinite(final_loss):
        raise RuntimeError("classification warmup did not produce a valid radius")
    model.eval()
    return optimizer, update_norms, {
        "epochs": epochs,
        "updates": epochs * len(loader),
        "final_cross_entropy": final_loss,
        "radius_history": update_norms,
        "optimizer": "SGD",
    }


def run_classification_unit(
    config: Mapping[str, Any], *, seed: int, device: str, smoke: bool
) -> dict[str, Any]:
    # Seed before constructing the model.  Seeding only inside ``_train_warmup``
    # is too late because parameter initialization has already consumed the
    # process-global Torch RNG.
    seed_everything(seed)
    train_x, train_y, stream_x, stream_y, data_manifest = load_classification_data(
        config, smoke=smoke, seed=seed
    )
    classes = int(len(np.unique(train_y)))
    model = DeepResidualClassifier(
        classes=classes,
        widths=(list(config["model"]["widths"]) if not smoke else [8, 16]),
        blocks_per_stage=(
            list(config["model"]["blocks_per_stage"]) if not smoke else [1, 1]
        ),
    )
    optimizer, radius_history, warmup = _train_warmup(
        model,
        train_x,
        train_y,
        config,
        device=device,
        seed=seed,
        smoke=smoke,
    )
    core = run_fixed_trajectory_audit(
        model=model,
        optimizer=optimizer,
        packets=classification_packets(
            stream_x, stream_y, config, seed=seed, smoke=smoke
        ),
        task="classification",
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
            "schema_version": "fmmt.iclr_breadth.classification_unit.v1",
            "setting": "classification_cross_entropy",
            "dataset_manifest": data_manifest,
            "warmup": warmup,
            "model": {
                "name": "DeepResidualClassifier",
                "normalization": "GroupNorm",
                "classes": classes,
                "stochastic_layers": False,
            },
            "smoke": bool(smoke),
        }
    )
    return core
