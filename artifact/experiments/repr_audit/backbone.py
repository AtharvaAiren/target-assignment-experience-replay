"""DINOv2 ViT-B/14 backbone split for representation adaptation.

Frozen head = patch_embed + pos + blocks[0:10]; the PENULTIMATE token sequence (block-9 output,
[257,768]) is cached once per image. The trainable TAIL = blocks[10:12] (frozen base weights +
LoRA on attn.qkv / mlp.fc1 / mlp.fc2, rank r) -> final norm -> CLS token -> linear head. With
LoRA disabled the tail reproduces the frozen DINOv2 feature exactly (F0 == P0@init check).
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn

TIMM_ID = "vit_base_patch14_dinov2.lvd142m"
IMG = 224
EMBED = 768
N_BLOCKS = 12
CUT = 10           # blocks 0..9 frozen+cached; blocks 10,11 trainable (last 2)


class LoRALinear(nn.Module):
    """Frozen base Linear + low-rank update (B@A), zero-initialized (B=0) so it starts as identity."""

    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.scale = alpha / r
        self.r = r

    def forward(self, x):
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale

    def lora_params(self):
        return [self.A, self.B]


def load_backbone(device="cuda"):
    import timm
    m = timm.create_model(TIMM_ID, pretrained=True, num_classes=0, img_size=IMG)
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    cfg = timm.data.resolve_data_config({}, model=m)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    return m, mean, std


@torch.no_grad()
def frozen_to_penultimate(m, x):
    """x [B,3,224,224] -> block-9 output [B,257,768] (input to the trainable tail)."""
    x = m.patch_embed(x)
    x = m._pos_embed(x)
    x = m.patch_drop(x)
    x = m.norm_pre(x)
    for i in range(CUT):
        x = m.blocks[i](x)
    return x


def make_tail(m, n_classes, r=8, alpha=16, lora=True, device="cuda"):
    """Deep-copy the last 2 blocks + final norm; inject LoRA on qkv/fc1/fc2; add a linear head."""
    b10 = copy.deepcopy(m.blocks[10])
    b11 = copy.deepcopy(m.blocks[11])
    norm = copy.deepcopy(m.norm)
    tail = Tail(b10, b11, norm, n_classes, r=r, alpha=alpha, lora=lora).to(device)
    return tail


class Tail(nn.Module):
    def __init__(self, b10, b11, norm, n_classes, r=8, alpha=16, lora=True):
        super().__init__()
        self.b10, self.b11, self.norm = b10, b11, norm
        for p in self.parameters():
            p.requires_grad_(False)                    # freeze base weights
        self._lora_mods = []
        if lora:
            for blk in (self.b10, self.b11):
                blk.attn.qkv = self._wrap(blk.attn.qkv, r, alpha)
                blk.mlp.fc1 = self._wrap(blk.mlp.fc1, r, alpha)
                blk.mlp.fc2 = self._wrap(blk.mlp.fc2, r, alpha)
        self.head = nn.Linear(EMBED, n_classes)         # trainable

    def _wrap(self, lin, r, alpha):
        w = LoRALinear(lin, r=r, alpha=alpha)
        self._lora_mods.append(w)
        return w

    def forward(self, x9):
        x = self.b11(self.b10(x9))
        x = self.norm(x)
        feat = x[:, 0]                                   # CLS token
        return self.head(feat), feat

    def lora_parameters(self):
        ps = []
        for w in self._lora_mods:
            ps += w.lora_params()
        return ps

    def n_lora_params(self):
        return sum(p.numel() for w in self._lora_mods for p in w.lora_params())


@torch.no_grad()
def frozen_pooled(m, x9):
    """Frozen DINOv2 pooled feature from a cached penultimate seq (LoRA-free tail path)."""
    x = m.blocks[11](m.blocks[10](x9))
    x = m.norm(x)
    return x[:, 0]


def _selftest():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m, mean, std = load_backbone(dev)
    x = torch.randn(2, 3, IMG, IMG, device=dev)
    with torch.no_grad():
        ref = m.forward_head(m.forward_features(x))      # [2,768] frozen feature
        x9 = frozen_to_penultimate(m, x)
        pooled = frozen_pooled(m, x9)                    # via manual last-2-blocks path
        tail = make_tail(m, 100, lora=True, device=dev)
        logits, feat = tail(x9)                          # LoRA B=0 -> feat == pooled
    print("x9", tuple(x9.shape), "ref", tuple(ref.shape))
    print("pooled==ref max|Δ|:", float((pooled - ref).abs().max()))
    print("tail-feat(LoRA=0)==ref max|Δ|:", float((feat - ref).abs().max()))
    print("tail trainable params:", sum(p.numel() for p in tail.parameters() if p.requires_grad),
          "| lora params:", tail.n_lora_params(), "| head:", sum(p.numel() for p in tail.head.parameters()))


if __name__ == "__main__":
    _selftest()
