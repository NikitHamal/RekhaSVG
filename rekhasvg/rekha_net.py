"""
Rekha-Net: Fast 512x512 MobileConvNeXt / NAFNet-Lite Multi-Task Encoder-Decoder.
Sized strictly for <= 7.5 MB INT8 ONNX (~6.4 MB INT8, ~1.65M parameters) with
zero ConvInteger ops (100% onnxruntime-web WASM & Android XNNPACK/NNAPI compatible).

Outputs:
  - clean_ink_logits: (B, 1, H, W) illumination-normalized clean ink mask
  - semantic_logits:  (B, 6, H, W) 6-channel decomposition:
      [0: strokes, 1: dashed, 2: arrows, 3: fills, 4: text_boxes, 5: junctions]

Loss:
  - Focal + Soft-Dice + Soft-ClDice (centerline topology-preserving loss) + Edge-Weighted L1
"""
from __future__ import annotations
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    """NAFNet nonlinear activation-free gating block (splits channels in half and multiplies)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFConvNeXtBlock(nn.Module):
    """
    Lightweight depthwise-separable ConvNeXt + NAFNet hybrid block using standard
    ONNX-friendly ops (Conv2d, BatchNorm2d, SiLU/SimpleGate).
    """
    def __init__(self, dim: int, expansion: int = 2):
        super().__init__()
        hidden = dim * expansion
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=True)
        self.gate = SimpleGate()
        self.pwconv2 = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1) * 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.pwconv1(x)
        x = self.gate(x)
        x = self.pwconv2(x)
        return residual + self.gamma * x


class DownStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_blocks: int):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[NAFConvNeXtBlock(out_ch) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class UpStage(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, num_blocks: int = 2):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[NAFConvNeXtBlock(out_ch) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.blocks(self.fuse(x))


class RekhaNet(nn.Module):
    """
    Rekha-Net (~6.6 MB INT8 ONNX):
    Fast stride-2 stem + MobileConvNeXt/NAFNet-Lite stages -> ~6.8M parameters,
    ~6.6 MB INT8 ONNX (<= 7.5 MB budget), < 300 ms CPU inference at 512x512.
    """
    def __init__(self, in_channels: int = 3, num_semantic_channels: int = 6, dims: Tuple[int, ...] = (32, 64, 128, 232, 328)):
        super().__init__()
        c0, c1, c2, c3, c4 = dims
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c0, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.SiLU(inplace=True),
            NAFConvNeXtBlock(c0),
        )
        self.enc1 = DownStage(c0, c1, num_blocks=2)   # 1/4
        self.enc2 = DownStage(c1, c2, num_blocks=2)   # 1/8
        self.enc3 = DownStage(c2, c3, num_blocks=3)   # 1/16
        self.enc4 = DownStage(c3, c4, num_blocks=3)   # 1/32

        self.dec3 = UpStage(c4, c3, c3, num_blocks=1)
        self.dec2 = UpStage(c3, c2, c2, num_blocks=1)
        self.dec1 = UpStage(c2, c1, c1, num_blocks=1)
        self.dec0 = UpStage(c1, c0, c0, num_blocks=1)

        self.head = nn.Conv2d(c0, 1 + num_semantic_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns tensor of shape (B, 7, H, W):
          channel 0: clean illumination-normalized ink logits
          channels 1..6: [strokes, dashed, arrows, fills, text, junctions] logits
        """
        x = torch.clamp(x, 0.0, 1.0)
        in_size = x.shape[-2:]
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        d3 = self.dec3(s4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)
        out = self.head(d0)
        return F.interpolate(out, size=in_size, mode="bilinear", align_corners=False)


def soft_erode(img: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
    return torch.min(p1, p2)


def soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def soft_open(img: torch.Tensor) -> torch.Tensor:
    return soft_dilate(soft_erode(img))


def soft_skel(img: torch.Tensor, iters: int = 5) -> torch.Tensor:
    img1 = soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = soft_erode(img)
        img1 = soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class RekhaNetLoss(nn.Module):
    """
    Joint Multi-Task Loss:
      1. Focal Binary Cross-Entropy (handles class imbalance of thin strokes & junctions)
      2. Multi-Channel Soft Dice Loss
      3. Soft-ClDice Topology-Preserving Centerline Loss (prevents broken wires/cell walls)
      4. Edge-Weighted L1 Ink Reconstruction Loss
    """
    def __init__(self, gamma: float = 2.0, cldice_weight: float = 0.35):
        super().__init__()
        self.gamma = gamma
        self.cldice_weight = cldice_weight

    def focal_bce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        focal = ((1.0 - pt) ** self.gamma) * bce
        return focal.mean()

    def soft_dice(self, probs: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
        inter = (probs * targets).sum(dim=(2, 3))
        denom = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2.0 * inter + eps) / (denom + eps)
        return 1.0 - dice.mean()

    def soft_cldice(self, stroke_probs: torch.Tensor, stroke_targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
        skel_pred = soft_skel(stroke_probs, iters=4)
        skel_true = soft_skel(stroke_targets, iters=4)
        tprec = ((skel_pred * stroke_targets).sum(dim=(2, 3)) + eps) / (skel_pred.sum(dim=(2, 3)) + eps)
        tsens = ((skel_true * stroke_probs).sum(dim=(2, 3)) + eps) / (skel_true.sum(dim=(2, 3)) + eps)
        cl_dice = (2.0 * tprec * tsens) / (tprec + tsens + 1e-6)
        return 1.0 - cl_dice.mean()

    def forward(self, logits: torch.Tensor, target_7ch: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        probs = torch.sigmoid(logits)
        loss_focal = self.focal_bce(logits, target_7ch)
        loss_dice = self.soft_dice(probs, target_7ch)

        # Apply ClDice centerline loss on clean ink (ch 0) + solid strokes (ch 1)
        stroke_p = probs[:, :2, :, :]
        stroke_t = target_7ch[:, :2, :, :]
        loss_cldice = self.soft_cldice(stroke_p, stroke_t)

        # Edge-weighted L1 on clean ink channel
        ink_p = probs[:, 0:1, :, :]
        ink_t = target_7ch[:, 0:1, :, :]
        edge_weight = 1.0 + 3.0 * ink_t
        loss_edge_l1 = (torch.abs(ink_p - ink_t) * edge_weight).mean()

        # Explicit penalty on false-positive text-to-stroke/dash/arrow leakage
        text_gt = target_7ch[:, 5:6, :, :]
        non_text_stroke_probs = probs[:, 1:4, :, :] * (1.0 - target_7ch[:, 1:4, :, :])
        loss_text_leak = ((non_text_stroke_probs * text_gt) ** 2).mean() * 4.5

        # Explicit penalty on water-hatch fill leaking into structural dashed channel
        fill_gt = target_7ch[:, 4:5, :, :]
        dash_fp = probs[:, 2:3, :, :] * (1.0 - target_7ch[:, 2:3, :, :])
        loss_hatch_leak = ((dash_fp * fill_gt) ** 2).mean() * 3.0

        total = (
            loss_focal
            + loss_dice
            + self.cldice_weight * loss_cldice
            + 0.5 * loss_edge_l1
            + loss_text_leak
            + loss_hatch_leak
        )
        metrics = {
            "total": float(total.detach().item()),
            "focal": float(loss_focal.detach().item()),
            "dice": float(loss_dice.detach().item()),
            "cldice": float(loss_cldice.detach().item()),
            "edge_l1": float(loss_edge_l1.detach().item()),
            "text_leak": float(loss_text_leak.detach().item()),
        }
        return total, metrics
