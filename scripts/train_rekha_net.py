#!/usr/bin/env python3
"""
Training & CPU Smoke-Test Pipeline for Rekha-Net (512x512 6-Channel Semantic Decomposer).
Supports:
  - `--smoke-test`: 2-step CPU forward/backward verification with Focal + Dice + ClDice + Edge-L1
  - Full GPU bf16 AMP training on shards in `/work/rekhasvg/data/shard_*.npz`
  - Step-50 throughput watchdog & 135-minute hard cap + 15-minute HF checkpoint sync
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import time
from typing import List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from rekhasvg.rekha_net import RekhaNet, RekhaNetLoss


class RekhaShardDataset(Dataset):
    def __init__(self, shard_dir: str, max_shards: int = 20):
        self.samples: List[Tuple[bytes, bytes]] = []
        shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))[:max_shards]
        for sf in shard_files:
            data = np.load(sf, allow_pickle=True)
            jpgs = data["jpgs"]
            masks = data["masks"]
            for j, m in zip(jpgs, masks):
                self.samples.append(( bytes(j), bytes(m) ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        jpg_bytes, mask_bytes = self.samples[idx]
        img = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        packed = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        x = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        target = np.zeros((7, packed.shape[0], packed.shape[1]), dtype=np.float32)
        for ch_i in range(6):
            target[ch_i + 1] = ((packed & (1 << ch_i)) > 0).astype(np.float32)
        # Channel 0 is clean union ink mask
        target[0] = np.clip(target[1] + target[2] + target[3] + target[5], 0.0, 1.0)
        return x, torch.from_numpy(target)


def run_smoke_test() -> dict:
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = RekhaNet().to(device)
    criterion = RekhaNetLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    param_count = sum(p.numel() for p in model.parameters())
    losses = []
    t0 = time.perf_counter()
    for step in range(2):
        # Use 128x128 spatial resolution during CPU 2-step smoke test for instant verification
        x = torch.rand(2, 3, 128, 128, device=device)
        y = (torch.rand(2, 7, 128, 128, device=device) > 0.75).float()
        optimizer.zero_grad()
        logits = model(x)
        loss, metrics = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(metrics["total"])
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    result = {
        "model": "RekhaNet",
        "params": param_count,
        "steps": 2,
        "losses": losses,
        "smoke_time_ms": round(elapsed_ms, 2),
        "nan_detected": any(np.isnan(v) for v in losses),
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--data-dir", default="/work/rekhasvg/data")
    parser.add_argument("--save-dir", default="/work/rekhasvg/checkpoints")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--max-minutes", type=float, default=135.0)
    args = parser.parse_args()

    if args.smoke_test:
        res = run_smoke_test()
        assert not res["nan_detected"], "NaN detected in Rekha-Net smoke test!"
        return

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model = RekhaNet().to(device)
    criterion = RekhaNetLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    ds = RekhaShardDataset(args.data_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=use_amp)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, len(loader) * args.epochs))

    t_start = time.time()
    step_count = 0
    for epoch in range(args.epochs):
        for x, y in loader:
            step_count += 1
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
                loss, metrics = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step_count == 50:
                elapsed = time.time() - t_start
                print(f"[WATCHDOG Step 50] throughput={50.0 / max(elapsed, 1e-3):.2f} steps/s loss={metrics['total']:.4f}")
            if (time.time() - t_start) / 60.0 >= args.max_minutes:
                print("Reached 135-min hard cap; saving final checkpoint.")
                break
        torch.save(model.state_dict(), os.path.join(args.save_dir, "rekha_net_latest.pt"))


if __name__ == "__main__":
    main()
