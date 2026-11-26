#!/usr/bin/env python3
"""
Training, Validation & CPU Smoke-Test Pipeline for Rekha-Net (512x512 6-Channel Semantic Decomposer).
Features:
  - `--smoke-test`: 2-step CPU forward/backward verification with Focal + Dice + ClDice + Edge-L1 + Text-Leakage Penalty
  - Full GPU bf16 AMP training on `/work/rekhasvg/data_v2/shard_*.npz` with strict <= 25 GB VRAM cap
  - Mandatory Step-50 throughput watchdog + 45-minute hard cap + >= 15-minute HF checkpoint sync
  - End-to-end validation on `eval/bench_500/` AND `eval/real_fixtures/` (Electrodialysis + AI2D)
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from rekhasvg.rekha_net import RekhaNet, RekhaNetLoss


class RekhaShardDataset(Dataset):
    def __init__(self, shard_dir: str, max_shards: int = 20, augment: bool = True):
        self.augment = augment
        self.samples: List[Tuple[bytes, bytes]] = []
        shard_files = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))[:max_shards]
        for sf in shard_files:
            data = np.load(sf, allow_pickle=True)
            jpgs = data["jpgs"]
            masks = data["masks"]
            for j, m in zip(jpgs, masks):
                self.samples.append((bytes(j), bytes(m)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        jpg_bytes, mask_bytes = self.samples[idx]
        img_bgr = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        packed = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        if self.augment and (idx % 5 == 0):
            # Mild illumination contrast jitter
            alpha = 0.90 + 0.20 * ((idx * 17) % 11) / 10.0
            img_rgb = np.clip(img_rgb.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        x = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        target = np.zeros((7, packed.shape[0], packed.shape[1]), dtype=np.float32)
        for ch_i in range(6):
            target[ch_i + 1] = ((packed & (1 << ch_i)) > 0).astype(np.float32)
        # Channel 0 is clean union structural ink mask (strokes + dashed + arrows, strictly excluding text!)
        target[0] = np.clip(target[1] + target[2] + target[3], 0.0, 1.0)
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
    for _ in range(2):
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


@torch.no_grad()
def evaluate_validation(
    model: RekhaNet,
    device: torch.device,
    eval_dir: str = "/work/rekhasvg/eval/bench_500",
    fixtures_dir: str = "/work/rekhasvg/eval/real_fixtures",
    max_bench: int = 120,
) -> Dict:
    model.eval()
    manifest_path = os.path.join(eval_dir, "manifest.json")
    items = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            items = json.load(f)[:max_bench]

    kernel = np.ones((3, 3), dtype=np.uint8)
    stroke_ious, text_ious, fill_ious, text_leak_rates = [], [], [], []

    for it in items:
        img_bgr = cv2.imread(os.path.join(eval_dir, it["img"]))
        packed = cv2.imread(os.path.join(eval_dir, it["mask"]), cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or packed is None:
            continue
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            logits = model(x)[0]
            probs = torch.sigmoid(logits).float().cpu().numpy()

        gt_line = (((packed & 1) > 0) | ((packed & 2) > 0) | ((packed & 4) > 0)).astype(np.uint8)
        gt_fill = ((packed & 8) > 0).astype(np.uint8)
        gt_text = ((packed & 16) > 0).astype(np.uint8)

        pred_text = (probs[5] >= 0.42).astype(np.uint8)
        pred_line = (((probs[1] >= 0.40) | (probs[2] >= 0.40) | (probs[3] >= 0.40)) & (pred_text == 0)).astype(np.uint8)
        pred_fill = (probs[4] >= 0.45).astype(np.uint8)

        # 1px tolerant IoU on structural lines
        p_d = cv2.dilate(pred_line, kernel, iterations=1) > 0
        g_d = cv2.dilate(gt_line, kernel, iterations=1) > 0
        inter = float(np.logical_and(p_d, g_d).sum())
        union = float(np.logical_or(p_d, g_d).sum())
        stroke_ious.append(inter / max(union, 1.0))

        # Fill IoU
        if np.any(gt_fill > 0):
            f_inter = float(np.logical_and(pred_fill > 0, gt_fill > 0).sum())
            f_union = float(np.logical_or(pred_fill > 0, gt_fill > 0).sum())
            fill_ious.append(f_inter / max(f_union, 1.0))

        # Text IoU & Text-to-Stroke leakage
        if np.any(gt_text > 0):
            t_inter = float(np.logical_and(pred_text > 0, gt_text > 0).sum())
            t_union = float(np.logical_or(pred_text > 0, gt_text > 0).sum())
            text_ious.append(t_inter / max(t_union, 1.0))
            leak = float(np.logical_and(pred_line > 0, gt_text > 0).sum()) / float(max(gt_text.sum(), 1))
            text_leak_rates.append(leak)

    # Evaluate on Electrodialysis & real fixtures
    fixture_ious = []
    fix_manifest = os.path.join(fixtures_dir, "manifest.json")
    if os.path.exists(fix_manifest):
        with open(fix_manifest, "r", encoding="utf-8") as f:
            f_items = json.load(f)
        for fit in f_items:
            if not fit.get("mask"):
                continue
            img_bgr = cv2.imread(os.path.join(fixtures_dir, fit["img"]))
            packed = cv2.imread(os.path.join(fixtures_dir, fit["mask"]), cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or packed is None:
                continue
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            x = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                probs = torch.sigmoid(model(x)[0]).float().cpu().numpy()
            gt_line = (((packed & 1) > 0) | ((packed & 2) > 0) | ((packed & 4) > 0)).astype(np.uint8)
            pred_text = (probs[5] >= 0.42).astype(np.uint8)
            pred_line = (((probs[1] >= 0.40) | (probs[2] >= 0.40) | (probs[3] >= 0.40)) & (pred_text == 0)).astype(np.uint8)
            p_d = cv2.dilate(pred_line, kernel, iterations=1) > 0
            g_d = cv2.dilate(gt_line, kernel, iterations=1) > 0
            fixture_ious.append(float(np.logical_and(p_d, g_d).sum()) / max(float(np.logical_or(p_d, g_d).sum()), 1.0))

    mean_iou = float(np.mean(stroke_ious)) if stroke_ious else 0.0
    mean_dice = (2.0 * mean_iou) / max(1.0 + mean_iou, 1e-6)
    model.train()
    return {
        "val_stroke_iou": round(mean_iou, 4),
        "val_stroke_dice": round(mean_dice, 4),
        "val_fill_iou": round(float(np.mean(fill_ious)) if fill_ious else 0.0, 4),
        "val_text_iou": round(float(np.mean(text_ious)) if text_ious else 0.0, 4),
        "val_text_to_stroke_leak": round(float(np.mean(text_leak_rates)) if text_leak_rates else 0.0, 5),
        "electrodialysis_fixture_iou": round(float(np.mean(fixture_ious)) if fixture_ious else 0.0, 4),
    }


def sync_models_to_hf(save_dir: str):
    for tok_path in ("/tmp/rekhasvg_hf_token", "/tmp/hf_token"):
        if os.path.exists(tok_path):
            with open(tok_path, "r", encoding="utf-8") as f:
                token = f.read().strip()
            break
    else:
        return False
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        repo_id = "NikitHamal/rekhasvg-models"
        api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
        api.upload_folder(
            folder_path=save_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Sync RekhaSVG checkpoints & telemetry",
        )
        return True
    except Exception as e:
        print(f"[HF SYNC WARNING] {type(e).__name__}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--data-dir", default="/work/rekhasvg/data_v2")
    parser.add_argument("--save-dir", default="/work/rekhasvg/checkpoints")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--max-minutes", type=float, default=42.0)
    args = parser.parse_args()

    if args.smoke_test:
        res = run_smoke_test()
        assert not res["nan_detected"], "NaN detected in Rekha-Net smoke test!"
        return

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        # Strict polite VRAM cap (<= 25 GB on 98 GB Blackwell so >= 55 GB stays free for Sakhiya)
        torch.cuda.set_per_process_memory_fraction(0.26)
        torch.backends.cudnn.benchmark = True

    model = RekhaNet().to(device)
    ckpt_init = os.path.join(args.save_dir, "rekha_net_best.pt")
    if os.path.exists(ckpt_init):
        try:
            model.load_state_dict(torch.load(ckpt_init, map_location=device))
            print(f"Resumed Rekha-Net from {ckpt_init}")
        except Exception:
            pass

    criterion = RekhaNetLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    ds = RekhaShardDataset(args.data_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=6, pin_memory=use_amp, drop_last=True)
    total_steps = max(1, len(loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.08, anneal_strategy="cos"
    )

    t_start = time.time()
    last_hf_sync = time.time()
    step_count = 0
    best_iou = 0.0
    history = []

    print(f"[REKHA-NET START] samples={len(ds)} batch={args.batch_size} epochs={args.epochs} total_steps={total_steps}")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x, y in loader:
            step_count += 1
            n_batches += 1
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
                loss, metrics = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += metrics["total"]

            if step_count == 50:
                elapsed = time.time() - t_start
                vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if use_amp else 0.0
                sps = 50.0 / max(elapsed, 1e-3)
                est_min = (total_steps / max(sps, 1e-3)) / 60.0
                print(
                    f"[WATCHDOG Step 50] throughput={sps:.2f} steps/s | est_total={est_min:.1f} min | "
                    f"max_vram={vram_mb:.0f} MB | loss={metrics['total']:.4f} | text_leak={metrics['text_leak']:.5f}"
                )

            if step_count % 200 == 0:
                vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if use_amp else 0.0
                print(
                    f"[Step {step_count}/{total_steps}] ep={epoch+1} loss={metrics['total']:.4f} "
                    f"dice_loss={metrics['dice']:.4f} cldice={metrics['cldice']:.4f} leak={metrics['text_leak']:.5f} vram={vram_mb:.0f}MB"
                )

            if (time.time() - t_start) / 60.0 >= args.max_minutes:
                print(f"Reached {args.max_minutes}-min cap; wrapping up Rekha-Net training.")
                break

        val_metrics = evaluate_validation(model, device)
        val_metrics["epoch"] = epoch + 1
        val_metrics["step"] = step_count
        val_metrics["train_loss"] = round(epoch_loss / max(n_batches, 1), 4)
        val_metrics["elapsed_min"] = round((time.time() - t_start) / 60.0, 2)
        history.append(val_metrics)
        print("[REKHA-NET VAL]", json.dumps(val_metrics))

        torch.save(model.state_dict(), os.path.join(args.save_dir, "rekha_net_latest.pt"))
        if val_metrics["val_stroke_iou"] >= best_iou:
            best_iou = val_metrics["val_stroke_iou"]
            torch.save(model.state_dict(), os.path.join(args.save_dir, "rekha_net_best.pt"))

        with open(os.path.join(args.save_dir, "rekha_net_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"best_val_iou": best_iou, "history": history}, f, indent=2)

        now = time.time()
        if (now - last_hf_sync) >= 900.0:
            sync_models_to_hf(args.save_dir)
            last_hf_sync = now

        if (time.time() - t_start) / 60.0 >= args.max_minutes:
            break

    sync_models_to_hf(args.save_dir)
    print(f"[REKHA-NET COMPLETE] best_val_stroke_iou={best_iou:.4f}")


if __name__ == "__main__":
    main()
