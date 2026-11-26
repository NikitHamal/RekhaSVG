#!/usr/bin/env python3
"""
Training, Validation & CPU Smoke-Test Pipeline for Rekha-Text (32x192 Bilingual English + Devanagari + STEM CTC).
Features:
  - `--smoke-test`: 2-step CPU forward/backward verification with CTC loss on Devanagari + English + STEM
  - Full GPU bf16 AMP training on `/work/rekhasvg/data_v2/text_crops_*.npz` (96,000 crops) + diagram-extracted
    label crops with <= 16 GB VRAM cap, Step-50 throughput check, and held-out CER evaluation on both
    English/STEM labels (target < 5% CER) and Nepali Devanagari labels (target < 8% CER).
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import random
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from rekhasvg.rekha_text import (
    REKHA_CHARSET,
    RekhaText,
    encode_text_targets,
    greedy_ctc_decode,
)


def edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_cer(preds: List[str], targets: List[str]) -> float:
    total_edits = 0
    total_chars = 0
    for p, t in zip(preds, targets):
        total_edits += edit_distance(p.strip(), t.strip())
        total_chars += max(len(t.strip()), 1)
    return float(total_edits) / float(max(total_chars, 1))


def normalize_crop_32x192(gray_crop: np.ndarray) -> np.ndarray:
    """Identical 32x192 normalization used in RekhaVectorizer.recognize_text_boxes."""
    ch_h, ch_w = gray_crop.shape[:2]
    scale = 28.0 / max(ch_h, 1)
    new_w = max(8, min(188, int(round(ch_w * scale))))
    resized = cv2.resize(gray_crop, (new_w, 28), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((32, 192), 255, dtype=np.uint8)
    p85 = float(np.percentile(resized, 85))
    if p85 > 40:
        resized = np.clip((resized.astype(np.float32) / p85) * 255.0, 0, 255).astype(np.uint8)
    canvas[2:30, 2 : 2 + new_w] = resized
    return canvas


class RekhaTextCropDataset(Dataset):
    def __init__(self, data_dir: str, eval_dir: str = "/work/rekhasvg/eval/bench_500", augment: bool = True):
        self.augment = augment
        self.samples: List[Tuple[bytes, str, bool]] = []
        crop_files = sorted(glob.glob(os.path.join(data_dir, "text_crops_*.npz")))
        for cf in crop_files:
            data = np.load(cf, allow_pickle=True)
            pngs = data["pngs"]
            texts = data["texts"]
            for p, t in zip(pngs, texts):
                self.samples.append((bytes(p), str(t), False))

        # Also extract real bbox crops from shard_000 and shard_001 so Rekha-Text matches exact diagram crops
        shard_files = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))[:2]
        for sf in shard_files:
            sdata = np.load(sf, allow_pickle=True)
            for j_bytes, meta_str in zip(sdata["jpgs"], sdata["metas"]):
                meta = json.loads(str(meta_str))
                img = cv2.imdecode(np.frombuffer(bytes(j_bytes), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                h, w = img.shape[:2]
                for lb in meta.get("labels", []):
                    bx, by, bw, bh = [int(round(v)) for v in lb.get("bbox", [0, 0, 0, 0])]
                    txt = str(lb.get("text", "")).strip()
                    if not txt or bw < 4 or bh < 4:
                        continue
                    x0, y0 = max(0, bx - 2), max(0, by - 2)
                    x1, y1 = min(w, bx + bw + 2), min(h, by + bh + 2)
                    can = normalize_crop_32x192(img[y0:y1, x0:x1])
                    ok, c_png = cv2.imencode(".png", can)
                    if ok:
                        self.samples.append((c_png.tobytes(), txt, True))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        png_bytes, txt, _ = self.samples[idx]
        gray = cv2.imdecode(np.frombuffer(png_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (32, 192):
            gray = np.full((32, 192), 255, dtype=np.uint8)
        if self.augment and (idx % 3 == 0):
            f_img = gray.astype(np.float32)
            contrast = 0.88 + 0.24 * ((idx * 13) % 10) / 10.0
            f_img = np.clip((f_img - 128.0) * contrast + 128.0, 0, 255)
            gray = f_img.astype(np.uint8)
        tensor = torch.from_numpy(gray).float().unsqueeze(0) / 255.0
        return tensor, txt


def collate_ctc(batch: List[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, List[str]]:
    imgs = torch.stack([b[0] for b in batch], dim=0)
    txts = [b[1] for b in batch]
    return imgs, txts


def run_smoke_test() -> dict:
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = RekhaText().to(device)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    param_count = sum(p.numel() for p in model.parameters())
    sample_texts = [
        "उत्तल लेन्स (Convex Lens)",
        "प्रतिरोध R₁ = 10 Ω",
        "कोष झिल्ली (Cell)",
        "ΔH = -92 kJ/mol",
    ]
    targets, target_lengths = encode_text_targets(sample_texts)
    losses = []
    t0 = time.perf_counter()
    for _ in range(2):
        x = torch.rand(len(sample_texts), 1, 32, 192, device=device)
        optimizer.zero_grad()
        log_probs = model(x)
        T = log_probs.shape[1]
        input_lengths = torch.full((len(sample_texts),), T, dtype=torch.long)
        loss = ctc_loss(log_probs.transpose(0, 1), targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))

    decoded = greedy_ctc_decode(log_probs.detach())
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    result = {
        "model": "RekhaText",
        "params": param_count,
        "vocab_size": len(REKHA_CHARSET),
        "steps": 2,
        "losses": losses,
        "decoded_len": len(decoded),
        "smoke_time_ms": round(elapsed_ms, 2),
        "nan_detected": any(np.isnan(v) for v in losses),
    }
    print(json.dumps(result, indent=2))
    return result


@torch.no_grad()
def evaluate_ocr_validation(
    model: RekhaText,
    device: torch.device,
    val_samples: List[Tuple[bytes, str, bool]],
    fixtures_dir: str = "/work/rekhasvg/eval/real_fixtures",
) -> Dict:
    model.eval()
    en_preds, en_targets = [], []
    ne_preds, ne_targets = [], []

    batch_size = 128
    for start in range(0, len(val_samples), batch_size):
        chunk = val_samples[start : start + batch_size]
        imgs = []
        txts = []
        for p_bytes, t_str, _ in chunk:
            g = cv2.imdecode(np.frombuffer(p_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            if g is None or g.shape != (32, 192):
                continue
            imgs.append(torch.from_numpy(g).float().unsqueeze(0) / 255.0)
            txts.append(t_str)
        if not imgs:
            continue
        x = torch.stack(imgs, dim=0).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            lp = model(x)
        preds = greedy_ctc_decode(lp.float().cpu())
        for pr, gt in zip(preds, txts):
            if any(0x0900 <= ord(c) <= 0x097F for c in gt):
                ne_preds.append(pr)
                ne_targets.append(gt)
            else:
                en_preds.append(pr)
                en_targets.append(gt)

    # Also test directly on the Electrodialysis textbook diagram labels!
    electro_preds = {}
    fix_manifest = os.path.join(fixtures_dir, "manifest.json")
    if os.path.exists(fix_manifest):
        with open(fix_manifest, "r", encoding="utf-8") as f:
            f_items = json.load(f)
        for fit in f_items:
            if fit.get("name") == "electrodialysis_textbook_clean":
                img_gray = cv2.imread(os.path.join(fixtures_dir, fit["img"]), cv2.IMREAD_GRAYSCALE)
                if img_gray is not None:
                    h, w = img_gray.shape[:2]
                    for lb in fit.get("labels", []):
                        bx, by, bw, bh = [int(round(v)) for v in lb["bbox"]]
                        x0, y0 = max(0, bx - 2), max(0, by - 2)
                        x1, y1 = min(w, bx + bw + 2), min(h, by + bh + 2)
                        can = normalize_crop_32x192(img_gray[y0:y1, x0:x1])
                        inp = torch.from_numpy(can).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
                        pr = greedy_ctc_decode(model(inp).float().cpu())[0]
                        electro_preds[lb["text"]] = pr

    en_cer = compute_cer(en_preds, en_targets) if en_targets else 1.0
    ne_cer = compute_cer(ne_preds, ne_targets) if ne_targets else 1.0
    overall_cer = compute_cer(en_preds + ne_preds, en_targets + ne_targets)
    model.train()
    return {
        "val_overall_cer": round(overall_cer, 4),
        "val_en_stem_cer": round(en_cer, 4),
        "val_nepali_devanagari_cer": round(ne_cer, 4),
        "electrodialysis_ocr": electro_preds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--data-dir", default="/work/rekhasvg/data_v2")
    parser.add_argument("--save-dir", default="/work/rekhasvg/checkpoints")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--max-minutes", type=float, default=22.0)
    args = parser.parse_args()

    if args.smoke_test:
        res = run_smoke_test()
        assert not res["nan_detected"], "NaN detected in Rekha-Text smoke test!"
        return

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    if use_amp:
        torch.cuda.set_per_process_memory_fraction(0.16)
        torch.backends.cudnn.benchmark = True

    full_ds = RekhaTextCropDataset(args.data_dir, augment=True)
    rng = random.Random(42)
    indices = list(range(len(full_ds)))
    rng.shuffle(indices)
    val_size = min(2000, max(200, len(indices) // 40))
    val_samples = [full_ds.samples[i] for i in indices[:val_size]]
    train_subset = torch.utils.data.Subset(full_ds, indices[val_size:])

    loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        pin_memory=use_amp,
        collate_fn=collate_ctc,
        drop_last=True,
    )

    model = RekhaText().to(device)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = max(1, len(loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=0.08, anneal_strategy="cos"
    )

    t_start = time.time()
    step_count = 0
    best_cer = 1.0
    history = []

    print(f"[REKHA-TEXT START] train_crops={len(train_subset)} val_crops={len(val_samples)} batch={args.batch_size} epochs={args.epochs}")
    for epoch in range(args.epochs):
        ep_loss = 0.0
        n_b = 0
        for imgs, txts in loader:
            step_count += 1
            n_b += 1
            imgs = imgs.to(device, non_blocking=True)
            targets, target_lengths = encode_text_targets(txts)
            targets = targets.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                log_probs = model(imgs)
            # Compute CTC loss in fp32 for numerical stability
            lp_fp32 = log_probs.float()
            T = lp_fp32.shape[1]
            inp_len = torch.full((imgs.shape[0],), T, dtype=torch.long, device=device)
            loss = ctc_loss(lp_fp32.transpose(0, 1), targets, inp_len, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            ep_loss += float(loss.detach().item())

            if step_count == 50:
                elapsed = time.time() - t_start
                vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if use_amp else 0.0
                sps = 50.0 / max(elapsed, 1e-3)
                print(f"[WATCHDOG Step 50] throughput={sps:.2f} steps/s | vram={vram_mb:.0f} MB | ctc_loss={loss.item():.4f}")

            if (time.time() - t_start) / 60.0 >= args.max_minutes:
                break

        val_res = evaluate_ocr_validation(model, device, val_samples)
        val_res["epoch"] = epoch + 1
        val_res["step"] = step_count
        val_res["train_ctc_loss"] = round(ep_loss / max(n_b, 1), 4)
        val_res["elapsed_min"] = round((time.time() - t_start) / 60.0, 2)
        history.append(val_res)
        print("[REKHA-TEXT VAL]", json.dumps(val_res, ensure_ascii=False))

        torch.save(model.state_dict(), os.path.join(args.save_dir, "rekha_text_latest.pt"))
        if val_res["val_overall_cer"] <= best_cer:
            best_cer = val_res["val_overall_cer"]
            torch.save(model.state_dict(), os.path.join(args.save_dir, "rekha_text_best.pt"))

        with open(os.path.join(args.save_dir, "rekha_text_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"best_val_cer": best_cer, "history": history}, f, ensure_ascii=False, indent=2)

        if val_res["val_en_stem_cer"] < 0.015 and val_res["val_nepali_devanagari_cer"] < 0.025 and epoch >= 3:
            print("Target CER well below threshold (<1.5% EN, <2.5% NE); finishing Rekha-Text early.")
            break
        if (time.time() - t_start) / 60.0 >= args.max_minutes:
            break

    print(f"[REKHA-TEXT COMPLETE] best_val_cer={best_cer:.4f}")


if __name__ == "__main__":
    main()
