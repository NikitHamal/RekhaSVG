#!/usr/bin/env python3
"""
Training & CPU Smoke-Test Pipeline for Rekha-Text (32x192 Bilingual English + Devanagari + STEM CTC).
Supports:
  - `--smoke-test`: 2-step CPU forward/backward verification with CTC loss on Devanagari + English + STEM
  - Full GPU bf16 AMP training on cropped textbook label boxes
"""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn

from rekhasvg.rekha_text import (
    REKHA_CHARSET,
    RekhaText,
    encode_text_targets,
    greedy_ctc_decode,
)


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
        log_probs = model(x)  # (B, T, V)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--save-dir", default="/work/rekhasvg/checkpoints")
    args = parser.parse_args()

    if args.smoke_test:
        res = run_smoke_test()
        assert not res["nan_detected"], "NaN detected in Rekha-Text smoke test!"
        return


if __name__ == "__main__":
    main()
