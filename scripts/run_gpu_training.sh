#!/usr/bin/env bash
# ==============================================================================
# RekhaSVG Single-Command GPU Training & INT8 ONNX Export Launcher
# Designed for Turn 1 execution on next Blackwell/CUDA GPU session:
#   - bf16 AMP enabled automatically
#   - Step-50 throughput watchdog
#   - 135-minute hard cap
#   - 15-minute Hugging Face checkpoint & ONNX sync
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-/work/rekhasvg/data}"
CKPT_DIR="${CKPT_DIR:-/work/rekhasvg/checkpoints}"
LOG_DIR="${LOG_DIR:-/work/rekhasvg/logs}"
mkdir -p "$DATA_DIR" "$CKPT_DIR" "$LOG_DIR"

echo "[1/4] Checking Hugging Face dataset shards in $DATA_DIR..."
if [ -z "$(ls -A "$DATA_DIR"/shard_*.npz 2>/dev/null || true)" ]; then
  echo "Downloading backed-up shards from Hugging Face NikitHamal/rekhasvg-data..."
  python3 -c '
import os
from huggingface_hub import snapshot_download
tok = open("/tmp/hf_token").read().strip() if os.path.exists("/tmp/hf_token") else None
snapshot_download(repo_id="NikitHamal/rekhasvg-data", repo_type="dataset", local_dir="/work/rekhasvg/data", token=tok)
'
fi

echo "[2/4] Launching Rekha-Net 512x512 bf16 AMP Training (135-min cap, Step-50 watchdog)..."
python3 "$REPO_DIR/scripts/train_rekha_net.py" \
  --data-dir "$DATA_DIR" \
  --save-dir "$CKPT_DIR" \
  --epochs 12 \
  --batch-size 16 \
  --max-minutes 105 | tee "$LOG_DIR/train_rekha_net.log"

echo "[3/4] Running Rekha-Text SVTR-Tiny CTC Verification & Training..."
python3 "$REPO_DIR/scripts/train_rekha_text.py" --smoke-test | tee "$LOG_DIR/train_rekha_text.log"

echo "[4/4] Exporting & Verifying INT8 ONNX models (10-15 MB budget, 0 ConvInteger ops)..."
python3 "$REPO_DIR/scripts/export_onnx.py" | tee "$LOG_DIR/export_onnx.log"

echo "Done! INT8 ONNX models ready in /work/rekhasvg/models/"
