#!/usr/bin/env python3
"""
Installs clean, zero-error, auto-refreshing live dashboard cards into the MoLab
Marimo notebook strictly via marimo._code_mode.
Shows:
1. CPU / RAM health & active worker telemetry
2. Procedural Nepal Textbook Synthetic Dataset counts across STEM domains + HF sync status
3. Visual preview of degraded photo -> ground-truth 6-channel masks -> reconstructed interactive SVG
4. Benchmark latency, rasterized IoU, Hausdorff distance, node reduction, and ONNX model status
"""
import asyncio
import marimo._code_mode as cm

CARD_SETUP_CODE = '''import json
import os
import time
import base64
import psutil
import marimo as mo

refresh_timer = mo.ui.refresh(default_interval="5s", label="Auto-refresh telemetry (5s)")
'''

CARD_DASHBOARD_CODE = '''_ = refresh_timer.value

cpu_pct = psutil.cpu_percent(interval=0.1)
mem = psutil.virtual_memory()
ram_used_gb = mem.used / (1024 ** 3)
ram_total_gb = mem.total / (1024 ** 3)

ds_status_path = "/work/rekhasvg/eval/dataset_status.json"
bench_path = "/work/rekhasvg/eval/baseline_v0.json"
onnx_status_path = "/work/rekhasvg/eval/onnx_status.json"
preview_path = "/work/rekhasvg/eval/preview_triptych.png"
preview_svg_path = "/work/rekhasvg/eval/preview_sample.svg"

ds_data = {"total_generated": 0, "target": 32000, "domains": {"physics": 0, "biology": 0, "chemistry": 0, "math_cs": 0}, "bench_500_count": 0, "hf_synced_shards": 0, "hf_repo": "NikitHamal/rekhasvg-data", "last_sync": "Pending initial shard", "rate_per_sec": 0.0}
if os.path.exists(ds_status_path):
    try:
        with open(ds_status_path, "r", encoding="utf-8") as _f:
            ds_data.update(json.load(_f))
    except Exception:
        pass

bench_data = {"n_samples": 0, "mean_iou": 0.0, "mean_hausdorff_px": 0.0, "node_reduction_ratio": 0.0, "mean_latency_ms": 0.0, "p95_latency_ms": 0.0, "python_js_parity_iou": 1.0}
if os.path.exists(bench_path):
    try:
        with open(bench_path, "r", encoding="utf-8") as _f:
            bench_data.update(json.load(_f))
    except Exception:
        pass

onnx_data = {"rekha_net_int8_mb": 0.0, "rekha_text_int8_mb": 0.0, "total_int8_mb": 0.0, "conv_integer_ops": 0, "nan_count": 0, "cpu_latency_ms": 0.0, "status": "Initializing"}
if os.path.exists(onnx_status_path):
    try:
        with open(onnx_status_path, "r", encoding="utf-8") as _f:
            onnx_data.update(json.load(_f))
    except Exception:
        pass

dom = ds_data.get("domains", {})
preview_html = "<div style='padding:14px;background:#F6F4FF;border-radius:12px;color:#3F3D56;font-family:sans-serif;'>Generating visual preview triptych (Degraded Camera Photo → 6-Ch Semantic Masks → Crisp Reconstructed Interactive SVG)...</div>"
if os.path.exists(preview_path):
    try:
        with open(preview_path, "rb") as _pf:
            _b64 = base64.b64encode(_pf.read()).decode("ascii")
        preview_html = f"<div style='background:#FFFFFF;padding:12px;border-radius:14px;border:1px solid #E2E0EC;'><div style='font-weight:700;color:#1D1B20;margin-bottom:8px;font-family:sans-serif;'>Visual Pipeline Inspection: Degraded Camera Input → Ground-Truth 6-Channel Masks → Reconstructed Interactive SVG</div><img src='data:image/png;base64,{_b64}' style='width:100%;max-width:980px;border-radius:8px;display:block;' /></div>"
    except Exception:
        pass

card_md = f"""
<div style="font-family:'Poppins',system-ui,sans-serif;background:#FEF7FF;border:2px solid #D0BCFF;border-radius:18px;padding:20px;color:#1D1B20;">
  <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;border-bottom:1px solid #E7E0EC;padding-bottom:12px;margin-bottom:16px;">
    <div>
      <span style="background:#6750A4;color:#FFFFFF;padding:4px 12px;border-radius:999px;font-size:12px;font-weight:700;letter-spacing:0.5px;">REKHASVG LIVE STUDIO TELEMETRY</span>
      <h2 style="margin:8px 0 2px 0;font-size:22px;color:#1D1B20;">RekhaSVG — On-Device Nepal Textbook Diagram → Crisp Interactive SVG</h2>
      <div style="font-size:13px;color:#49454F;">Three-Stage Neural-Geometric Hybrid Pipeline • 10–15 MB INT8 ONNX Budget • Python & 1:1 Browser JS Compiler</div>
    </div>
    <div style="text-align:right;font-size:12px;color:#49454F;">
      <div><strong>CPU Usage:</strong> {cpu_pct:.1f}% ({psutil.cpu_count()} vCPUs)</div>
      <div><strong>RAM:</strong> {ram_used_gb:.2f} GB / {ram_total_gb:.1f} GB ({mem.percent:.1f}%)</div>
      <div><strong>HF Dataset Repo:</strong> <code>{ds_data.get("hf_repo", "NikitHamal/rekhasvg-data")}</code></div>
    </div>
  </div>

  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:16px;">
    <div style="background:#F3EDF7;border-radius:14px;padding:14px;border:1px solid #E7E0EC;">
      <div style="font-size:11px;font-weight:700;color:#6750A4;text-transform:uppercase;">Synthetic STEM Dataset Factory</div>
      <div style="font-size:26px;font-weight:800;color:#1D1B20;margin:4px 0;">{ds_data.get("total_generated", 0):,} <span style="font-size:13px;font-weight:500;color:#49454F;">/ {ds_data.get("target", 32000):,}</span></div>
      <div style="font-size:12px;color:#49454F;">
        Physics: <b>{dom.get("physics", 0):,}</b> • Bio: <b>{dom.get("biology", 0):,}</b><br/>
        Chem: <b>{dom.get("chemistry", 0):,}</b> • Math/CS: <b>{dom.get("math_cs", 0):,}</b><br/>
        Holdout <code>bench_500</code>: <b>{ds_data.get("bench_500_count", 0)}/500</b> • Rate: <b>{ds_data.get("rate_per_sec", 0.0):.1f} img/s</b>
      </div>
    </div>

    <div style="background:#E8DEF8;border-radius:14px;padding:14px;border:1px solid #D0BCFF;">
      <div style="font-size:11px;font-weight:700;color:#4F378B;text-transform:uppercase;">Stage 3 Geometric SVG Compiler (bench_500)</div>
      <div style="font-size:26px;font-weight:800;color:#1D1B20;margin:4px 0;">{bench_data.get("mean_latency_ms", 0.0):.1f} ms <span style="font-size:13px;font-weight:500;color:#49454F;">(1-thread CPU)</span></div>
      <div style="font-size:12px;color:#1D1B20;">
        Raster IoU: <b>{bench_data.get("mean_iou", 0.0):.3f}</b> • Hausdorff: <b>{bench_data.get("mean_hausdorff_px", 0.0):.2f} px</b><br/>
        Node Reduction vs Naive: <b>{bench_data.get("node_reduction_ratio", 0.0):.1f}× fewer nodes</b><br/>
        Python↔JS 1:1 Parity IoU: <b>{bench_data.get("python_js_parity_iou", 1.0):.4f}</b>
      </div>
    </div>

    <div style="background:#F3EDF7;border-radius:14px;padding:14px;border:1px solid #E7E0EC;">
      <div style="font-size:11px;font-weight:700;color:#6750A4;text-transform:uppercase;">INT8 ONNX Neural Budget (10–15 MB Target)</div>
      <div style="font-size:26px;font-weight:800;color:#1D1B20;margin:4px 0;">{onnx_data.get("total_int8_mb", 0.0):.2f} MB <span style="font-size:13px;font-weight:500;color:#49454F;">Total INT8</span></div>
      <div style="font-size:12px;color:#49454F;">
        Rekha-Net: <b>{onnx_data.get("rekha_net_int8_mb", 0.0):.2f} MB</b> • Rekha-Text: <b>{onnx_data.get("rekha_text_int8_mb", 0.0):.2f} MB</b><br/>
        ConvInteger Ops: <b>{onnx_data.get("conv_integer_ops", 0)}</b> • NaN Count: <b>{onnx_data.get("nan_count", 0)}</b><br/>
        CPU ONNX Latency: <b>{onnx_data.get("cpu_latency_ms", 0.0):.1f} ms</b> ({onnx_data.get("status", "Ready")})
      </div>
    </div>

    <div style="background:#F3EDF7;border-radius:14px;padding:14px;border:1px solid #E7E0EC;">
      <div style="font-size:11px;font-weight:700;color:#6750A4;text-transform:uppercase;">Hugging Face Backup (15-Min Sync)</div>
      <div style="font-size:20px;font-weight:800;color:#1D1B20;margin:6px 0;">{ds_data.get("hf_synced_shards", 0)} Shards Backed Up</div>
      <div style="font-size:12px;color:#49454F;">
        Last Sync: <b>{ds_data.get("last_sync", "In progress")}</b><br/>
        GitHub Pages Studio: <a href="https://nikithamal.github.io/RekhaSVG/" target="_blank">nikithamal.github.io/RekhaSVG</a>
      </div>
    </div>
  </div>
  {preview_html}
</div>
"""

mo.vstack([refresh_timer, mo.Html(card_md)])
'''


async def install():
    async with cm.get_context() as ctx:
        existing_ids = [c.id for c in ctx.cells]
        c1 = ctx.create_cell(CARD_SETUP_CODE, hide_code=True, name="rekha_setup")
        c2 = ctx.create_cell(CARD_DASHBOARD_CODE, after=c1, hide_code=True, name="rekha_dashboard")
        for old_id in existing_ids:
            try:
                ctx.delete_cell(old_id)
            except Exception:
                pass
        ctx.run_cell(c1)
        ctx.run_cell(c2)
        print("Installed live Marimo cards:", c1, c2)
    async with cm.get_context() as ctx:
        for cell in ctx.cells:
            print("CELL:", cell.id, cell.name, cell.status, cell.errors)


if __name__ == "__main__":
    asyncio.run(install())
