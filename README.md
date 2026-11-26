# RekhaSVG (रेखा-SVG)

**On-Device Nepal Textbook Diagram to Crisp Interactive SVG Converter**  
*Three-Stage Neural-Geometric Hybrid Pipeline • 10.68 MB Total INT8 ONNX • < 1s CPU/WASM • Zero Cloud APIs*

[![Live Interactive Studio](https://img.shields.io/badge/GitHub%20Pages-Live%20Interactive%20Studio-6750A4?style=flat-square)](https://nikithamal.github.io/RekhaSVG/)

Purpose-built for Nepali **SEE (Class 10)**, **NEB (+2 Science)**, **IOE (Pulchowk Entrance)**, **CEE (Medical Entrance)**, and university STEM students and teachers. Unlike naive bitmap auto-tracers (`Potrace` / `VTracer`) that turn smartphone camera shadows into black blobs, trace text labels into uneditable jagged paths, and emit 500+ dead polygon fragments, **RekhaSVG** decomposes textbook photos into clean semantic layers and mathematically crisp SVG primitives.

---

## Three-Stage Neural-Geometric Architecture

1. **Stage 1 — Rekha-Net (`6.31 MB` INT8 ONNX, `512×512` MobileConvNeXt / NAFNet-Lite)**:
   - Joint illumination/shadow cleaning and 6-channel semantic decomposition:
     - `ch0`: Solid diagram strokes (`<line>`, `<circle>`, `<ellipse>`, `<polyline>`, `<path>`)
     - `ch1`: Dashed / construction / virtual rays (`stroke-dasharray`)
     - `ch2`: Arrowheads & callout pointers (`<marker id="rekha-arrow">`)
     - `ch3`: Shaded region fills (`<g id="layer-fills">`)
     - `ch4`: English, Nepali Devanagari, and STEM math/chemistry label bounding boxes
     - `ch5`: Topological corner & T-junction keypoints
   - Trained with **Focal + Soft-Dice + Soft-ClDice (Centerline Topology-Preserving Loss)** + **Edge-Weighted L1** so thin circuit wires and cell walls never break.

2. **Stage 2 — Rekha-Text (`4.37 MB` INT8 ONNX, `32×192` SVTR-Tiny / MobileCRNN CTC)**:
   - Reads bilingual English scientific terms, Nepali Devanagari labels (`उत्तल लेन्स`, `कोष झिल्ली`, `प्रतिरोध`, `कोण`, `केन्द्र`, `प्रकाश किरण`, `न्यूक्लियस`), and STEM notation (`α, β, θ, Ω, μ, Δ, H₂O, R₁, V=IR`).

3. **Stage 3 — Deterministic Geometric Primitive & Schneider Cubic Bezier Compiler (`0 MB` weight, Python + 1:1 Browser JS)**:
   - Integral-image Sauvola shadow & background division removal
   - Junction-guided Zhang-Suen skeletonization
   - Algebraic Taubin circle & principal-axis ellipse fitting
   - Snapped horizontal/vertical/45° technical polylines (`90°` corners)
   - Minimal-node Philip Schneider $C^1$ Cubic Bezier curve fitting (`M ... C ...`)
   - Graph-proximity callout arrow-to-label linkage (`data-callout`, `data-quiz-answer`)

---

## Verified Benchmark & Budget Summary (`eval/`)

| Metric | Target Budget | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **Rekha-Net INT8 ONNX Size** | $\le 7.5\text{ MB}$ | **6.31 MB** | PASS |
| **Rekha-Text INT8 ONNX Size** | $\le 4.5\text{ MB}$ | **4.37 MB** | PASS |
| **Total Combined INT8 ONNX** | $10.0 - 15.0\text{ MB}$ | **10.68 MB** | PASS |
| **`ConvInteger` Ops** | $0$ (WASM/XNNPACK safe) | **0** | PASS |
| **ONNX CPU Inference Latency** | $< 600\text{ ms}$ | **207.03 ms** (`203.8 ms` Net + `3.2 ms` Text) | PASS |
| **Stage 3 Compiler Latency (1-thread)** | $< 300\text{ ms}$ | **84.35 ms** (`P95: 89.54 ms`) | PASS |
| **Node Count Reduction vs Naive** | $\ge 5\times$ | **25.17× fewer nodes** | PASS |
| **Classical-Hybrid v0 `bench_500` Dice / IoU** | Baseline logged | **0.7324 Dice / 0.5778 IoU** | PASS |
| **Python $\leftrightarrow$ Browser JS Parity** | Exact 1:1 match | **1.0000 IoU (Bit-for-Bit SVG match)** | PASS |
| **Synthetic Nepal STEM Dataset (`rekhasvg-data`)** | $30\text{k} - 40\text{k}$ samples | **32,000 train + 500 bench + 20 stress** | PASS |

---

## Turn-1 GPU Session Handoff Instructions

When connecting a CUDA / Blackwell GPU box to train `Rekha-Net` and `Rekha-Text` to convergence on the 32,000 backed-up samples in `NikitHamal/rekhasvg-data`:

```bash
git clone https://github.com/NikitHamal/RekhaSVG.git
cd RekhaSVG
bash scripts/run_gpu_training.sh
```

`scripts/run_gpu_training.sh` automatically:
1. Pulls the 16 compressed shards (`32,000` samples) from Hugging Face `NikitHamal/rekhasvg-data` if not already cached in `/work/rekhasvg/data`.
2. Launches `scripts/train_rekha_net.py` with `bf16` mixed precision (`torch.autocast`), Step-50 throughput watchdog, and a 135-minute hard cap.
3. Exports trained weights via `scripts/export_onnx.py` to `rekha_net_int8.onnx` (`6.31 MB`) and `rekha_text_int8.onnx` (`4.37 MB`) with 0 `ConvInteger` ops, ready to drop into `docs/models/`.
