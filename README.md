# RekhaSVG (रेखा-SVG) — Neural v1

**On-Device Nepal & NCERT/NEB Textbook Diagram to Crisp Interactive SVG Converter**  
*Three-Stage Neural-Geometric Hybrid Pipeline • 10.71 MB Total INT8 ONNX • < 210 ms CPU/WASM • Zero Cloud APIs*

[![Live Interactive Studio](https://img.shields.io/badge/GitHub%20Pages-Live%20Interactive%20Studio-6750A4?style=flat-square)](https://nikithamal.github.io/RekhaSVG/)
[![Hugging Face Models](https://img.shields.io/badge/Hugging%20Face-rekhasvg--models-FFB300?style=flat-square)](https://huggingface.co/NikitHamal/rekhasvg-models)

Purpose-built for Nepali **SEE (Class 10)**, **NEB (+2 Science)**, **IOE (Pulchowk Entrance)**, **CEE (Medical Entrance)**, and NCERT/university STEM students and teachers. Unlike naive bitmap auto-tracers (`Potrace` / `VTracer`) that turn smartphone camera shadows into black blobs, trace printed text into uneditable jagged stick paths, snap tiny characters (`+`, `o`) into giant circles, or chain blue water-hatch textures into chaotic zig-zags, **RekhaSVG Neural v1** decomposes textbook illustrations and camera photos into clean semantic layers and mathematically crisp SVG primitives.

---

## Three-Stage Neural-Geometric Architecture (Neural v1)

1. **Stage 1 — Rekha-Net (`6.32 MB` INT8 ONNX, `512×512` MobileConvNeXt / NAFNet-Lite)**:
   - Joint illumination/shadow cleaning and 6-channel semantic decomposition:
     - `ch0`: Solid diagram strokes (`<line>`, `<circle>`, `<ellipse>`, `<polyline>`, `<path>`)
     - `ch1`: Structural dashed / construction rays (`stroke-dasharray`)
     - `ch2`: Arrowheads & callout pointers (`<marker id="rekha-arrow">`)
     - `ch3`: Shaded & colored/hatched liquid fills (`<g id="layer-fills">`)
     - `ch4`: English, Nepali Devanagari, and STEM math/chemistry label bounding boxes
     - `ch5`: Topological corner & T-junction keypoints
   - Trained in `bf16` AMP on the **Zero-Tofu v2 Textbook Corpus** with **Focal + Soft-Dice + Soft-ClDice (Centerline Topology-Preserving Loss) + Edge-Weighted L1 + Explicit Text-to-Stroke & Water-Hatch Leakage Penalties**.

2. **Stage 2 — Rekha-Text (`4.39 MB` INT8 ONNX, `32×192` SVTR-Tiny / MobileCRNN CTC)**:
   - Trained on **96,000+ spotless zero-tofu word and line crops** across serif, sans-serif, bold, and Devanagari fonts (`0.03%` overall validation CER: `0.04%` English/STEM CER, `0.01%` Nepali Devanagari CER).
   - Automatic horizontal projection valley splitting for stacked two-line labels (`semi permeable / membrane`, `Water + / Electrolyte`), emitting editable `<text>` elements linked to their callout arrows.

3. **Stage 3 — Deterministic Geometric Primitive & Schneider Cubic Bezier Compiler (`0 MB` weight, Python + 1:1 Browser JS)**:
   - 4-quadrant angular span & bounding-box guardrails on Taubin circle and principal-axis ellipse fitting (preventing tiny `+` or `o` glyphs from snapping into giant circles)
   - 2D periodic water-hatch/stipple filter + strict colinear dashed-line grouping
   - Dilated text-box exclusion masks so printed text is never vectorized as strokes

---

## Verified Neural v1 Benchmark & Budget Summary (`eval/`)

| Metric | Target Budget | Classical v0 | **Neural v1 (Trained INT8 ONNX)** | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Rekha-Net INT8 ONNX Size** | $\le 8.5\text{ MB}$ | 6.31 MB (untrained) | **6.32 MB** (trained) | PASS |
| **Rekha-Text INT8 ONNX Size** | $\le 5.5\text{ MB}$ | 4.37 MB (untrained) | **4.39 MB** (trained) | PASS |
| **Total Combined INT8 ONNX** | $10.0 - 15.0\text{ MB}$ | 10.68 MB | **10.71 MB** | PASS |
| **`ConvInteger` Ops / NaNs** | $0 / 0$ | $0 / 0$ | **0 ConvInteger / 0 NaNs** | PASS |
| **ONNX CPU Latency (Python / Node)** | $< 250\text{ ms} / < 15\text{ ms}$ | 207.03 ms | **185.7 ms Net + 3.85 ms Text (`186.9 ms` Node)** | PASS |
| **Holdout `bench_500` Mean IoU / Dice** | $\ge 0.82 / \ge 0.90$ | 0.5778 IoU / 0.7324 Dice | **0.8938 IoU (+0.3160) / 0.9408 Dice** | PASS |
| **Rekha-Net Validation IoU / Dice** | $\ge 0.82 / \ge 0.90$ | — | **0.9920 IoU / 0.9960 Dice** | PASS |
| **Electrodialysis Real Textbook Fixture IoU** | Clean separation | Failed (preset bleed + zigzags) | **0.9341 IoU (0 bleed, 0 zig-zags, 11/11 labels)** | PASS |
| **Rekha-Text Validation CER (EN / NE)** | $< 5\%\text{ EN} / < 8\%\text{ NE}$ | — | **0.04% EN/STEM CER / 0.01% Devanagari CER** | PASS |
| **Text-to-Stroke Leakage Rate** | Near-zero | High | **0.00000 (0.0%)** | PASS |
| **Node Count Reduction vs Naive** | $\ge 5\times$ | 25.17× | **30.24× fewer nodes (`18.89 ms` Stage 3)** | PASS |
| **Zero-Tofu v2 Dataset (`rekhasvg-data/v2`)** | Zero `□` boxes | Tofu present in v0 | **16,000 diagrams + 96,000 OCR crops (0 Tofu)** | PASS |
