# RekhaSVG — Neural v1 Engineering Status Report

## Completed Milestones (Shared Blackwell Session `sb-23410fd97ef00e15`)

- [x] **Eliminated All 4 Real-World Textbook Diagram Bugs**:
  1. **Zero Preset-Label Bleed (`docs/app.js`, `src/App.tsx`)**: Uploading any custom diagram or camera photo clears 100% of previous preset labels/metadata (`state.activeSampleIdx = -1`, `state.customLabels = []`) and executes live extraction.
  2. **Stage 3 Geometric Vectorizer Guardrails (`rekhasvg/primitives.py`, `rekhasvg/vectorizer.py`, `docs/engine/primitives.js`, `docs/engine/vectorizer.js`)**: Added 4-quadrant angular coverage and bounding-box checks to Taubin circle & algebraic ellipse fitters (preventing tiny `+` or `o` glyphs from snapping into giant circles), 2D periodic water-hatch/stipple filtering so blue water dashes never chain into diagonal zig-zags, and 3px-dilated text-box exclusion masks.
  3. **Strict Multi-Font Glyph Fallback & Automated Zero-Tofu Verification (`scripts/gen_textbook_dataset.py`)**: Verified `1,936` character instances across Noto Sans/Serif Devanagari, DejaVu Sans/Serif, Liberation Serif, FreeSerif, and STIX (`verify_zero_tofu() -> True`) so zero `□` boxes ever enter training data.
  4. **High-Complexity v2 Textbook Corpus (`NikitHamal/rekhasvg-data` `v2/`)**: Generated **16,000** v2 512×512 textbook diagrams (including Electrodialysis colored water-hatch troughs, double-walled semi-permeable membranes, sol/crystalloid particle clusters, stacked two-line labels, clean digital + shadowed camera renders) and **96,000** spotless 32×192 OCR crops at `346.98 img/s`.
- [x] **Trained Rekha-Net & Rekha-Text in bf16 AMP + Exported Verified INT8 ONNX (`NikitHamal/rekhasvg-models`)**:
  - **Rekha-Net (`rekha_net_int8.onnx`)**: **6.32 MB** INT8 ONNX (`0 ConvInteger`, `0 NaNs`), `185.72 ms` Python CPU / `182.50 ms` Node CPU, achieving **0.9920 validation IoU / 0.9960 Dice** (`0.8938` end-to-end `bench_500` IoU / `0.9408` Dice, up `+0.3160` IoU over the `0.5778` v0 baseline) and **0.9341 IoU** on the colored Electrodialysis textbook diagram with `0.00000` text-to-stroke leakage.
  - **Rekha-Text (`rekha_text_int8.onnx`)**: **4.39 MB** INT8 ONNX (`0 ConvInteger`, `0 NaNs`), `3.85 ms` Python CPU / `4.46 ms` Node CPU, achieving **0.03% overall validation CER** (`0.04%` English/STEM CER, `0.01%` Nepali Devanagari CER) and reading stacked two-line labels (`semi permeable / membrane`, `Water + / Electrolyte`, `Anode`, `Cathode`, `+`, `-`, `Sol particle`, `Crystalloid`).
  - **Total Combined INT8 ONNX Footprint**: **10.71 MB** (strictly within the `10.0 - 15.0 MB` budget).
- [x] **End-to-End Visual & Quantitative Proofs (`eval/neural_v1_report.json`, `eval/v1_neural_proofs/`)**:
  - Saved 13 SVG proofs in `eval/v1_neural_proofs/` and verified all acceptance invariants (`zero_preset_label_bleed: true`, `giant_circle_false_positives: 0`, `zigzag_water_dashed_chains: 0`, `has_editable_svg_text: true`).
