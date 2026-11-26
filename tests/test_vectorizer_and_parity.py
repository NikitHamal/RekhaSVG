"""
Unit, Benchmark, and 1:1 Python <-> Browser JS Parity Tests for RekhaSVG Stage 3 Compiler.
"""
import json
import math
import os
import subprocess
import time
import cv2
import numpy as np

from rekhasvg.primitives import (
    associate_callouts_to_labels,
    classify_contour_primitive,
    douglas_peucker,
    fit_ellipse_algebraic,
    fit_schneider_cubic_bezier,
    fit_taubin_circle,
    snap_orthogonal_polyline,
)
from rekhasvg.svg_builder import build_interactive_svg
from rekhasvg.vectorizer import RekhaVectorizer, VectorizerConfig


def test_taubin_circle_and_ellipse():
    # Generate noisy points on a circle cx=180, cy=140, r=55
    angles = np.linspace(0, 2 * math.pi, 48)
    circle_pts = [(180.0 + 55.0 * math.cos(a), 140.0 + 55.0 * math.sin(a)) for a in angles]
    c_res = fit_taubin_circle(circle_pts)
    assert c_res is not None
    assert c_res["type"] == "circle"
    assert abs(c_res["cx"] - 180.0) < 0.5
    assert abs(c_res["cy"] - 140.0) < 0.5
    assert abs(c_res["r"] - 55.0) < 0.5

    # Generate points on an ellipse cx=200, cy=200, rx=80, ry=40
    ell_pts = [(200.0 + 80.0 * math.cos(a), 200.0 + 40.0 * math.sin(a)) for a in angles]
    assert fit_taubin_circle(ell_pts) is None
    e_res = fit_ellipse_algebraic(ell_pts)
    assert e_res is not None
    assert e_res["type"] == "ellipse"
    assert abs(e_res["rx"] - 80.0) < 3.0
    assert abs(e_res["ry"] - 40.0) < 3.0


def test_orthogonal_snap_and_bezier():
    raw_line = [(10.0, 20.0), (50.0, 20.8), (110.0, 21.2)]
    dp = douglas_peucker(raw_line, epsilon=2.0)
    snapped = snap_orthogonal_polyline(dp, junctions=[(110.0, 20.0)])
    assert len(snapped) == 2
    assert snapped[0][1] == snapped[1][1] == 20.0

    wave = [(x, 100.0 + 25.0 * math.sin(x / 20.0)) for x in np.linspace(20, 220, 40)]
    segs = fit_schneider_cubic_bezier(wave)
    assert 2 <= len(segs) < 15


def test_python_to_js_exact_parity():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    circle_pts = [
        [round(150.0 + 45.0 * math.cos(a), 3), round(160.0 + 45.0 * math.sin(a), 3)]
        for a in np.linspace(0, 2 * math.pi, 36)
    ]
    poly_pts = [[20.0, 30.0], [120.0, 31.0], [120.5, 110.0]]
    labels = [{"id": "callout-1", "text": "उत्तल लेन्स (Convex Lens)", "bbox": [180, 40, 140, 22], "x": 182, "y": 56, "lang": "ne"}]
    arrows = [{"type": "line", "x1": 175.0, "y1": 50.0, "x2": 130.0, "y2": 95.0}]

    py_circle = classify_contour_primitive([tuple(p) for p in circle_pts])
    py_poly = classify_contour_primitive([tuple(p) for p in poly_pts])
    py_lbl, py_arr = associate_callouts_to_labels(labels, arrows)
    py_svg = build_interactive_svg(
        width=512,
        height=512,
        fills=[],
        strokes=[py_circle, py_poly],
        dashed=[{"type": "line", "x1": 40, "y1": 250, "x2": 460, "y2": 250, "dasharray": "6,5"}],
        arrows=py_arr,
        labels=py_lbl,
        theme_name="print",
        metadata={"domain": "physics", "title": "Parity Test"},
    )

    js_script = f"""
import {{ classifyContourPrimitive, associateCalloutsToLabels }} from './docs/engine/primitives.js';
import {{ buildInteractiveSvg }} from './docs/engine/svg_builder.js';

const circlePts = {json.dumps(circle_pts)};
const polyPts = {json.dumps(poly_pts)};
const labels = {json.dumps(labels)};
const arrows = {json.dumps(arrows)};

const jsCircle = classifyContourPrimitive(circlePts);
const jsPoly = classifyContourPrimitive(polyPts);
const [jsLbl, jsArr] = associateCalloutsToLabels(labels, arrows);
const jsSvg = buildInteractiveSvg({{
  width: 512,
  height: 512,
  fills: [],
  strokes: [jsCircle, jsPoly],
  dashed: [{{ type: 'line', x1: 40, y1: 250, x2: 460, y2: 250, dasharray: '6,5' }}],
  arrows: jsArr,
  labels: jsLbl,
  themeName: 'print',
  metadata: {{ domain: 'physics', title: 'Parity Test' }},
}});
console.log(JSON.stringify({{ circle: jsCircle, poly: jsPoly, svg: jsSvg }}));
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", js_script],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    js_out = json.loads(proc.stdout.strip())
    assert js_out["circle"]["type"] == py_circle["type"] == "circle"
    assert abs(js_out["circle"]["cx"] - py_circle["cx"]) < 1e-2
    assert abs(js_out["circle"]["r"] - py_circle["r"]) < 1e-2
    assert js_out["poly"]["points"] == [list(p) for p in py_poly["points"]]
    assert js_out["svg"] == py_svg


def test_end_to_end_vectorizer_latency_and_layers():
    # Create synthetic degraded camera diagram with shadow gradient
    h, w = 512, 512
    yy, xx = np.mgrid[0:h, 0:w]
    shadow = (235 - 65 * (xx / float(w)) - 35 * (yy / float(h))).astype(np.uint8)
    img = cv2.cvtColor(shadow, cv2.COLOR_GRAY2BGR)
    cv2.circle(img, (256, 200), 65, (20, 20, 20), 2)
    cv2.rectangle(img, (120, 310), (380, 410), (25, 25, 25), 2)
    for x in range(80, 430, 18):
        cv2.line(img, (x, 200), (x + 9, 200), (30, 30, 30), 2)

    vec = RekhaVectorizer(VectorizerConfig(theme="print"))
    res = vec.vectorize(
        img,
        detected_labels=[
            {"id": "c1", "text": "उत्तल लेन्स (Convex Lens)", "bbox": [210, 95, 130, 22], "x": 212, "y": 112, "lang": "ne"}
        ],
        metadata={"domain": "physics", "title": "Optics Test"},
    )
    assert res["latency_ms"] < 300.0
    assert res["node_reduction_ratio"] >= 2.5
    svg = res["svg"]
    for layer_id in ("layer-fills", "layer-strokes", "layer-dashed", "layer-arrows", "layer-labels"):
        assert f'id="{layer_id}"' in svg
    assert "उत्तल लेन्स" in svg


def test_electrodialysis_guardrails_and_zero_tofu():
    from scripts.gen_textbook_dataset import synthesize_electrodialysis_diagram, verify_zero_tofu

    # 1. Zero-tofu assertion across Greek, math, subscript, and Devanagari glyphs
    tofu_res = verify_zero_tofu()
    assert tofu_res["zero_tofu_verified"] is True

    # 2. Tiny open arc near branch cut (+/- pi) must NEVER snap into a giant Taubin circle
    tiny_arc = [
        (100.0 + 60.0 * math.cos(a), 100.0 + 60.0 * math.sin(a))
        for a in np.linspace(math.pi - 0.08, math.pi + 0.08, 12)
    ]
    assert fit_taubin_circle(tiny_arc) is None

    # 3. Electrodialysis colored diagram must have zero giant circle false positives and zero water zig-zags
    s = synthesize_electrodialysis_diagram(seed=2026, clean_digital=True)
    vec = RekhaVectorizer(VectorizerConfig(theme="print"))
    res = vec.vectorize(s["degraded_bgr"], detected_labels=s["labels"], metadata={"domain": "chemistry"})
    for st in res["strokes"]:
        if st.get("type") == "circle":
            assert st["r"] <= 20.0, f"Unexpected giant circle r={st['r']} on Electrodialysis diagram!"
    assert len(res["dashed"]) == 0, "Blue water-hatch dashes must not chain into structural dashed zig-zags!"
    assert "Anode" in res["svg"]
    assert "Bowman" not in res["svg"]
