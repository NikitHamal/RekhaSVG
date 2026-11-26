#!/usr/bin/env python3
"""
Procedural Nepal Textbook Diagram Data Factory (scripts/gen_textbook_dataset.py)
Generates:
  1. Unseen 500-diagram holdout benchmark (eval/bench_500/) covering all 4 STEM
     domains (Physics, Biology, Chemistry, Math/CS) + 20 camera stress tests.
  2. 32,000 paired training samples packed into compressed shards and backed up to
     the private Hugging Face dataset repo `NikitHamal/rekhasvg-data` every 15 minutes
     (>= 900 seconds interval to strictly avoid HF 429 rate limits).
  3. Visual preview triptych (/work/rekhasvg/eval/preview_triptych.png) for the
     live Marimo notebook telemetry card.
"""
from __future__ import annotations
import json
import math
import multiprocessing as mp
import os
import random
import time
from typing import Dict, List, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rekhasvg.svg_builder import build_interactive_svg

FONT_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
    "/usr/share/fonts/opentype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

STEM_VOCAB = {
    "physics": [
        ("उत्तल लेन्स (Convex Lens)", "ne"),
        ("प्रकाश किरण (Light Ray)", "ne"),
        ("प्रतिरोध R₁ = 10 Ω", "ne"),
        ("फोकस दूरी (2F₁)", "ne"),
        ("भोल्टमिटर V = IR", "ne"),
        ("विद्युत धारा (I = 2A)", "ne"),
        ("θ = 45°", "en"),
        ("F = ma", "en"),
        ("ΔΦ / Δt", "en"),
        ("C₁ = 4.7 μF", "en"),
    ],
    "biology": [
        ("कोष झिल्ली (Cell Membrane)", "ne"),
        ("न्यूक्लियस (Nucleus)", "ne"),
        ("माइटोकोन्ड्रिया (Mitochondria)", "ne"),
        ("रिक्तिका (Vacuole)", "ne"),
        ("हरितकण (Chloroplast)", "ne"),
        ("साइटोप्लाज्म (Cytoplasm)", "ne"),
        ("Axon Terminal", "en"),
        ("Nephron Loop", "en"),
    ],
    "chemistry": [
        ("अम्ल र क्षार (Acid & Base)", "ne"),
        ("परमाणु केन्दक (Atomic Nucleus)", "ne"),
        ("आसवन फ्लास्क (Distillation)", "ne"),
        ("H₂SO₄ + 2NaOH", "en"),
        ("2H₂O (l)", "en"),
        ("ΔH = -92 kJ/mol", "en"),
        ("Benzene C₆H₆", "en"),
        ("pH = 7.4", "en"),
    ],
    "math_cs": [
        ("समकोण त्रिभुज (Right ΔABC)", "ne"),
        ("केन्द्र O (Center)", "ne"),
        ("कोण α + β = 90°", "ne"),
        ("f(x) = sin(θ)", "en"),
        ("A ∩ B", "en"),
        ("NAND Gate", "en"),
        ("y = mx + c", "en"),
        ("r = 5 cm", "en"),
    ],
}


def _load_pil_font(size: int = 15):
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_text_bilingual(
    canvas_bgr: np.ndarray,
    mask_text: np.ndarray,
    text: str,
    x: int,
    y: int,
    font_size: int = 15,
) -> List[int]:
    """Draws bilingual Devanagari/English/Math text onto BGR image & updates text channel mask."""
    h, w = canvas_bgr.shape[:2]
    pil_img = Image.fromarray(cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _load_pil_font(font_size)
    bbox = draw.textbbox((x, y), text, font=font)
    bx0, by0, bx1, by1 = [int(v) for v in bbox]
    bx0 = max(2, min(w - 20, bx0))
    by0 = max(2, min(h - 16, by0))
    bx1 = max(bx0 + 12, min(w - 2, bx1))
    by1 = max(by0 + 10, min(h - 2, by1))
    draw.text((x, y), text, fill=(22, 20, 26), font=font)
    canvas_bgr[:] = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    mask_text[by0:by1, bx0:bx1] = 255
    return [bx0, by0, bx1 - bx0, by1 - by0]


def draw_dashed_line(
    img: np.ndarray,
    mask_dashed: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    dash_len: int = 9,
    gap_len: int = 7,
    thickness: int = 2,
):
    dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if dist < 1e-3:
        return
    dx = (pt2[0] - pt1[0]) / dist
    dy = (pt2[1] - pt1[1]) / dist
    cur = 0.0
    while cur < dist:
        end_d = min(dist, cur + dash_len)
        x0 = int(round(pt1[0] + dx * cur))
        y0 = int(round(pt1[1] + dy * cur))
        x1 = int(round(pt1[0] + dx * end_d))
        y1 = int(round(pt1[1] + dy * end_d))
        cv2.line(img, (x0, y0), (x1, y1), (35, 35, 42), thickness, cv2.LINE_AA)
        cv2.line(mask_dashed, (x0, y0), (x1, y1), 255, thickness)
        cur += dash_len + gap_len


def draw_callout_arrow(
    img: np.ndarray,
    mask_strokes: np.ndarray,
    mask_arrows: np.ndarray,
    tail: Tuple[int, int],
    head: Tuple[int, int],
    thickness: int = 2,
):
    cv2.line(img, tail, head, (30, 25, 45), thickness, cv2.LINE_AA)
    cv2.line(mask_strokes, tail, head, 255, thickness)
    ang = math.atan2(head[1] - tail[1], head[0] - tail[0])
    tip_len = 10.0
    p_left = (
        int(round(head[0] - tip_len * math.cos(ang - 0.45))),
        int(round(head[1] - tip_len * math.sin(ang - 0.45))),
    )
    p_right = (
        int(round(head[0] - tip_len * math.cos(ang + 0.45))),
        int(round(head[1] - tip_len * math.sin(ang + 0.45))),
    )
    tri = np.array([head, p_left, p_right], dtype=np.int32)
    cv2.fillPoly(img, [tri], (30, 25, 45))
    cv2.fillPoly(mask_arrows, [tri], 255)


def synthesize_stem_diagram(seed: int, domain: str = None, stress_mode: bool = False) -> Dict:
    """
    Synthesizes a complete Nepal STEM textbook diagram with:
      - clean_bgr (512x512)
      - degraded_bgr (512x512 realistic phone camera capture)
      - masks_6ch (6, 512, 512) uint8: [strokes, dashed, arrows, fills, text, junctions]
      - gt_svg (ground-truth interactive SVG string)
      - labels (list of bbox + bilingual text + callout_id)
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    domains = ["physics", "biology", "chemistry", "math_cs"]
    dom = domain or domains[seed % len(domains)]
    h, w = 512, 512

    clean = np.full((h, w, 3), 255, dtype=np.uint8)
    m_strokes = np.zeros((h, w), dtype=np.uint8)
    m_dashed = np.zeros((h, w), dtype=np.uint8)
    m_arrows = np.zeros((h, w), dtype=np.uint8)
    m_fills = np.zeros((h, w), dtype=np.uint8)
    m_text = np.zeros((h, w), dtype=np.uint8)
    m_junctions = np.zeros((h, w), dtype=np.uint8)

    gt_fills, gt_strokes, gt_dashed, gt_arrows, gt_labels = [], [], [], [], []
    vocab = list(STEM_VOCAB[dom])
    rng.shuffle(vocab)

    if dom == "physics":
        subtype = seed % 2
        if subtype == 0:
            # Ray Optics: Convex Lens + Principal Axis + Focal Rays
            cx, cy = rng.randint(235, 275), rng.randint(235, 275)
            rx, ry = rng.randint(22, 30), rng.randint(95, 120)
            cv2.ellipse(clean, (cx, cy), (rx, ry), 0, 0, 360, (225, 218, 245), -1)
            cv2.ellipse(m_fills, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
            cv2.ellipse(clean, (cx, cy), (rx, ry), 0, 0, 360, (20, 20, 25), 2, cv2.LINE_AA)
            cv2.ellipse(m_strokes, (cx, cy), (rx, ry), 0, 0, 360, 255, 2)
            gt_fills.append({"type": "ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry})
            gt_strokes.append({"type": "ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry})

            # Dashed Principal Axis
            draw_dashed_line(clean, m_dashed, (45, cy), (465, cy))
            gt_dashed.append({"type": "line", "x1": 45, "y1": cy, "x2": 465, "y2": cy, "dasharray": "6,5"})

            # Incident & refracted rays
            obj_x, obj_y = cx - 130, cy - 65
            img_x, img_y = cx + 135, cy + 65
            for p_a, p_b in [((obj_x, cy), (obj_x, obj_y)), ((obj_x, obj_y), (cx, obj_y)), ((cx, obj_y), (img_x, img_y)), ((obj_x, obj_y), (img_x, img_y))]:
                cv2.line(clean, p_a, p_b, (20, 20, 25), 2, cv2.LINE_AA)
                cv2.line(m_strokes, p_a, p_b, 255, 2)
                cv2.circle(m_junctions, p_a, 3, 255, -1)
                cv2.circle(m_junctions, p_b, 3, 255, -1)
                gt_strokes.append({"type": "line", "x1": p_a[0], "y1": p_a[1], "x2": p_b[0], "y2": p_b[1]})
            targets = [(cx, cy - ry + 10), (obj_x, obj_y), (cx + 70, cy)]
        else:
            # DC Circuit Loop with Battery, Resistor, Voltmeter Circle
            x0, y0, x1, y1 = 105, 135, 405, 375
            cv2.rectangle(clean, (x0, y0), (x1, y1), (20, 20, 25), 2)
            cv2.rectangle(m_strokes, (x0, y0), (x1, y1), 255, 2)
            for jx, jy in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]:
                cv2.circle(m_junctions, (jx, jy), 3, 255, -1)
            gt_strokes.append({"type": "polygon", "points": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]})

            vm_cx, vm_cy, vm_r = (x0 + x1) // 2, y1, 28
            cv2.circle(clean, (vm_cx, vm_cy), vm_r, (232, 222, 248), -1)
            cv2.circle(m_fills, (vm_cx, vm_cy), vm_r, 255, -1)
            cv2.circle(clean, (vm_cx, vm_cy), vm_r, (20, 20, 25), 2, cv2.LINE_AA)
            cv2.circle(m_strokes, (vm_cx, vm_cy), vm_r, 255, 2)
            gt_fills.append({"type": "circle", "cx": vm_cx, "cy": vm_cy, "r": vm_r})
            gt_strokes.append({"type": "circle", "cx": vm_cx, "cy": vm_cy, "r": vm_r})
            draw_dashed_line(clean, m_dashed, (x0 + 30, (y0 + y1) // 2), (x1 - 30, (y0 + y1) // 2))
            gt_dashed.append({"type": "line", "x1": x0 + 30, "y1": (y0 + y1) // 2, "x2": x1 - 30, "y2": (y0 + y1) // 2, "dasharray": "6,5"})
            targets = [((x0 + x1) // 2, y0), (vm_cx, vm_cy - vm_r), (x1, (y0 + y1) // 2)]

    elif dom == "biology":
        # Cell / Organelle Diagram with smooth organic membrane + nucleus + organelles
        cx, cy = 256, 260
        rx, ry = rng.randint(125, 145), rng.randint(95, 115)
        cv2.ellipse(clean, (cx, cy), (rx, ry), 0, 0, 360, (235, 228, 248), -1)
        cv2.ellipse(m_fills, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        cv2.ellipse(clean, (cx, cy), (rx, ry), 0, 0, 360, (20, 20, 25), 2, cv2.LINE_AA)
        cv2.ellipse(m_strokes, (cx, cy), (rx, ry), 0, 0, 360, 255, 2)
        gt_fills.append({"type": "ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry})
        gt_strokes.append({"type": "ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry})

        # Nucleus circle
        ncx, ncy, nr = cx - 35, cy - 10, 36
        cv2.circle(clean, (ncx, ncy), nr, (215, 202, 242), -1)
        cv2.circle(m_fills, (ncx, ncy), nr, 255, -1)
        cv2.circle(clean, (ncx, ncy), nr, (20, 20, 25), 2, cv2.LINE_AA)
        cv2.circle(m_strokes, (ncx, ncy), nr, 255, 2)
        gt_fills.append({"type": "circle", "cx": ncx, "cy": ncy, "r": nr})
        gt_strokes.append({"type": "circle", "cx": ncx, "cy": ncy, "r": nr})

        # Mitochondria oval
        mcx, mcy = cx + 55, cy + 20
        cv2.ellipse(clean, (mcx, mcy), (28, 16), 25, 0, 360, (20, 20, 25), 2, cv2.LINE_AA)
        cv2.ellipse(m_strokes, (mcx, mcy), (28, 16), 25, 0, 360, 255, 2)
        gt_strokes.append({"type": "ellipse", "cx": mcx, "cy": mcy, "rx": 28, "ry": 16, "angle": 25})
        draw_dashed_line(clean, m_dashed, (cx - rx + 20, cy + 45), (cx + rx - 20, cy + 45))
        gt_dashed.append({"type": "line", "x1": cx - rx + 20, "y1": cy + 45, "x2": cx + rx - 20, "y2": cy + 45, "dasharray": "6,5"})
        targets = [(cx, cy - ry), (ncx, ncy - nr), (mcx, mcy - 16)]

    elif dom == "chemistry":
        # Distillation / Titration Flask + Bohr shell or reaction level
        fx, fy, fr = 240, 300, 62
        cv2.circle(clean, (fx, fy), fr, (230, 224, 248), -1)
        cv2.circle(m_fills, (fx, fy), fr, 255, -1)
        cv2.circle(clean, (fx, fy), fr, (20, 20, 25), 2, cv2.LINE_AA)
        cv2.circle(m_strokes, (fx, fy), fr, 255, 2)
        gt_fills.append({"type": "circle", "cx": fx, "cy": fy, "r": fr})
        gt_strokes.append({"type": "circle", "cx": fx, "cy": fy, "r": fr})

        # Flask neck & condenser tube
        neck = [((fx - 16, fy - 58), (fx - 16, fy - 135)), ((fx + 16, fy - 58), (fx + 16, fy - 135)), ((fx + 16, fy - 110), (fx + 135, fy - 65))]
        for p_a, p_b in neck:
            cv2.line(clean, p_a, p_b, (20, 20, 25), 2, cv2.LINE_AA)
            cv2.line(m_strokes, p_a, p_b, 255, 2)
            cv2.circle(m_junctions, p_a, 3, 255, -1)
            cv2.circle(m_junctions, p_b, 3, 255, -1)
            gt_strokes.append({"type": "line", "x1": p_a[0], "y1": p_a[1], "x2": p_b[0], "y2": p_b[1]})
        draw_dashed_line(clean, m_dashed, (fx - fr + 10, fy + 15), (fx + fr - 10, fy + 15))
        gt_dashed.append({"type": "line", "x1": fx - fr + 10, "y1": fy + 15, "x2": fx + fr - 10, "y2": fy + 15, "dasharray": "6,5"})
        targets = [(fx, fy - fr), (fx + 85, fy - 85), (fx + fr, fy)]

    else:
        # Math / CS: Inscribed Circle + Right Triangle + Cartesian Axes
        ox, oy = 120, 370
        p_top = (120, 150)
        p_right = (380, 370)
        tri_pts = [p_top, (ox, oy), p_right]
        for i in range(3):
            pa, pb = tri_pts[i], tri_pts[(i + 1) % 3]
            cv2.line(clean, pa, pb, (20, 20, 25), 2, cv2.LINE_AA)
            cv2.line(m_strokes, pa, pb, 255, 2)
            cv2.circle(m_junctions, pa, 3, 255, -1)
        gt_strokes.append({"type": "polygon", "points": tri_pts})

        icx, icy, ir = 205, 300, 48
        cv2.circle(clean, (icx, icy), ir, (232, 224, 248), -1)
        cv2.circle(m_fills, (icx, icy), ir, 255, -1)
        cv2.circle(clean, (icx, icy), ir, (20, 20, 25), 2, cv2.LINE_AA)
        cv2.circle(m_strokes, (icx, icy), ir, 255, 2)
        gt_fills.append({"type": "circle", "cx": icx, "cy": icy, "r": ir})
        gt_strokes.append({"type": "circle", "cx": icx, "cy": icy, "r": ir})
        draw_dashed_line(clean, m_dashed, (ox, oy), (260, 250))
        gt_dashed.append({"type": "line", "x1": ox, "y1": oy, "x2": 260, "y2": 250, "dasharray": "6,5"})
        targets = [p_top, (icx, icy - ir), p_right]

    # Add 3 callout arrows + bilingual English/Nepali labels
    label_positions = [(55, 42), (265, 52), (285, 435)]
    for idx in range(min(3, len(targets))):
        txt, lang = vocab[idx]
        lx, ly = label_positions[idx]
        bbox = draw_text_bilingual(clean, m_text, txt, lx, ly, font_size=15)
        tail = (bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] if ly < 250 else bbox[1])
        head = targets[idx]
        draw_callout_arrow(clean, m_strokes, m_arrows, tail, head)
        cid = f"callout-{idx + 1}"
        gt_arrows.append({"type": "line", "x1": tail[0], "y1": tail[1], "x2": head[0], "y2": head[1], "callout_id": cid})
        gt_labels.append({
            "id": cid,
            "callout_id": cid,
            "text": txt,
            "lang": lang,
            "bbox": bbox,
            "x": bbox[0] + 2,
            "y": bbox[1] + bbox[3] - 3,
            "font_size": 14.0,
        })

    # Degrade clean diagram into realistic smartphone camera textbook photo
    degraded = clean.astype(np.float32)

    # 1. Warm paper tint + faint notebook ruled lines
    tint = np.array([rng.uniform(0.90, 0.97), rng.uniform(0.95, 0.99), rng.uniform(0.97, 1.0)], dtype=np.float32)
    degraded *= tint
    if seed % 3 == 0 or stress_mode:
        for ry_line in range(40, h, 28):
            degraded[ry_line : ry_line + 1, :, :] *= 0.94

    # 2. Strong directional hand/phone shadow gradient
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ang = rng.uniform(0, 2 * math.pi)
    proj = (xx - 256.0) * math.cos(ang) + (yy - 256.0) * math.sin(ang)
    shadow_strength = rng.uniform(0.28, 0.45) if stress_mode else rng.uniform(0.15, 0.32)
    illum = 1.0 - shadow_strength * (1.0 / (1.0 + np.exp(-proj / 75.0)))
    degraded *= illum[:, :, None]

    # 3. Gaussian sensor grain & slight blur
    noise_sigma = 6.5 if stress_mode else 3.5
    degraded += np_rng.normal(0.0, noise_sigma, size=degraded.shape).astype(np.float32)
    degraded_u8 = np.clip(degraded, 0, 255).astype(np.uint8)
    if stress_mode:
        degraded_u8 = cv2.GaussianBlur(degraded_u8, (3, 3), 0.6)

    masks_6ch = np.stack([m_strokes, m_dashed, m_arrows, m_fills, m_text, m_junctions], axis=0)
    gt_svg = build_interactive_svg(
        width=w,
        height=h,
        fills=gt_fills,
        strokes=gt_strokes,
        dashed=gt_dashed,
        arrows=gt_arrows,
        labels=gt_labels,
        theme_name="print",
        metadata={"domain": dom, "title": f"Nepal STEM {dom.title()} #{seed}"},
    )

    return {
        "seed": seed,
        "domain": dom,
        "stress_mode": stress_mode,
        "clean_bgr": clean,
        "degraded_bgr": degraded_u8,
        "masks_6ch": masks_6ch,
        "labels": gt_labels,
        "gt_svg": gt_svg,
    }


def save_preview_triptych(sample: Dict, out_path: str):
    """Renders [Degraded Camera Photo | 6-Channel Semantic Composite | Clean Reconstructed Output] triptych."""
    deg = sample["degraded_bgr"]
    m6 = sample["masks_6ch"]
    h, w = deg.shape[:2]

    # Colorize 6-channel semantic target mask
    sem_vis = np.full((h, w, 3), 24, dtype=np.uint8)
    sem_vis[m6[3] > 0] = (140, 95, 195)   # Fills (violet)
    sem_vis[m6[0] > 0] = (240, 240, 245)  # Solid strokes (white)
    sem_vis[m6[1] > 0] = (90, 210, 255)   # Dashed lines (amber/gold)
    sem_vis[m6[2] > 0] = (120, 250, 140)  # Arrows (emerald)
    sem_vis[m6[4] > 0] = (255, 175, 100)  # Text boxes (sky blue)
    sem_vis[m6[5] > 0] = (80, 80, 255)    # Junction keypoints (red)

    clean = sample["clean_bgr"]
    sep = np.full((h, 8, 3), 210, dtype=np.uint8)
    triptych = np.concatenate([deg, sep, sem_vis, sep, clean], axis=1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, triptych)


def _worker_make_lite(args: Tuple[int, str]) -> Tuple[int, str, bytes, bytes, str]:
    seed, dom = args
    s = synthesize_stem_diagram(seed, domain=dom, stress_mode=(seed % 15 == 0))
    ok1, deg_jpg = cv2.imencode(".jpg", s["degraded_bgr"], [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    # Pack 6 binary channels into 1 uint8 bitmask image for 6x smaller shard storage & faster I/O
    packed_mask = np.zeros((512, 512), dtype=np.uint8)
    for ch_i in range(6):
        packed_mask |= ((s["masks_6ch"][ch_i] > 0).astype(np.uint8) << ch_i)
    ok2, mask_png = cv2.imencode(".png", packed_mask)
    meta_json = json.dumps({"seed": seed, "domain": dom, "labels": s["labels"]}, ensure_ascii=False)
    return seed, dom, deg_jpg.tobytes(), mask_png.tobytes(), meta_json


def sync_to_hf(shard_dir: str, status_dict: Dict):
    tok_path = "/tmp/hf_token"
    if not os.path.exists(tok_path):
        return False
    with open(tok_path, "r", encoding="utf-8") as f:
        token = f.read().strip()
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        repo_id = "NikitHamal/rekhasvg-data"
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(
            folder_path=shard_dir,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Sync RekhaSVG shards ({status_dict.get('total_generated', 0)} samples)",
        )
        status_dict["last_sync"] = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        return True
    except Exception as e:
        status_dict["last_sync"] = f"Sync warning: {type(e).__name__}"
        return False


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="/work/rekhasvg/data")
    parser.add_argument("--eval-dir", default="/work/rekhasvg/eval/bench_500")
    parser.add_argument("--status-file", default="/work/rekhasvg/eval/dataset_status.json")
    parser.add_argument("--target", type=int, default=32000)
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.status_file), exist_ok=True)

    # 1. Generate preview triptych immediately
    sample0 = synthesize_stem_diagram(101, domain="physics", stress_mode=True)
    save_preview_triptych(sample0, "/work/rekhasvg/eval/preview_triptych.png")

    # 2. Generate unseen 500-diagram benchmark + 20 stress tests first
    bench_manifest = []
    domains = ["physics", "biology", "chemistry", "math_cs"]
    for i in range(520):
        dom = domains[i % 4]
        is_stress = i >= 500
        s = synthesize_stem_diagram(900000 + i, domain=dom, stress_mode=is_stress)
        img_name = f"bench_{i:04d}_{dom}{'_stress' if is_stress else ''}.jpg"
        mask_name = f"bench_{i:04d}_mask.png"
        cv2.imwrite(os.path.join(args.eval_dir, img_name), s["degraded_bgr"], [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        packed_mask = np.zeros((512, 512), dtype=np.uint8)
        for ch_i in range(6):
            packed_mask |= ((s["masks_6ch"][ch_i] > 0).astype(np.uint8) << ch_i)
        cv2.imwrite(os.path.join(args.eval_dir, mask_name), packed_mask)
        bench_manifest.append({
            "id": i,
            "domain": dom,
            "stress_mode": is_stress,
            "img": img_name,
            "mask": mask_name,
            "labels": s["labels"],
        })
    with open(os.path.join(args.eval_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(bench_manifest, f, ensure_ascii=False, indent=2)

    # 3. Generate 32,000 training samples across workers in shards of 2,000
    status = {
        "total_generated": 0,
        "target": args.target,
        "domains": {"physics": 0, "biology": 0, "chemistry": 0, "math_cs": 0},
        "bench_500_count": 500,
        "stress_20_count": 20,
        "hf_synced_shards": 0,
        "hf_repo": "NikitHamal/rekhasvg-data",
        "last_sync": "Initializing first shard",
        "rate_per_sec": 0.0,
    }
    with open(args.status_file, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    t_start = time.time()
    last_hf_sync = 0.0
    num_shards = (args.target + args.shard_size - 1) // args.shard_size

    with mp.Pool(processes=args.workers) as pool:
        for shard_idx in range(num_shards):
            s_start = shard_idx * args.shard_size
            s_end = min(args.target, s_start + args.shard_size)
            tasks = [(10000 + k, domains[k % 4]) for k in range(s_start, s_end)]
            results = pool.map(_worker_make_lite, tasks, chunksize=32)

            jpg_list, mask_list, meta_list = [], [], []
            for seed, dom, deg_jpg, mask_png, meta_json in results:
                status["total_generated"] += 1
                status["domains"][dom] += 1
                jpg_list.append(np.frombuffer(deg_jpg, dtype=np.uint8))
                mask_list.append(np.frombuffer(mask_png, dtype=np.uint8))
                meta_list.append(meta_json)

            shard_path = os.path.join(args.out_dir, f"shard_{shard_idx:03d}.npz")
            np.savez_compressed(
                shard_path,
                jpgs=np.array(jpg_list, dtype=object),
                masks=np.array(mask_list, dtype=object),
                metas=np.array(meta_list, dtype=object),
            )
            elapsed = max(1e-3, time.time() - t_start)
            status["rate_per_sec"] = round(status["total_generated"] / elapsed, 2)

            # Sync on first shard, every 15 minutes (>= 900s), and on final shard
            now = time.time()
            if shard_idx == 0 or shard_idx == num_shards - 1 or (now - last_hf_sync) >= 900.0:
                if sync_to_hf(args.out_dir, status):
                    status["hf_synced_shards"] = shard_idx + 1
                last_hf_sync = now

            with open(args.status_file, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2)


if __name__ == "__main__":
    main()
