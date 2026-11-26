"""
Stage 3 Deterministic Geometric Primitive & Cubic Bezier Vectorizer +
Classical-Hybrid v0 Decomposer.
Supports:
  1. Perspective corner dewarp & adaptive shadow/illumination normalization
     (Sauvola + morphological background division removing phone shadows, yellow
     paper tint, faint notebook lines, and bleed-through).
  2. Morphological skeletonization guided by corner/junction keypoints.
  3. 6-channel semantic decomposition (either from Rekha-Net ONNX probabilities
     or the built-in Classical-Hybrid v0 heuristic separator).
  4. Primitive fitting (lines, circles, ellipses, snapped polylines, Schneider
     Cubic Beziers), dashed segment grouping, arrowhead detection, and callout-to-label
     graph association.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from skimage.morphology import skeletonize

from .primitives import (
    Point,
    associate_callouts_to_labels,
    classify_contour_primitive,
    douglas_peucker,
)
from .svg_builder import build_interactive_svg


@dataclass
class VectorizerConfig:
    target_size: int = 512
    sauvola_window: int = 31
    sauvola_k: float = 0.16
    min_stroke_len: int = 8
    dash_max_len: int = 24
    dash_gap_max: float = 22.0
    simplify_eps: float = 2.0
    theme: str = "print"


class RekhaVectorizer:
    def __init__(self, config: Optional[VectorizerConfig] = None):
        self.cfg = config or VectorizerConfig()

    def normalize_illumination(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Removes strong directional camera shadows, warm lighting casts, faint ruled lines,
        and page bleed-through via morphological background division + Sauvola thresholding.
        Returns (clean_gray_uint8, clean_binary_ink_mask_uint8).
        """
        if img_bgr.ndim == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr.copy()

        # Estimate slow-varying paper illumination & shadow envelope via large morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        bg = cv2.GaussianBlur(bg, (21, 21), 0)
        norm_gray = cv2.divide(gray, np.maximum(bg, 15), scale=255)

        # Suppress faint horizontal notebook lines and high-frequency halftone grain
        denoised = cv2.fastNlMeansDenoising(norm_gray, None, h=7, templateWindowSize=7, searchWindowSize=15)

        # Fast integral-image Sauvola adaptive thresholding
        w = self.cfg.sauvola_window
        f_img = denoised.astype(np.float32)
        mean = cv2.boxFilter(f_img, ddepth=-1, ksize=(w, w))
        sq_mean = cv2.boxFilter(f_img * f_img, ddepth=-1, ksize=(w, w))
        std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))
        thresh = mean * (1.0 + self.cfg.sauvola_k * ((std / 128.0) - 1.0))
        ink_mask = ((f_img < thresh) & (norm_gray < 218)).astype(np.uint8) * 255

        # Remove tiny isolated speckle noise (< 5 px)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 5:
                ink_mask[labels == i] = 0

        return norm_gray, ink_mask

    def detect_junctions_and_corners(self, skel: np.ndarray, gray: np.ndarray) -> List[Point]:
        """
        Combines skeleton topological branch points (>=3 neighbors) with Shi-Tomasi
        corners so T-junctions and 90-degree vertices snap crisply.
        """
        k = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
        s01 = (skel > 0).astype(np.uint8)
        neigh = cv2.filter2D(s01, -1, k)
        branch_pts = np.argwhere(neigh >= 13)
        junctions: List[Point] = [(float(x), float(y)) for y, x in branch_pts]

        corners = cv2.goodFeaturesToTrack(gray, maxCorners=120, qualityLevel=0.04, minDistance=7)
        if corners is not None:
            for c in corners:
                cx, cy = float(c[0][0]), float(c[0][1])
                junctions.append((cx, cy))
        return junctions

    def decompose_classical_6ch(
        self,
        img_bgr: np.ndarray,
        known_labels: Optional[List[Dict]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Classical-Hybrid v0 6-channel semantic decomposition on real textbook photos:
          ch0: solid strokes
          ch1: dashed/construction lines
          ch2: arrowheads/callout pointers
          ch3: shaded region fills
          ch4: text/math/Devanagari label boxes
          ch5: corner/junction keypoints
        """
        h, w = img_bgr.shape[:2]
        norm_gray, ink = self.normalize_illumination(img_bgr)
        ch_strokes = np.zeros((h, w), dtype=np.uint8)
        ch_dashed = np.zeros((h, w), dtype=np.uint8)
        ch_arrows = np.zeros((h, w), dtype=np.uint8)
        ch_fills = np.zeros((h, w), dtype=np.uint8)
        ch_text = np.zeros((h, w), dtype=np.uint8)
        ch_junctions = np.zeros((h, w), dtype=np.uint8)

        # 1. Detect mid-tone shaded fills (regions darker than paper background but lighter than ink)
        fill_cand = ((norm_gray >= 135) & (norm_gray <= 232) & (ink == 0)).astype(np.uint8) * 255
        fill_cand = cv2.morphologyEx(fill_cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        num_f, f_lbl, f_stats, _ = cv2.connectedComponentsWithStats(fill_cand, connectivity=8)
        for i in range(1, num_f):
            area = f_stats[i, cv2.CC_STAT_AREA]
            if 180 <= area <= int(0.35 * h * w):
                ch_fills[f_lbl == i] = 255

        # 2. Separate text boxes (either provided or connected-component glyph grouping)
        work_ink = ink.copy()
        if known_labels:
            for lb in known_labels:
                bx, by, bw, bh = [int(round(v)) for v in lb.get("bbox", [0, 0, 0, 0])]
                pad = 2
                x0, y0 = max(0, bx - pad), max(0, by - pad)
                x1, y1 = min(w, bx + bw + pad), min(h, by + bh + pad)
                ch_text[y0:y1, x0:x1] = 255
                work_ink[y0:y1, x0:x1] = 0
        else:
            num_c, c_lbl, c_stats, _ = cv2.connectedComponentsWithStats(work_ink, connectivity=8)
            for i in range(1, num_c):
                cw = c_stats[i, cv2.CC_STAT_WIDTH]
                ch = c_stats[i, cv2.CC_STAT_HEIGHT]
                area = c_stats[i, cv2.CC_STAT_AREA]
                # Compact character-sized glyph components
                if 5 <= cw <= 26 and 7 <= ch <= 28 and 12 <= area <= 340:
                    fill_ratio = area / float(max(cw * ch, 1))
                    if 0.18 <= fill_ratio <= 0.78:
                        ch_text[c_lbl == i] = 255

        # 3. Separate dashed segments, arrowheads, and continuous strokes
        num_c, c_lbl, c_stats, _ = cv2.connectedComponentsWithStats(work_ink, connectivity=8)
        for i in range(1, num_c):
            if np.any(ch_text[c_lbl == i] > 0):
                continue
            cw = c_stats[i, cv2.CC_STAT_WIDTH]
            ch = c_stats[i, cv2.CC_STAT_HEIGHT]
            area = c_stats[i, cv2.CC_STAT_AREA]
            diag = math.hypot(cw, ch)
            comp_mask = (c_lbl == i).astype(np.uint8) * 255
            if 6 <= diag <= 22 and area <= 75:
                ch_dashed[c_lbl == i] = 255
            else:
                ch_strokes[c_lbl == i] = 255
                # Detect triangular arrowhead tips along stroke ends
                dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 3)
                if np.max(dist) >= 2.8:
                    ch_arrows[dist >= 2.6] = 255

        skel = skeletonize(ch_strokes > 0).astype(np.uint8) * 255
        j_pts = self.detect_junctions_and_corners(skel, norm_gray)
        for jx, jy in j_pts:
            ix, iy = int(round(jx)), int(round(jy))
            if 0 <= ix < w and 0 <= iy < h:
                cv2.circle(ch_junctions, (ix, iy), 2, 255, -1)

        return {
            "strokes": ch_strokes,
            "dashed": ch_dashed,
            "arrows": ch_arrows,
            "fills": ch_fills,
            "text": ch_text,
            "junctions": ch_junctions,
            "norm_gray": norm_gray,
            "ink": ink,
        }

    def _trace_skeleton_chains(self, binary_mask: np.ndarray, min_len: int = 8) -> List[List[Point]]:
        """
        Extracts ordered 1-pixel centerline chains and closed loops from a stroke mask.
        """
        if np.count_nonzero(binary_mask) == 0:
            return []
        skel = skeletonize(binary_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        chains: List[List[Point]] = []
        for cnt in contours:
            if len(cnt) < min_len:
                continue
            pts_raw = [(float(p[0][0]), float(p[0][1])) for p in cnt]
            # FindContours on a 1px skeleton traces out and back; take the forward half if open
            half_n = len(pts_raw) // 2
            if half_n >= 4 and math.hypot(pts_raw[0][0] - pts_raw[half_n][0], pts_raw[0][1] - pts_raw[half_n][1]) > 6.0:
                pts = pts_raw[: half_n + 1]
            else:
                pts = pts_raw
            chains.append(pts)
        return chains

    def _group_dashed_chains(self, dashed_mask: np.ndarray) -> List[Dict]:
        """Groups colinear short dash blobs into unified dashed line primitives."""
        num_d, _, stats, centroids = cv2.connectedComponentsWithStats(dashed_mask, connectivity=8)
        if num_d <= 1:
            return []
        centers = [(float(centroids[i][0]), float(centroids[i][1])) for i in range(1, num_d)]
        used = [False] * len(centers)
        dashed_prims: List[Dict] = []

        for i in range(len(centers)):
            if used[i]:
                continue
            used[i] = True
            chain = [centers[i]]
            while True:
                last = chain[-1]
                best_j = None
                best_d = self.cfg.dash_gap_max
                for j in range(len(centers)):
                    if used[j]:
                        continue
                    d = math.hypot(centers[j][0] - last[0], centers[j][1] - last[1])
                    if d < best_d:
                        if len(chain) >= 2:
                            prev = chain[-2]
                            v1 = (last[0] - prev[0], last[1] - prev[1])
                            v2 = (centers[j][0] - last[0], centers[j][1] - last[1])
                            n1 = math.hypot(*v1)
                            n2 = math.hypot(*v2)
                            if n1 > 1e-3 and n2 > 1e-3:
                                cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
                                if cos_a < 0.78:
                                    continue
                        best_d = d
                        best_j = j
                if best_j is not None:
                    used[best_j] = True
                    chain.append(centers[best_j])
                else:
                    break
            if len(chain) >= 2:
                dashed_prims.append({
                    "type": "line" if len(chain) == 2 else "polyline",
                    "x1": round(chain[0][0], 2),
                    "y1": round(chain[0][1], 2),
                    "x2": round(chain[-1][0], 2),
                    "y2": round(chain[-1][1], 2),
                    "points": [(round(p[0], 2), round(p[1], 2)) for p in douglas_peucker(chain, 2.5)],
                    "dasharray": "6,5",
                })
        return dashed_prims

    def vectorize(
        self,
        img_bgr: np.ndarray,
        semantic_channels: Optional[Dict[str, np.ndarray]] = None,
        detected_labels: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Full Stage 3 compilation from image + semantic masks -> interactive SVG + telemetry.
        """
        t0 = time.perf_counter()
        h, w = img_bgr.shape[:2]
        ch = semantic_channels or self.decompose_classical_6ch(img_bgr, known_labels=detected_labels)

        # Extract junctions
        j_pts: List[Point] = []
        if "junctions" in ch and np.count_nonzero(ch["junctions"]) > 0:
            num_j, _, _, j_cent = cv2.connectedComponentsWithStats(ch["junctions"], connectivity=8)
            j_pts = [(float(j_cent[i][0]), float(j_cent[i][1])) for i in range(1, num_j)]

        # 1. Compile Fills
        fills_out: List[Dict] = []
        if "fills" in ch and np.count_nonzero(ch["fills"]) > 0:
            f_cnts, _ = cv2.findContours(ch["fills"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in f_cnts:
                if cv2.contourArea(cnt) < 120:
                    continue
                pts = [(float(p[0][0]), float(p[0][1])) for p in cnt]
                pts.append(pts[0])
                prim = classify_contour_primitive(pts, junctions=j_pts, prefer_organic=True)
                if prim.get("type") != "empty":
                    fills_out.append(prim)

        # 2. Compile Solid Strokes & Arrows
        prefer_org = (metadata or {}).get("domain") == "biology"
        stroke_chains = self._trace_skeleton_chains(ch["strokes"], min_len=self.cfg.min_stroke_len)
        strokes_out: List[Dict] = []
        arrows_out: List[Dict] = []
        arrow_mask = ch.get("arrows", np.zeros((h, w), dtype=np.uint8))

        raw_node_count = 0
        compiled_node_count = 0

        for chain in stroke_chains:
            raw_node_count += len(chain)
            prim = classify_contour_primitive(chain, junctions=j_pts, prefer_organic=prefer_org)
            if prim.get("type") == "empty":
                continue
            if prim["type"] in ("circle", "ellipse", "line"):
                compiled_node_count += 2
            elif prim["type"] in ("polyline", "polygon"):
                compiled_node_count += len(prim.get("points", []))
            else:
                compiled_node_count += prim.get("node_count", 4)

            # Check if endpoint touches arrowhead channel
            p_start, p_end = chain[0], chain[-1]
            ex, ey = int(round(p_end[0])), int(round(p_end[1]))
            sx, sy = int(round(p_start[0])), int(round(p_start[1]))
            is_arrow_end = (0 <= ex < w and 0 <= ey < h and np.any(arrow_mask[max(0, ey - 5):min(h, ey + 6), max(0, ex - 5):min(w, ex + 6)] > 0))
            is_arrow_start = (0 <= sx < w and 0 <= sy < h and np.any(arrow_mask[max(0, sy - 5):min(h, sy + 6), max(0, sx - 5):min(w, sx + 6)] > 0))

            if prim["type"] == "line" and (is_arrow_end or is_arrow_start):
                if is_arrow_start and not is_arrow_end:
                    prim["x1"], prim["y1"], prim["x2"], prim["y2"] = prim["x2"], prim["y2"], prim["x1"], prim["y1"]
                arrows_out.append(prim)
            else:
                strokes_out.append(prim)

        # 3. Compile Dashed Lines
        dashed_out = self._group_dashed_chains(ch["dashed"])

        # 4. Associate Callout Arrows with Labels
        labels_in = detected_labels or []
        labels_out, arrows_out = associate_callouts_to_labels(labels_in, arrows_out)

        svg_str = build_interactive_svg(
            width=w,
            height=h,
            fills=fills_out,
            strokes=strokes_out,
            dashed=dashed_out,
            arrows=arrows_out,
            labels=labels_out,
            theme_name=self.cfg.theme,
            metadata=metadata,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        node_reduction = float(raw_node_count) / float(max(compiled_node_count, 1))

        return {
            "svg": svg_str,
            "width": w,
            "height": h,
            "fills": fills_out,
            "strokes": strokes_out,
            "dashed": dashed_out,
            "arrows": arrows_out,
            "labels": labels_out,
            "latency_ms": round(elapsed_ms, 2),
            "raw_nodes": raw_node_count,
            "compiled_nodes": compiled_node_count,
            "node_reduction_ratio": round(max(node_reduction, 1.0), 2),
        }
