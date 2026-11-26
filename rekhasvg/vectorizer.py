"""
Stage 3 Deterministic Geometric Primitive & Cubic Bezier Vectorizer +
Neural 6-Channel Semantic Decomposer (Rekha-Net + Rekha-Text INT8 ONNX) &
Hardened Classical Fallback.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import os
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
from .rekha_text import REKHA_CHARSET
from .svg_builder import build_interactive_svg


@dataclass
class VectorizerConfig:
    target_size: int = 512
    sauvola_window: int = 31
    sauvola_k: float = 0.16
    min_stroke_len: int = 10
    dash_max_len: int = 24
    dash_gap_max: float = 22.0
    simplify_eps: float = 2.0
    theme: str = "print"


class RekhaVectorizer:
    def __init__(
        self,
        config: Optional[VectorizerConfig] = None,
        net_onnx_path: Optional[str] = None,
        text_onnx_path: Optional[str] = None,
    ):
        self.cfg = config or VectorizerConfig()
        self.net_sess = None
        self.text_sess = None
        if net_onnx_path or text_onnx_path:
            self.load_onnx_models(net_onnx_path, text_onnx_path)

    def load_onnx_models(
        self,
        net_onnx_path: Optional[str] = None,
        text_onnx_path: Optional[str] = None,
    ) -> bool:
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            if net_onnx_path and os.path.exists(net_onnx_path):
                self.net_sess = ort.InferenceSession(
                    net_onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
                )
            if text_onnx_path and os.path.exists(text_onnx_path):
                self.text_sess = ort.InferenceSession(
                    text_onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
                )
            return self.net_sess is not None
        except Exception:
            return False

    def normalize_illumination(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Removes directional camera shadows, warm paper tint, faint ruled lines,
        and bleed-through via morphological background division + Sauvola thresholding.
        """
        if img_bgr.ndim == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr.copy()

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        bg = cv2.GaussianBlur(bg, (21, 21), 0)
        norm_gray = cv2.divide(gray, np.maximum(bg, 15), scale=255)

        denoised = cv2.fastNlMeansDenoising(norm_gray, None, h=7, templateWindowSize=7, searchWindowSize=15)

        w = self.cfg.sauvola_window
        f_img = denoised.astype(np.float32)
        mean = cv2.boxFilter(f_img, ddepth=-1, ksize=(w, w))
        sq_mean = cv2.boxFilter(f_img * f_img, ddepth=-1, ksize=(w, w))
        std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))
        thresh = mean * (1.0 + self.cfg.sauvola_k * ((std / 128.0) - 1.0))
        ink_mask = ((f_img < thresh) & (norm_gray < 218)).astype(np.uint8) * 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(ink_mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 5:
                ink_mask[labels == i] = 0

        return norm_gray, ink_mask

    def detect_junctions_and_corners(self, skel: np.ndarray, gray: np.ndarray) -> List[Point]:
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

    def _split_multiline_text_boxes(self, text_mask: np.ndarray, ink_mask: np.ndarray) -> List[List[int]]:
        """
        Extracts individual single-line text bounding boxes [x, y, w, h] from a text region mask.
        Automatically splits stacked two-line or multi-line labels (e.g., 'semi permeable / membrane'
        or 'Water + / Electrolyte') via horizontal ink projection valleys.
        """
        h, w = text_mask.shape[:2]
        dil = cv2.dilate((text_mask > 0).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 3)))
        num_b, _, stats, _ = cv2.connectedComponentsWithStats(dil, connectivity=8)
        boxes: List[List[int]] = []
        for i in range(1, num_b):
            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            if bw < 6 or bh < 6 or bw * bh < 45:
                continue
            if bh >= 28:
                sub_ink = (ink_mask[by : by + bh, bx : bx + bw] > 0).astype(np.int32)
                row_proj = np.sum(sub_ink, axis=1)
                # Find horizontal valleys separating stacked lines
                thresh = max(1, int(0.08 * np.max(row_proj))) if np.max(row_proj) > 0 else 0
                in_line = False
                line_start = 0
                sub_lines = []
                for r_idx in range(bh):
                    if row_proj[r_idx] > thresh and not in_line:
                        in_line = True
                        line_start = r_idx
                    elif row_proj[r_idx] <= thresh and in_line:
                        if r_idx - line_start >= 6:
                            sub_lines.append((line_start, r_idx))
                        in_line = False
                if in_line and bh - line_start >= 6:
                    sub_lines.append((line_start, bh))
                if len(sub_lines) >= 2:
                    for ls, le in sub_lines:
                        y0 = max(0, by + ls - 2)
                        y1 = min(h, by + le + 2)
                        cols = np.where(np.sum(sub_ink[ls:le, :], axis=0) > 0)[0]
                        if len(cols) >= 2:
                            x0 = max(0, bx + int(cols[0]) - 2)
                            x1 = min(w, bx + int(cols[-1]) + 3)
                            boxes.append([x0, y0, x1 - x0, y1 - y0])
                    continue
            boxes.append([bx, by, bw, bh])
        boxes.sort(key=lambda b: (b[1] // 18, b[0]))
        return boxes

    def recognize_text_boxes(self, img_bgr: np.ndarray, boxes: List[List[int]]) -> List[Dict]:
        """Runs Rekha-Text INT8 ONNX CTC recognizer on each single-line text box."""
        if not boxes:
            return []
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        h, w = gray.shape[:2]
        labels_out: List[Dict] = []
        for idx, (bx, by, bw, bh) in enumerate(boxes):
            x0, y0 = max(0, bx - 2), max(0, by - 2)
            x1, y1 = min(w, bx + bw + 2), min(h, by + bh + 2)
            crop = gray[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            text_pred = ""
            if self.text_sess is not None:
                ch_h, ch_w = crop.shape[:2]
                scale = 28.0 / max(ch_h, 1)
                new_w = max(8, min(188, int(round(ch_w * scale))))
                resized = cv2.resize(crop, (new_w, 28), interpolation=cv2.INTER_CUBIC)
                canvas = np.full((32, 192), 255, dtype=np.uint8)
                # Normalize crop background to white
                p85 = float(np.percentile(resized, 85))
                if p85 > 40:
                    resized = np.clip((resized.astype(np.float32) / p85) * 255.0, 0, 255).astype(np.uint8)
                canvas[2:30, 2 : 2 + new_w] = resized
                inp = (canvas.astype(np.float32) / 255.0)[None, None, :, :]
                log_probs = self.text_sess.run(None, {"crop": inp})[0]  # (1, T, V)
                seq = np.argmax(log_probs[0], axis=-1).tolist()
                chars = []
                prev = 0
                for c_idx in seq:
                    if c_idx != 0 and c_idx != prev and c_idx < len(REKHA_CHARSET):
                        chars.append(REKHA_CHARSET[c_idx])
                    prev = c_idx
                text_pred = "".join(chars).strip()
            if not text_pred:
                continue
            has_dev = any(0x0900 <= ord(c) <= 0x097F for c in text_pred)
            cid = f"callout-{idx + 1}"
            labels_out.append({
                "id": cid,
                "callout_id": cid,
                "text": text_pred,
                "lang": "ne" if has_dev else "en",
                "bbox": [bx, by, bw, bh],
                "x": bx + 2,
                "y": by + bh - 3,
                "font_size": 14.0,
            })
        return labels_out

    def _detect_classical_word_boxes(self, work_ink: np.ndarray, img_bgr: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Groups character-scale connected components into horizontal words/lines and filters out
        colored water-hatch dashes so printed text is separated into ch_text and never traced as strokes.
        """
        h, w = work_ink.shape[:2]
        ch_text = np.zeros((h, w), dtype=np.uint8)
        num_c, c_lbl, c_stats, _ = cv2.connectedComponentsWithStats(work_ink, connectivity=8)
        char_boxes = []
        for i in range(1, num_c):
            bx = int(c_stats[i, cv2.CC_STAT_LEFT])
            by = int(c_stats[i, cv2.CC_STAT_TOP])
            cw = int(c_stats[i, cv2.CC_STAT_WIDTH])
            ch = int(c_stats[i, cv2.CC_STAT_HEIGHT])
            area = int(c_stats[i, cv2.CC_STAT_AREA])
            if 3 <= cw <= 32 and 5 <= ch <= 32 and 8 <= area <= 420:
                # Exclude flat horizontal water-hatch dashes (aspect > 2.6 and height <= 5)
                if cw > 2.6 * ch and ch <= 5:
                    continue
                fill_ratio = area / float(max(cw * ch, 1))
                if 0.14 <= fill_ratio <= 0.85:
                    char_boxes.append((i, bx, by, cw, ch))

        # Group horizontally adjacent glyph boxes into word lines
        char_list = list(char_boxes)
        used = [False] * len(char_list)
        word_boxes: List[List[int]] = []
        for idx_a in range(len(char_list)):
            if used[idx_a]:
                continue
            group = [char_list[idx_a]]
            used[idx_a] = True
            changed = True
            while changed:
                changed = False
                for idx_b in range(len(char_list)):
                    if used[idx_b]:
                        continue
                    _, bx_b, by_b, cw_b, ch_b = char_list[idx_b]
                    cy_b = by_b + 0.5 * ch_b
                    for _, bx_g, by_g, cw_g, ch_g in group:
                        cy_g = by_g + 0.5 * ch_g
                        max_h = max(ch_b, ch_g)
                        if abs(cy_b - cy_g) <= 0.65 * max_h:
                            gap_x = max(bx_b - (bx_g + cw_g), bx_g - (bx_b + cw_b), 0)
                            if gap_x <= 1.65 * max_h:
                                used[idx_b] = True
                                group.append(char_list[idx_b])
                                changed = True
                                break
            # A word has >= 2 characters, or is a standalone '+' / '-' symbol (nearly square 6..18px)
            is_plus_minus = (
                len(group) == 1
                and 6 <= group[0][3] <= 18
                and (0.75 <= group[0][3] / float(max(group[0][4], 1)) <= 1.35)
            )
            if len(group) >= 2 or is_plus_minus:
                x0 = max(0, min(g[1] for g in group) - 2)
                y0 = max(0, min(g[2] for g in group) - 2)
                x1 = min(w, max(g[1] + g[3] for g in group) + 2)
                y1 = min(h, max(g[2] + g[4] for g in group) + 2)
                ch_text[y0:y1, x0:x1] = 255
                word_boxes.append([x0, y0, x1 - x0, y1 - y0])

        return ch_text, word_boxes

    def decompose_neural_6ch(
        self,
        img_bgr: np.ndarray,
        known_labels: Optional[List[Dict]] = None,
    ) -> Tuple[Dict[str, np.ndarray], List[Dict]]:
        """Runs Rekha-Net INT8 ONNX for 6-channel semantic decomposition + Rekha-Text OCR."""
        h, w = img_bgr.shape[:2]
        norm_gray, classical_ink = self.normalize_illumination(img_bgr)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if (h, w) != (512, 512):
            rgb_512 = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_AREA)
        else:
            rgb_512 = rgb
        inp = (rgb_512.transpose(2, 0, 1).astype(np.float32) / 255.0)[None, :, :, :]
        logits = self.net_sess.run(None, {"image": inp})[0][0]  # (7, 512, 512)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        if (h, w) != (512, 512):
            probs = np.stack([cv2.resize(probs[c], (w, h), interpolation=cv2.INTER_LINEAR) for c in range(7)], axis=0)

        ch_strokes = (probs[1] >= 0.42).astype(np.uint8) * 255
        ch_dashed = (probs[2] >= 0.42).astype(np.uint8) * 255
        ch_arrows = (probs[3] >= 0.40).astype(np.uint8) * 255
        ch_fills = (probs[4] >= 0.45).astype(np.uint8) * 255
        ch_text = (probs[5] >= 0.40).astype(np.uint8) * 255
        ch_junctions = (probs[6] >= 0.40).astype(np.uint8) * 255

        # Enforce strict text exclusion mask on strokes, dashed lines, and arrows
        if known_labels:
            for lb in known_labels:
                bx, by, bw, bh = [int(round(v)) for v in lb.get("bbox", [0, 0, 0, 0])]
                x0, y0 = max(0, bx - 3), max(0, by - 3)
                x1, y1 = min(w, bx + bw + 3), min(h, by + bh + 3)
                ch_text[y0:y1, x0:x1] = 255

        # Hybridize on out-of-domain diagrams (e.g. AI2D cartoons) if neural masks are sparse
        if np.count_nonzero(ch_text) < 25 and not known_labels:
            ch_text_cls, _ = self._detect_classical_word_boxes(classical_ink.copy(), img_bgr)
            ch_text = cv2.bitwise_or(ch_text, ch_text_cls)
        if np.count_nonzero(ch_strokes) < 60:
            work_ink = classical_ink.copy()
            work_ink[ch_text > 0] = 0
            ch_strokes = cv2.bitwise_or(ch_strokes, work_ink)

        text_excl = cv2.dilate(ch_text, np.ones((5, 5), np.uint8), iterations=1)
        ch_strokes[text_excl > 0] = 0
        ch_dashed[text_excl > 0] = 0
        ch_arrows[text_excl > 0] = 0

        labels_out = list(known_labels) if known_labels else []
        if not labels_out and np.count_nonzero(ch_text) > 0:
            boxes = self._split_multiline_text_boxes(ch_text, classical_ink)
            labels_out = self.recognize_text_boxes(img_bgr, boxes)

        return {
            "strokes": ch_strokes,
            "dashed": ch_dashed,
            "arrows": ch_arrows,
            "fills": ch_fills,
            "text": ch_text,
            "junctions": ch_junctions,
            "norm_gray": norm_gray,
            "ink": classical_ink,
        }, labels_out

    def decompose_classical_6ch(
        self,
        img_bgr: np.ndarray,
        known_labels: Optional[List[Dict]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Hardened Classical-Hybrid 6-channel semantic decomposition on textbook diagrams:
          - Detects colored/shaded liquid & organ fills (including blue dashed water textures)
          - Groups character blobs into word/line boxes and excludes them from stroke/dash tracing
        """
        h, w = img_bgr.shape[:2]
        norm_gray, ink = self.normalize_illumination(img_bgr)
        ch_strokes = np.zeros((h, w), dtype=np.uint8)
        ch_dashed = np.zeros((h, w), dtype=np.uint8)
        ch_arrows = np.zeros((h, w), dtype=np.uint8)
        ch_fills = np.zeros((h, w), dtype=np.uint8)
        ch_text = np.zeros((h, w), dtype=np.uint8)
        ch_junctions = np.zeros((h, w), dtype=np.uint8)

        # 1. Detect colored & mid-tone shaded fills + blue/cyan water-hatch strokes
        work_ink = ink.copy()
        if img_bgr.ndim == 3:
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            b_ch, g_ch, r_ch = cv2.split(img_bgr)
            # Colored blue/cyan ink strokes (e.g., water dashes inside a beaker)
            blue_ink = ((b_ch.astype(np.int16) - r_ch.astype(np.int16) >= 35) & (sat >= 45) & (ink > 0))
            if np.count_nonzero(blue_ink) > 40:
                # Remove blue water-hatch dashes from structural ink and merge into fill mask
                work_ink[blue_ink] = 0
                water_closed = cv2.morphologyEx(
                    blue_ink.astype(np.uint8) * 255,
                    cv2.MORPH_CLOSE,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 19)),
                )
                ch_fills |= water_closed
            color_fill = ((sat >= 28) & (val >= 115) & (work_ink == 0)).astype(np.uint8) * 255
        else:
            color_fill = np.zeros((h, w), dtype=np.uint8)

        mid_fill = ((norm_gray >= 135) & (norm_gray <= 232) & (work_ink == 0)).astype(np.uint8) * 255
        fill_cand = cv2.bitwise_or(mid_fill, color_fill)
        fill_cand = cv2.morphologyEx(fill_cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        num_f, f_lbl, f_stats, _ = cv2.connectedComponentsWithStats(fill_cand, connectivity=8)
        for i in range(1, num_f):
            area = f_stats[i, cv2.CC_STAT_AREA]
            if 180 <= area <= int(0.45 * h * w):
                ch_fills[f_lbl == i] = 255

        # 2. Separate text boxes (either provided or word-grouped character components)
        if known_labels:
            for lb in known_labels:
                bx, by, bw, bh = [int(round(v)) for v in lb.get("bbox", [0, 0, 0, 0])]
                pad = 3
                x0, y0 = max(0, bx - pad), max(0, by - pad)
                x1, y1 = min(w, bx + bw + pad), min(h, by + bh + pad)
                ch_text[y0:y1, x0:x1] = 255
                work_ink[y0:y1, x0:x1] = 0
        else:
            ch_text_auto, _ = self._detect_classical_word_boxes(work_ink, img_bgr)
            ch_text |= ch_text_auto
            work_ink[ch_text > 0] = 0

        # 3. Separate dashed segments, arrowheads, and continuous strokes
        num_c, c_lbl, c_stats, _ = cv2.connectedComponentsWithStats(work_ink, connectivity=8)
        for i in range(1, num_c):
            cw = c_stats[i, cv2.CC_STAT_WIDTH]
            ch = c_stats[i, cv2.CC_STAT_HEIGHT]
            area = c_stats[i, cv2.CC_STAT_AREA]
            diag = math.hypot(cw, ch)
            if area < 6:
                continue
            comp_mask = (c_lbl == i).astype(np.uint8) * 255
            if 6 <= diag <= 22 and area <= 75:
                ch_dashed[c_lbl == i] = 255
            else:
                ch_strokes[c_lbl == i] = 255
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
        if np.count_nonzero(binary_mask) == 0:
            return []
        skel = skeletonize(binary_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(skel, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        chains: List[List[Point]] = []
        for cnt in contours:
            if len(cnt) < min_len:
                continue
            pts_raw = [(float(p[0][0]), float(p[0][1])) for p in cnt]
            half_n = len(pts_raw) // 2
            if half_n >= 4 and math.hypot(pts_raw[0][0] - pts_raw[half_n][0], pts_raw[0][1] - pts_raw[half_n][1]) > 6.0:
                pts = pts_raw[: half_n + 1]
            else:
                pts = pts_raw
            # Suppress tiny character-scale residual blobs (< 12px bbox & < 16px span)
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bw, bh = max(xs) - min(xs), max(ys) - min(ys)
            if bw < 10.0 and bh < 10.0:
                continue
            chains.append(pts)
        return chains

    def _group_dashed_chains(self, dashed_mask: np.ndarray) -> List[Dict]:
        """
        Groups strictly colinear short dash blobs into unified dashed line primitives,
        while rejecting 2D periodic water-hatch/stipple textures so they never chain into zig-zags.
        """
        num_d, d_lbl, stats, centroids = cv2.connectedComponentsWithStats(dashed_mask, connectivity=8)
        if num_d <= 1:
            return []
        centers: List[Point] = []
        orientations: List[Point] = []
        for i in range(1, num_d):
            ys, xs = np.nonzero(d_lbl == i)
            if len(xs) < 3:
                continue
            cx, cy = float(centroids[i][0]), float(centroids[i][1])
            dx = xs.astype(np.float64) - cx
            dy = ys.astype(np.float64) - cy
            cov = np.array([[np.mean(dx * dx), np.mean(dx * dy)], [np.mean(dx * dy), np.mean(dy * dy)]])
            eigvals, eigvecs = np.linalg.eigh(cov)
            vec = eigvecs[:, 1]
            centers.append((cx, cy))
            orientations.append((float(vec[0]), float(vec[1])))

        n_c = len(centers)
        if n_c < 2:
            return []

        # Filter out 2D periodic hatch/stipple clusters (where >= 3 neighbors within 32px lie off-axis)
        is_hatch_2d = [False] * n_c
        for i in range(n_c):
            cx_i, cy_i = centers[i]
            ux_i, uy_i = orientations[i]
            off_axis_neighbors = 0
            for j in range(n_c):
                if i == j:
                    continue
                vx = centers[j][0] - cx_i
                vy = centers[j][1] - cy_i
                dist = math.hypot(vx, vy)
                if dist <= 32.0:
                    cross_dist = abs(vx * (-uy_i) + vy * ux_i)
                    if cross_dist >= 6.5:
                        off_axis_neighbors += 1
            if off_axis_neighbors >= 3:
                is_hatch_2d[i] = True

        used = list(is_hatch_2d)
        dashed_prims: List[Dict] = []

        for i in range(n_c):
            if used[i]:
                continue
            used[i] = True
            chain = [centers[i]]
            while True:
                last = chain[-1]
                best_j = None
                best_d = self.cfg.dash_gap_max
                for j in range(n_c):
                    if used[j]:
                        continue
                    vx = centers[j][0] - last[0]
                    vy = centers[j][1] - last[1]
                    d = math.hypot(vx, vy)
                    if 2.0 < d < best_d:
                        if len(chain) >= 2:
                            prev = chain[-2]
                            v1 = (last[0] - prev[0], last[1] - prev[1])
                            n1 = math.hypot(*v1)
                            if n1 > 1e-3:
                                cos_a = (v1[0] * vx + v1[1] * vy) / (n1 * d)
                                if cos_a < 0.92:
                                    continue
                            v_chord = (centers[j][0] - chain[0][0], centers[j][1] - chain[0][1])
                            n_chord = math.hypot(*v_chord)
                            if n_chord > 1e-3:
                                cos_c = (v_chord[0] * vx + v_chord[1] * vy) / (n_chord * d)
                                if cos_c < 0.90:
                                    continue
                        best_d = d
                        best_j = j
                if best_j is not None:
                    used[best_j] = True
                    chain.append(centers[best_j])
                else:
                    break
            total_span = math.hypot(chain[-1][0] - chain[0][0], chain[-1][1] - chain[0][1])
            if len(chain) >= 3 or (len(chain) == 2 and total_span >= 18.0):
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
        labels_in = list(detected_labels) if detected_labels else []

        if semantic_channels is not None:
            ch = dict(semantic_channels)
            if "text" in ch and np.count_nonzero(ch["text"]) > 0:
                text_excl = cv2.dilate(ch["text"], np.ones((5, 5), np.uint8), iterations=1)
                ch["strokes"] = ch["strokes"].copy()
                ch["dashed"] = ch["dashed"].copy()
                ch["strokes"][text_excl > 0] = 0
                ch["dashed"][text_excl > 0] = 0
                if not labels_in and self.text_sess is not None:
                    _, classical_ink = self.normalize_illumination(img_bgr)
                    boxes = self._split_multiline_text_boxes(ch["text"], classical_ink)
                    labels_in = self.recognize_text_boxes(img_bgr, boxes)
        elif self.net_sess is not None:
            ch, labels_in = self.decompose_neural_6ch(img_bgr, known_labels=labels_in)
        else:
            ch = self.decompose_classical_6ch(img_bgr, known_labels=labels_in)
            if not labels_in and self.text_sess is not None and np.count_nonzero(ch["text"]) > 0:
                boxes = self._split_multiline_text_boxes(ch["text"], ch["ink"])
                labels_in = self.recognize_text_boxes(img_bgr, boxes)

        j_pts: List[Point] = []
        if "junctions" in ch and np.count_nonzero(ch["junctions"]) > 0:
            num_j, _, _, j_cent = cv2.connectedComponentsWithStats(ch["junctions"], connectivity=8)
            j_pts = [(float(j_cent[i][0]), float(j_cent[i][1])) for i in range(1, num_j)]

        # 1. Compile Fills (with optional sampled tint color from textbook illustration)
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
