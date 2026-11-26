"""
Deterministic Geometric Primitive & Schneider Cubic Bezier Fitting Engine.
Mirrored 1:1 in docs/engine/primitives.js for exact Python <-> Browser JS parity.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple
import numpy as np

Point = Tuple[float, float]


def point_line_distance(pt: Point, a: Point, b: Point) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return math.hypot(pt[0] - a[0], pt[1] - a[1])
    t = max(0.0, min(1.0, ((pt[0] - a[0]) * dx + (pt[1] - a[1]) * dy) / len_sq))
    proj_x = a[0] + t * dx
    proj_y = a[1] + t * dy
    return math.hypot(pt[0] - proj_x, pt[1] - proj_y)


def douglas_peucker(points: List[Point], epsilon: float = 2.0) -> List[Point]:
    """Iterative/recursive Ramer-Douglas-Peucker simplification."""
    if len(points) <= 2:
        return list(points)
    max_dist = -1.0
    index = 0
    start, end = points[0], points[-1]
    for i in range(1, len(points) - 1):
        d = point_line_distance(points[i], start, end)
        if d > max_dist:
            index = i
            max_dist = d
    if max_dist > epsilon:
        left = douglas_peucker(points[: index + 1], epsilon)
        right = douglas_peucker(points[index:], epsilon)
        return left[:-1] + right
    return [start, end]


def snap_orthogonal_polyline(
    points: List[Point],
    angle_tol_deg: float = 6.5,
    junctions: Optional[List[Point]] = None,
    snap_radius: float = 6.0,
) -> List[Point]:
    """
    Snaps near-horizontal (0/180 deg), near-vertical (90/270 deg), and near-45-deg
    segments cleanly, and anchors vertices to detected corner/junction keypoints.
    """
    if len(points) < 2:
        return list(points)
    snapped: List[Point] = [ (round(points[0][0], 2), round(points[0][1], 2)) ]
    for i in range(1, len(points)):
        prev_x, prev_y = snapped[-1]
        cur_x, cur_y = float(points[i][0]), float(points[i][1])
        if junctions:
            best_j = None
            best_d = snap_radius
            for jx, jy in junctions:
                d = math.hypot(cur_x - jx, cur_y - jy)
                if d < best_d:
                    best_d = d
                    best_j = (jx, jy)
            if best_j is not None:
                cur_x, cur_y = float(best_j[0]), float(best_j[1])
        dx = cur_x - prev_x
        dy = cur_y - prev_y
        seg_len = math.hypot(dx, dy)
        if seg_len > 3.0:
            ang = math.degrees(math.atan2(dy, dx)) % 360.0
            if min(abs(ang - 0.0), abs(ang - 180.0), abs(ang - 360.0)) <= angle_tol_deg:
                cur_y = prev_y
            elif min(abs(ang - 90.0), abs(ang - 270.0)) <= angle_tol_deg:
                cur_x = prev_x
            else:
                for diag in (45.0, 135.0, 225.0, 315.0):
                    if abs(ang - diag) <= angle_tol_deg * 0.75:
                        rad = math.radians(diag)
                        cur_x = prev_x + seg_len * math.cos(rad)
                        cur_y = prev_y + seg_len * math.sin(rad)
                        break
        snapped.append((round(cur_x, 2), round(cur_y, 2)))
    return snapped


def fit_taubin_circle(points: List[Point], max_rel_rmse: float = 0.065) -> Optional[Dict]:
    """
    Algebraic Taubin/Kasa circle fit. Returns {type: 'circle', cx, cy, r, rmse}
    if closed/nearly-closed contour has low radial RMSE and sufficient angular span.
    """
    n = len(points)
    if n < 8:
        return None
    pts = np.asarray(points, dtype=np.float64)
    start_end_dist = float(np.hypot(pts[0, 0] - pts[-1, 0], pts[0, 1] - pts[-1, 1]))
    mx = float(np.mean(pts[:, 0]))
    my = float(np.mean(pts[:, 1]))
    u = pts[:, 0] - mx
    v = pts[:, 1] - my
    suu = float(np.sum(u * u))
    suv = float(np.sum(u * v))
    svv = float(np.sum(v * v))
    suuu = float(np.sum(u * u * u))
    svvv = float(np.sum(v * v * v))
    suvv = float(np.sum(u * v * v))
    svuu = float(np.sum(v * u * u))
    det = suu * svv - suv * suv
    if abs(det) < 1e-9:
        return None
    rhs1 = 0.5 * (suuu + suvv)
    rhs2 = 0.5 * (svvv + svuu)
    uc = (svv * rhs1 - suv * rhs2) / det
    vc = (suu * rhs2 - suv * rhs1) / det
    cx = mx + uc
    cy = my + vc
    radii = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    r = float(np.mean(radii))
    if r < 5.0 or r > 2000.0:
        return None
    if start_end_dist > max(14.0, 0.45 * r):
        return None
    rmse = float(np.sqrt(np.mean((radii - r) ** 2)))
    rel_rmse = rmse / max(r, 1.0)
    if rel_rmse > max_rel_rmse:
        return None
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    unwrapped = np.unwrap(angles)
    span = float(abs(unwrapped[-1] - unwrapped[0]))
    if span < 1.35 * math.pi and float(np.ptp(angles)) < 1.55 * math.pi:
        return None
    return {
        "type": "circle",
        "cx": round(cx, 2),
        "cy": round(cy, 2),
        "r": round(r, 2),
        "rmse": round(rel_rmse, 4),
    }


def fit_ellipse_algebraic(points: List[Point], max_rel_rmse: float = 0.075) -> Optional[Dict]:
    """
    Principal-axis covariance ellipse fit for closed ovals (e.g., mitochondria, nuclei,
    Venn ovals, angled lenses).
    """
    n = len(points)
    if n < 12:
        return None
    pts = np.asarray(points, dtype=np.float64)
    cx = float(np.mean(pts[:, 0]))
    cy = float(np.mean(pts[:, 1]))
    centered = pts - np.array([cx, cy])
    cov = (centered.T @ centered) / float(n)
    eigvals, eigvecs = np.linalg.eigh(cov)
    if eigvals[0] <= 1e-3 or eigvals[1] <= 1e-3:
        return None
    # For a uniform ellipse boundary, mean(x^2) = a^2 / 2
    rx = float(math.sqrt(2.0 * max(eigvals[1], 1e-3)))
    ry = float(math.sqrt(2.0 * max(eigvals[0], 1e-3)))
    if ry < 5.0 or rx / max(ry, 1e-6) > 6.0:
        return None
    start_end_dist = float(np.hypot(pts[0, 0] - pts[-1, 0], pts[0, 1] - pts[-1, 1]))
    if start_end_dist > max(16.0, 0.45 * min(rx, ry)):
        return None
    major_vec = eigvecs[:, 1]
    angle_rad = float(math.atan2(major_vec[1], major_vec[0]))
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    xr = centered[:, 0] * cos_a - centered[:, 1] * sin_a
    yr = centered[:, 0] * sin_a + centered[:, 1] * cos_a
    norm_r = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
    rel_rmse = float(np.sqrt(np.mean((norm_r - 1.0) ** 2)))
    if rel_rmse > max_rel_rmse:
        return None
    return {
        "type": "ellipse",
        "cx": round(cx, 2),
        "cy": round(cy, 2),
        "rx": round(rx, 2),
        "ry": round(ry, 2),
        "angle": round(math.degrees(angle_rad), 2),
        "rmse": round(rel_rmse, 4),
    }


def fit_schneider_cubic_bezier(
    points: List[Point],
    simplify_eps: float = 2.2,
    tension: float = 0.32,
) -> List[Tuple[Point, Point, Point, Point]]:
    """
    Minimal-node C1 smooth Cubic Bezier spline fitter inspired by Philip Schneider's
    Graphics Gems algorithm with chord-length parameterization and tangent estimation.
    Returns list of cubic segments [(P0, C1, C2, P3), ...].
    """
    pts = douglas_peucker(points, epsilon=simplify_eps)
    n = len(pts)
    if n < 2:
        return []
    if n == 2:
        p0, p3 = pts[0], pts[1]
        c1 = (round(p0[0] + (p3[0] - p0[0]) / 3.0, 2), round(p0[1] + (p3[1] - p0[1]) / 3.0, 2))
        c2 = (round(p0[0] + 2.0 * (p3[0] - p0[0]) / 3.0, 2), round(p0[1] + 2.0 * (p3[1] - p0[1]) / 3.0, 2))
        return [(p0, c1, c2, p3)]

    closed = math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 6.0 and n >= 4
    tangents: List[Point] = []
    for i in range(n):
        if i == 0:
            prev_pt = pts[-2] if closed else pts[0]
            next_pt = pts[1]
        elif i == n - 1:
            prev_pt = pts[n - 2]
            next_pt = pts[1] if closed else pts[n - 1]
        else:
            prev_pt = pts[i - 1]
            next_pt = pts[i + 1]
        tx = next_pt[0] - prev_pt[0]
        ty = next_pt[1] - prev_pt[1]
        norm = math.hypot(tx, ty)
        if norm < 1e-8:
            tangents.append((0.0, 0.0))
        else:
            tangents.append((tx / norm, ty / norm))

    segments: List[Tuple[Point, Point, Point, Point]] = []
    for i in range(n - 1):
        p0 = pts[i]
        p3 = pts[i + 1]
        chord = math.hypot(p3[0] - p0[0], p3[1] - p0[1])
        alpha = chord * tension
        t0 = tangents[i]
        t1 = tangents[i + 1]
        c1 = (round(p0[0] + alpha * t0[0], 2), round(p0[1] + alpha * t0[1], 2))
        c2 = (round(p3[0] - alpha * t1[0], 2), round(p3[1] - alpha * t1[1], 2))
        segments.append(((round(p0[0], 2), round(p0[1], 2)), c1, c2, (round(p3[0], 2), round(p3[1], 2))))
    return segments


def cubic_segments_to_svg_path(segments: List[Tuple[Point, Point, Point, Point]], closed: bool = False) -> str:
    if not segments:
        return ""
    p0 = segments[0][0]
    parts = [f"M {p0[0]:.2f} {p0[1]:.2f}"]
    for _, c1, c2, p3 in segments:
        parts.append(f"C {c1[0]:.2f} {c1[1]:.2f}, {c2[0]:.2f} {c2[1]:.2f}, {p3[0]:.2f} {p3[1]:.2f}")
    if closed:
        parts.append("Z")
    return " ".join(parts)


def classify_contour_primitive(
    points: List[Point],
    junctions: Optional[List[Point]] = None,
    prefer_organic: bool = False,
) -> Dict:
    """
    Classifies a contour or skeleton chain into the most compact mathematical SVG primitive:
    - 'line' (2-point snapped segment)
    - 'circle' (Taubin algebraic circle)
    - 'ellipse' (principal-axis ellipse)
    - 'polyline' / 'polygon' (snapped orthogonal/technical segments)
    - 'bezier' (Schneider C1 cubic Bezier path for smooth curves/biology shapes)
    """
    if len(points) < 2:
        return {"type": "empty"}

    circle = fit_taubin_circle(points)
    if circle is not None:
        return circle

    ellipse = fit_ellipse_algebraic(points)
    if ellipse is not None:
        return ellipse

    dp_pts = douglas_peucker(points, epsilon=2.2)
    if len(dp_pts) == 2:
        snapped = snap_orthogonal_polyline(dp_pts, junctions=junctions)
        return {
            "type": "line",
            "x1": snapped[0][0],
            "y1": snapped[0][1],
            "x2": snapped[1][0],
            "y2": snapped[1][1],
        }

    # Measure corner sharpness to decide technical polyline vs smooth organic Bezier
    sharp_corners = 0
    for i in range(1, len(dp_pts) - 1):
        v1x = dp_pts[i][0] - dp_pts[i - 1][0]
        v1y = dp_pts[i][1] - dp_pts[i - 1][1]
        v2x = dp_pts[i + 1][0] - dp_pts[i][0]
        v2y = dp_pts[i + 1][1] - dp_pts[i][1]
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 > 2.0 and n2 > 2.0:
            cos_t = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)))
            turn_deg = math.degrees(math.acos(cos_t))
            if turn_deg > 28.0:
                sharp_corners += 1

    closed = math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) < 7.0
    if not prefer_organic and ( len(dp_pts) <= 6 or sharp_corners >= max(1, len(dp_pts) - 3) ):
        snapped = snap_orthogonal_polyline(dp_pts, junctions=junctions)
        return {
            "type": "polygon" if closed and len(snapped) >= 3 else "polyline",
            "points": snapped,
        }

    segs = fit_schneider_cubic_bezier(points, simplify_eps=2.4)
    return {
        "type": "bezier",
        "d": cubic_segments_to_svg_path(segs, closed=closed),
        "segments": segs,
        "closed": closed,
        "node_count": len(segs) + 1,
    }


def associate_callouts_to_labels(
    labels: List[Dict],
    arrows: List[Dict],
    max_dist: float = 95.0,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Links each text label box to its nearest pointer arrow / leader line via
    endpoint graph proximity and assigns matching `callout_id` attributes.
    """
    out_labels = [dict(lb) for lb in labels]
    out_arrows = [dict(ar) for ar in arrows]
    for idx, lb in enumerate(out_labels):
        cid = lb.get("id") or f"callout-{idx + 1}"
        lb["callout_id"] = cid
        bx, by, bw, bh = lb.get("bbox", [lb.get("x", 0), lb.get("y", 0), 40, 16])
        cx = bx + bw * 0.5
        cy = by + bh * 0.5
        best_ar_idx = None
        best_dist = max_dist
        for a_i, ar in enumerate(out_arrows):
            tail = (ar.get("x1", 0.0), ar.get("y1", 0.0))
            head = (ar.get("x2", 0.0), ar.get("y2", 0.0))
            # Distance from either arrow endpoint to the label bounding box
            for pt in (tail, head):
                dx = max(bx - pt[0], 0.0, pt[0] - (bx + bw))
                dy = max(by - pt[1], 0.0, pt[1] - (by + bh))
                d = math.hypot(dx, dy)
                if d < best_dist:
                    best_dist = d
                    best_ar_idx = a_i
        if best_ar_idx is not None:
            out_arrows[best_ar_idx]["callout_id"] = cid
            lb["target_x"] = out_arrows[best_ar_idx].get("x2", cx)
            lb["target_y"] = out_arrows[best_ar_idx].get("y2", cy)
    return out_labels, out_arrows
