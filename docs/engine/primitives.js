/**
 * RekhaSVG — 1:1 Browser JavaScript Mirror of rekhasvg/primitives.py
 * Zero-dependency Deterministic Geometric Primitive & Schneider Cubic Bezier Engine.
 */

export function round2(v) {
  return Math.round(v * 100) / 100;
}

export function pointLineDistance(pt, a, b) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const lenSq = dx * dx + dy * dy;
  if (lenSq < 1e-12) {
    return Math.hypot(pt[0] - a[0], pt[1] - a[1]);
  }
  const t = Math.max(0.0, Math.min(1.0, ((pt[0] - a[0]) * dx + (pt[1] - a[1]) * dy) / lenSq));
  const projX = a[0] + t * dx;
  const projY = a[1] + t * dy;
  return Math.hypot(pt[0] - projX, pt[1] - projY);
}

export function douglasPeucker(points, epsilon = 2.0) {
  if (!points || points.length <= 2) {
    return points ? points.slice() : [];
  }
  let maxDist = -1.0;
  let index = 0;
  const start = points[0];
  const end = points[points.length - 1];
  for (let i = 1; i < points.length - 1; i++) {
    const d = pointLineDistance(points[i], start, end);
    if (d > maxDist) {
      index = i;
      maxDist = d;
    }
  }
  if (maxDist > epsilon) {
    const left = douglasPeucker(points.slice(0, index + 1), epsilon);
    const right = douglasPeucker(points.slice(index), epsilon);
    return left.slice(0, -1).concat(right);
  }
  return [start, end];
}

export function snapOrthogonalPolyline(points, angleTolDeg = 6.5, junctions = null, snapRadius = 6.0) {
  if (!points || points.length < 2) {
    return points ? points.slice() : [];
  }
  const snapped = [[round2(points[0][0]), round2(points[0][1])]];
  for (let i = 1; i < points.length; i++) {
    const [prevX, prevY] = snapped[snapped.length - 1];
    let curX = Number(points[i][0]);
    let curY = Number(points[i][1]);

    if (junctions && junctions.length > 0) {
      let bestJ = null;
      let bestD = snapRadius;
      for (const [jx, jy] of junctions) {
        const d = Math.hypot(curX - jx, curY - jy);
        if (d < bestD) {
          bestD = d;
          bestJ = [jx, jy];
        }
      }
      if (bestJ !== null) {
        curX = Number(bestJ[0]);
        curY = Number(bestJ[1]);
      }
    }

    const dx = curX - prevX;
    const dy = curY - prevY;
    const segLen = Math.hypot(dx, dy);
    if (segLen > 3.0) {
      const ang = ((Math.atan2(dy, dx) * 180.0) / Math.PI + 360.0) % 360.0;
      const horizDiff = Math.min(Math.abs(ang - 0.0), Math.abs(ang - 180.0), Math.abs(ang - 360.0));
      const vertDiff = Math.min(Math.abs(ang - 90.0), Math.abs(ang - 270.0));
      if (horizDiff <= angleTolDeg) {
        curY = prevY;
      } else if (vertDiff <= angleTolDeg) {
        curX = prevX;
      } else {
        for (const diag of [45.0, 135.0, 225.0, 315.0]) {
          if (Math.abs(ang - diag) <= angleTolDeg * 0.75) {
            const rad = (diag * Math.PI) / 180.0;
            curX = prevX + segLen * Math.cos(rad);
            curY = prevY + segLen * Math.sin(rad);
            break;
          }
        }
      }
    }
    snapped.push([round2(curX), round2(curY)]);
  }
  return snapped;
}

export function fitTaubinCircle(points, maxRelRmse = 0.065) {
  const n = points ? points.length : 0;
  if (n < 8) return null;
  const startEndDist = Math.hypot(
    points[0][0] - points[n - 1][0],
    points[0][1] - points[n - 1][1]
  );

  let mx = 0.0;
  let my = 0.0;
  for (let i = 0; i < n; i++) {
    mx += points[i][0];
    my += points[i][1];
  }
  mx /= n;
  my /= n;

  let suu = 0, suv = 0, svv = 0, suuu = 0, svvv = 0, suvv = 0, svuu = 0;
  for (let i = 0; i < n; i++) {
    const u = points[i][0] - mx;
    const v = points[i][1] - my;
    const uu = u * u;
    const vv = v * v;
    suu += uu;
    suv += u * v;
    svv += vv;
    suuu += uu * u;
    svvv += vv * v;
    suvv += u * vv;
    svuu += v * uu;
  }

  const det = suu * svv - suv * suv;
  if (Math.abs(det) < 1e-9) return null;
  const rhs1 = 0.5 * (suuu + suvv);
  const rhs2 = 0.5 * (svvv + svuu);
  const uc = (svv * rhs1 - suv * rhs2) / det;
  const vc = (suu * rhs2 - suv * rhs1) / det;
  const cx = mx + uc;
  const cy = my + vc;

  const radii = new Float64Array(n);
  let rSum = 0.0;
  for (let i = 0; i < n; i++) {
    const rad = Math.hypot(points[i][0] - cx, points[i][1] - cy);
    radii[i] = rad;
    rSum += rad;
  }
  const r = rSum / n;
  if (r < 8.0 || r > 240.0) return null;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  let arcLen = 0.0;
  for (let i = 0; i < n; i++) {
    const px = points[i][0];
    const py = points[i][1];
    if (px < minX) minX = px;
    if (px > maxX) maxX = px;
    if (py < minY) minY = py;
    if (py > maxY) maxY = py;
    if (i > 0) {
      arcLen += Math.hypot(px - points[i - 1][0], py - points[i - 1][1]);
    }
  }
  const bboxW = maxX - minX;
  const bboxH = maxY - minY;
  if (bboxW < 1.65 * r || bboxW > 2.35 * r || bboxH < 1.65 * r || bboxH > 2.35 * r) return null;
  if (startEndDist > Math.max(10.0, Math.min(18.0, 0.35 * r))) return null;
  if (arcLen < 1.55 * Math.PI * r) return null;

  let sqErr = 0.0;
  let q0 = false, q1 = false, q2 = false, q3 = false;
  for (let i = 0; i < n; i++) {
    const diff = radii[i] - r;
    sqErr += diff * diff;
    const a = Math.atan2(points[i][1] - cy, points[i][0] - cx);
    if (a >= 0.0 && a < 0.5 * Math.PI) q0 = true;
    else if (a >= 0.5 * Math.PI && a <= Math.PI) q1 = true;
    else if (a >= -Math.PI && a < -0.5 * Math.PI) q2 = true;
    else if (a >= -0.5 * Math.PI && a < 0.0) q3 = true;
  }
  const rmse = Math.sqrt(sqErr / n);
  const relRmse = rmse / Math.max(r, 1.0);
  if (relRmse > maxRelRmse) return null;
  if (!(q0 && q1 && q2 && q3)) return null;

  return {
    type: "circle",
    cx: round2(cx),
    cy: round2(cy),
    r: round2(r),
    rmse: Math.round(relRmse * 10000) / 10000,
  };
}

export function fitEllipseAlgebraic(points, maxRelRmse = 0.075) {
  const n = points ? points.length : 0;
  if (n < 16) return null;
  let cx = 0, cy = 0;
  let arcLen = 0.0;
  for (let i = 0; i < n; i++) {
    cx += points[i][0];
    cy += points[i][1];
    if (i > 0) {
      arcLen += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
    }
  }
  cx /= n;
  cy /= n;

  let c00 = 0, c01 = 0, c11 = 0;
  for (let i = 0; i < n; i++) {
    const dx = points[i][0] - cx;
    const dy = points[i][1] - cy;
    c00 += dx * dx;
    c01 += dx * dy;
    c11 += dy * dy;
  }
  c00 /= n;
  c01 /= n;
  c11 /= n;

  const trace = c00 + c11;
  const det = c00 * c11 - c01 * c01;
  const disc = Math.sqrt(Math.max(0.0, (trace * trace) / 4.0 - det));
  const eigMajor = trace / 2.0 + disc;
  const eigMinor = trace / 2.0 - disc;
  if (eigMinor <= 1e-3 || eigMajor <= 1e-3) return null;

  const rx = Math.sqrt(2.0 * eigMajor);
  const ry = Math.sqrt(2.0 * eigMinor);
  if (ry < 8.0 || rx > 240.0 || rx / Math.max(ry, 1e-6) > 6.0) return null;

  const startEndDist = Math.hypot(
    points[0][0] - points[n - 1][0],
    points[0][1] - points[n - 1][1]
  );
  if (startEndDist > Math.max(12.0, Math.min(20.0, 0.35 * Math.min(rx, ry)))) return null;
  const meanR = Math.sqrt(0.5 * (rx * rx + ry * ry));
  if (arcLen < 1.45 * Math.PI * meanR) return null;

  const angleRad = 0.5 * Math.atan2(2.0 * c01, c00 - c11);
  const cosA = Math.cos(-angleRad);
  const sinA = Math.sin(-angleRad);
  let errSq = 0.0;
  let posX = false, negX = false, posY = false, negY = false;
  for (let i = 0; i < n; i++) {
    const dx = points[i][0] - cx;
    const dy = points[i][1] - cy;
    const xr = dx * cosA - dy * sinA;
    const yr = dx * sinA + dy * cosA;
    if (xr > 0.3 * rx) posX = true;
    if (xr < -0.3 * rx) negX = true;
    if (yr > 0.3 * ry) posY = true;
    if (yr < -0.3 * ry) negY = true;
    const normR = Math.sqrt((xr / rx) ** 2 + (yr / ry) ** 2);
    errSq += (normR - 1.0) ** 2;
  }
  if (!(posX && negX && posY && negY)) return null;
  const relRmse = Math.sqrt(errSq / n);
  if (relRmse > maxRelRmse) return null;

  return {
    type: "ellipse",
    cx: round2(cx),
    cy: round2(cy),
    rx: round2(rx),
    ry: round2(ry),
    angle: round2((angleRad * 180.0) / Math.PI),
    rmse: Math.round(relRmse * 10000) / 10000,
  };
}

export function fitSchneiderCubicBezier(points, simplifyEps = 2.2, tension = 0.32) {
  const pts = douglasPeucker(points, simplifyEps);
  const n = pts.length;
  if (n < 2) return [];
  if (n === 2) {
    const [p0, p3] = pts;
    const c1 = [round2(p0[0] + (p3[0] - p0[0]) / 3.0), round2(p0[1] + (p3[1] - p0[1]) / 3.0)];
    const c2 = [round2(p0[0] + (2.0 * (p3[0] - p0[0])) / 3.0), round2(p0[1] + (2.0 * (p3[1] - p0[1])) / 3.0)];
    return [[[round2(p0[0]), round2(p0[1])], c1, c2, [round2(p3[0]), round2(p3[1])]]];
  }

  const closed = Math.hypot(pts[0][0] - pts[n - 1][0], pts[0][1] - pts[n - 1][1]) < 6.0 && n >= 4;
  const tangents = [];
  for (let i = 0; i < n; i++) {
    const prevPt = i === 0 ? (closed ? pts[n - 2] : pts[0]) : pts[i - 1];
    const nextPt = i === n - 1 ? (closed ? pts[1] : pts[n - 1]) : pts[i + 1];
    const tx = nextPt[0] - prevPt[0];
    const ty = nextPt[1] - prevPt[1];
    const norm = Math.hypot(tx, ty);
    tangents.push(norm < 1e-8 ? [0.0, 0.0] : [tx / norm, ty / norm]);
  }

  const segments = [];
  for (let i = 0; i < n - 1; i++) {
    const p0 = pts[i];
    const p3 = pts[i + 1];
    const chord = Math.hypot(p3[0] - p0[0], p3[1] - p0[1]);
    const alpha = chord * tension;
    const t0 = tangents[i];
    const t1 = tangents[i + 1];
    const c1 = [round2(p0[0] + alpha * t0[0]), round2(p0[1] + alpha * t0[1])];
    const c2 = [round2(p3[0] - alpha * t1[0]), round2(p3[1] - alpha * t1[1])];
    segments.push([[round2(p0[0]), round2(p0[1])], c1, c2, [round2(p3[0]), round2(p3[1])]]);
  }
  return segments;
}

export function cubicSegmentsToSvgPath(segments, closed = false) {
  if (!segments || segments.length === 0) return "";
  const p0 = segments[0][0];
  const parts = [`M ${p0[0].toFixed(2)} ${p0[1].toFixed(2)}`];
  for (const [, c1, c2, p3] of segments) {
    parts.push(`C ${c1[0].toFixed(2)} ${c1[1].toFixed(2)}, ${c2[0].toFixed(2)} ${c2[1].toFixed(2)}, ${p3[0].toFixed(2)} ${p3[1].toFixed(2)}`);
  }
  if (closed) parts.push("Z");
  return parts.join(" ");
}

export function classifyContourPrimitive(points, junctions = null, preferOrganic = false) {
  if (!points || points.length < 2) return { type: "empty" };

  const circle = fitTaubinCircle(points);
  if (circle !== null) return circle;

  const ellipse = fitEllipseAlgebraic(points);
  if (ellipse !== null) return ellipse;

  const dpPts = douglasPeucker(points, 2.2);
  if (dpPts.length === 2) {
    const snapped = snapOrthogonalPolyline(dpPts, 6.5, junctions);
    return {
      type: "line",
      x1: snapped[0][0],
      y1: snapped[0][1],
      x2: snapped[1][0],
      y2: snapped[1][1],
    };
  }

  let sharpCorners = 0;
  for (let i = 1; i < dpPts.length - 1; i++) {
    const v1x = dpPts[i][0] - dpPts[i - 1][0];
    const v1y = dpPts[i][1] - dpPts[i - 1][1];
    const v2x = dpPts[i + 1][0] - dpPts[i][0];
    const v2y = dpPts[i + 1][1] - dpPts[i][1];
    const n1 = Math.hypot(v1x, v1y);
    const n2 = Math.hypot(v2x, v2y);
    if (n1 > 2.0 && n2 > 2.0) {
      const cosT = Math.max(-1.0, Math.min(1.0, (v1x * v2x + v1y * v2y) / (n1 * n2)));
      const turnDeg = (Math.acos(cosT) * 180.0) / Math.PI;
      if (turnDeg > 28.0) sharpCorners++;
    }
  }

  const closed = Math.hypot(points[0][0] - points[points.length - 1][0], points[0][1] - points[points.length - 1][1]) < 7.0;
  if (!preferOrganic && (dpPts.length <= 6 || sharpCorners >= Math.max(1, dpPts.length - 3))) {
    const snapped = snapOrthogonalPolyline(dpPts, 6.5, junctions);
    return {
      type: closed && snapped.length >= 3 ? "polygon" : "polyline",
      points: snapped,
    };
  }

  const segs = fitSchneiderCubicBezier(points, 2.4);
  return {
    type: "bezier",
    d: cubicSegmentsToSvgPath(segs, closed),
    segments: segs,
    closed,
    node_count: segs.length + 1,
  };
}

export function associateCalloutsToLabels(labels, arrows, maxDist = 95.0) {
  const outLabels = (labels || []).map((lb) => ({ ...lb }));
  const outArrows = (arrows || []).map((ar) => ({ ...ar }));

  outLabels.forEach((lb, idx) => {
    const cid = lb.id || `callout-${idx + 1}`;
    lb.callout_id = cid;
    const bbox = lb.bbox || [lb.x || 0, lb.y || 0, 40, 16];
    const [bx, by, bw, bh] = bbox;
    let bestArIdx = null;
    let bestDist = maxDist;

    outArrows.forEach((ar, aIdx) => {
      const pts = [
        [ar.x1 || 0, ar.y1 || 0],
        [ar.x2 || 0, ar.y2 || 0],
      ];
      for (const pt of pts) {
        const dx = Math.max(bx - pt[0], 0.0, pt[0] - (bx + bw));
        const dy = Math.max(by - pt[1], 0.0, pt[1] - (by + bh));
        const d = Math.hypot(dx, dy);
        if (d < bestDist) {
          bestDist = d;
          bestArIdx = aIdx;
        }
      }
    });

    if (bestArIdx !== null) {
      outArrows[bestArIdx].callout_id = cid;
      lb.target_x = outArrows[bestArIdx].x2;
      lb.target_y = outArrows[bestArIdx].y2;
    }
  });

  return [outLabels, outArrows];
}
