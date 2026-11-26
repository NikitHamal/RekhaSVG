/**
 * RekhaSVG — 1:1 Browser JavaScript Vectorizer Engine (docs/engine/vectorizer.js)
 * Zero-dependency browser & Node.js compatible engine performing:
 *   1. Integral-image Sauvola shadow & illumination normalization
 *   2. Zhang-Suen morphological thinning with junction preservation
 *   3. Connected-component word/text-box exclusion & 2D water-hatch filtering
 *   4. Live ONNX Runtime Web inference for rekha_net_int8.onnx & rekha_text_int8.onnx
 */

import {
  associateCalloutsToLabels,
  classifyContourPrimitive,
  douglasPeucker,
  round2,
} from "./primitives.js";
import { buildInteractiveSvg } from "./svg_builder.js";

const ENGLISH_CHARS = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~";
const DEVANAGARI_CHARS =
  "ँंःअआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह" +
  "़ािीुूृेैोौ्०१२३४५६७८९";
const STEM_SYMBOLS = "αβγθλμπσωΔΦΩ±×÷→⇌≈≠≤≥∞°₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹∩∪⊂∈√∫";
export const REKHA_CHARSET = ["<blank>", ...new Set((ENGLISH_CHARS + DEVANAGARI_CHARS + STEM_SYMBOLS).split(""))];

export class RekhaVectorizerJS {
  constructor(config = {}) {
    this.cfg = {
      sauvolaWindow: config.sauvolaWindow || 31,
      sauvolaK: config.sauvolaK || 0.16,
      minStrokeLen: config.minStrokeLen || 10,
      dashGapMax: config.dashGapMax || 22.0,
      theme: config.theme || "print",
    };
    this.onnxNetSession = null;
    this.onnxTextSession = null;
    this.ort = null;
  }

  /**
   * Loads Rekha-Net and Rekha-Text INT8 ONNX models via onnxruntime-web WASM.
   */
  async tryLoadOnnxModels(
    ort,
    netUrl = "./models/rekha_net_int8.onnx",
    textUrl = "./models/rekha_text_int8.onnx"
  ) {
    if (!ort || !ort.InferenceSession) return false;
    this.ort = ort;
    const hfBase = "https://huggingface.co/NikitHamal/rekhasvg-models/resolve/main";
    try {
      try {
        this.onnxNetSession = await ort.InferenceSession.create(netUrl, { executionProviders: ["wasm"] });
      } catch {
        this.onnxNetSession = await ort.InferenceSession.create(`${hfBase}/rekha_net_int8.onnx`, {
          executionProviders: ["wasm"],
        });
      }
      try {
        this.onnxTextSession = await ort.InferenceSession.create(textUrl, { executionProviders: ["wasm"] });
      } catch {
        this.onnxTextSession = await ort.InferenceSession.create(`${hfBase}/rekha_text_int8.onnx`, {
          executionProviders: ["wasm"],
        });
      }
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Integral-image Sauvola adaptive illumination & camera shadow removal on RGBA buffer.
   * Also flags blue/cyan water-hatch pixels so liquid fills are separated from strokes.
   */
  normalizeIlluminationRGBA(rgba, width, height) {
    const n = width * height;
    const gray = new Uint8Array(n);
    const blueWaterInk = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const r = rgba[i * 4];
      const g = rgba[i * 4 + 1];
      const b = rgba[i * 4 + 2];
      gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
      if (b - r >= 35 && b > 95) {
        blueWaterInk[i] = 1;
      }
    }

    const gw = width + 1;
    const sum = new Float64Array((width + 1) * (height + 1));
    const sqSum = new Float64Array((width + 1) * (height + 1));

    for (let y = 1; y <= height; y++) {
      let rowS = 0;
      let rowSq = 0;
      for (let x = 1; x <= width; x++) {
        const v = gray[(y - 1) * width + (x - 1)];
        rowS += v;
        rowSq += v * v;
        const idx = y * gw + x;
        sum[idx] = sum[(y - 1) * gw + x] + rowS;
        sqSum[idx] = sqSum[(y - 1) * gw + x] + rowSq;
      }
    }

    const half = Math.floor(this.cfg.sauvolaWindow / 2);
    const bgHalf = 22;
    const ink = new Uint8Array(n);
    const normGray = new Uint8Array(n);

    for (let y = 0; y < height; y++) {
      const y0 = Math.max(0, y - half);
      const y1 = Math.min(height, y + half + 1);
      const by0 = Math.max(0, y - bgHalf);
      const by1 = Math.min(height, y + bgHalf + 1);
      for (let x = 0; x < width; x++) {
        const x0 = Math.max(0, x - half);
        const x1 = Math.min(width, x + half + 1);
        const bx0 = Math.max(0, x - bgHalf);
        const bx1 = Math.min(width, x + bgHalf + 1);

        const count = (x1 - x0) * (y1 - y0);
        const bgCount = (bx1 - bx0) * (by1 - by0);
        const s = sum[y1 * gw + x1] - sum[y0 * gw + x1] - sum[y1 * gw + x0] + sum[y0 * gw + x0];
        const sq = sqSum[y1 * gw + x1] - sqSum[y0 * gw + x1] - sqSum[y1 * gw + x0] + sqSum[y0 * gw + x0];
        const bgS = sum[by1 * gw + bx1] - sum[by0 * gw + bx1] - sum[by1 * gw + bx0] + sum[by0 * gw + bx0];

        const mean = s / count;
        const bgMean = Math.max(bgS / bgCount, 20.0);
        const std = Math.sqrt(Math.max(0.0, sq / count - mean * mean));
        const thresh = mean * (1.0 + this.cfg.sauvolaK * (std / 128.0 - 1.0));
        const pIdx = y * width + x;
        const val = gray[pIdx];
        const normVal = Math.min(255, Math.round((val / bgMean) * 245));
        normGray[pIdx] = normVal;
        if (val < thresh && normVal < 218 && !blueWaterInk[pIdx]) {
          ink[pIdx] = 1;
        }
      }
    }

    return { normGray, ink, gray };
  }

  /**
   * Detects connected components on a binary mask and groups character-scale blobs
   * into horizontal word/line bounding boxes so printed text is excluded from stroke tracing.
   */
  extractWordBoxesAndFilterInk(ink, width, height) {
    const visited = new Uint8Array(width * height);
    const comps = [];
    const q = new Int32Array(width * height);

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const start = y * width + x;
        if (!ink[start] || visited[start]) continue;
        let head = 0;
        let tail = 0;
        q[tail++] = start;
        visited[start] = 1;
        let minX = x, maxX = x, minY = y, maxY = y;
        const pixels = [];
        while (head < tail) {
          const curr = q[head++];
          pixels.push(curr);
          const cy = Math.floor(curr / width);
          const cx = curr - cy * width;
          if (cx < minX) minX = cx;
          if (cx > maxX) maxX = cx;
          if (cy < minY) minY = cy;
          if (cy > maxY) maxY = cy;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (dx === 0 && dy === 0) continue;
              const nx = cx + dx;
              const ny = cy + dy;
              if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                const nIdx = ny * width + nx;
                if (ink[nIdx] && !visited[nIdx]) {
                  visited[nIdx] = 1;
                  q[tail++] = nIdx;
                }
              }
            }
          }
        }
        const cw = maxX - minX + 1;
        const ch = maxY - minY + 1;
        const area = pixels.length;
        if (area < 5) {
          for (const p of pixels) ink[p] = 0;
          continue;
        }
        const fillRatio = area / Math.max(1, cw * ch);
        const isCharCandidate =
          cw >= 3 && cw <= 32 && ch >= 5 && ch <= 32 && area >= 8 && area <= 420 &&
          !(cw > 2.6 * ch && ch <= 5) && fillRatio >= 0.14 && fillRatio <= 0.85;
        if (isCharCandidate) {
          comps.push({ minX, minY, cw, ch, area, pixels });
        }
      }
    }

    const used = new Uint8Array(comps.length);
    const wordBoxes = [];
    for (let i = 0; i < comps.length; i++) {
      if (used[i]) continue;
      used[i] = 1;
      const group = [comps[i]];
      let changed = true;
      while (changed) {
        changed = false;
        for (let j = 0; j < comps.length; j++) {
          if (used[j]) continue;
          const b = comps[j];
          const cyB = b.minY + 0.5 * b.ch;
          for (const g of group) {
            const cyG = g.minY + 0.5 * g.ch;
            const maxH = Math.max(b.ch, g.ch);
            if (Math.abs(cyB - cyG) <= 0.65 * maxH) {
              const gapX = Math.max(b.minX - (g.minX + g.cw), g.minX - (b.minX + b.cw), 0);
              if (gapX <= 1.65 * maxH) {
                used[j] = 1;
                group.push(b);
                changed = true;
                break;
              }
            }
          }
        }
      }
      const isPlusMinus =
        group.length === 1 &&
        group[0].cw >= 6 &&
        group[0].cw <= 18 &&
        group[0].cw / Math.max(1, group[0].ch) >= 0.75 &&
        group[0].cw / Math.max(1, group[0].ch) <= 1.35;
      if (group.length >= 2 || isPlusMinus) {
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
        for (const g of group) {
          if (g.minX < x0) x0 = g.minX;
          if (g.minY < y0) y0 = g.minY;
          if (g.minX + g.cw > x1) x1 = g.minX + g.cw;
          if (g.minY + g.ch > y1) y1 = g.minY + g.ch;
          for (const p of g.pixels) ink[p] = 0;
        }
        wordBoxes.push([Math.max(0, x0 - 2), Math.max(0, y0 - 2), Math.min(width, x1 + 2) - Math.max(0, x0 - 2), Math.min(height, y1 + 2) - Math.max(0, y0 - 2)]);
      }
    }
    wordBoxes.sort((a, b) => Math.floor(a[1] / 18) - Math.floor(b[1] / 18) || a[0] - b[0]);
    return wordBoxes;
  }

  /**
   * Single-pass Zhang-Suen morphological thinning to 1px skeleton.
   */
  thinMask(binary, width, height, maxIters = 18) {
    const skel = new Uint8Array(binary);
    const toClear = [];
    for (let iter = 0; iter < maxIters; iter++) {
      let changed = false;
      for (let step = 0; step < 2; step++) {
        toClear.length = 0;
        for (let y = 1; y < height - 1; y++) {
          for (let x = 1; x < width - 1; x++) {
            const idx = y * width + x;
            if (!skel[idx]) continue;
            const p2 = skel[(y - 1) * width + x];
            const p3 = skel[(y - 1) * width + (x + 1)];
            const p4 = skel[y * width + (x + 1)];
            const p5 = skel[(y + 1) * width + (x + 1)];
            const p6 = skel[(y + 1) * width + x];
            const p7 = skel[(y + 1) * width + (x - 1)];
            const p8 = skel[y * width + (x - 1)];
            const p9 = skel[(y - 1) * width + (x - 1)];

            const B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
            if (B < 2 || B > 6) continue;
            const A =
              (!p2 && p3 ? 1 : 0) +
              (!p3 && p4 ? 1 : 0) +
              (!p4 && p5 ? 1 : 0) +
              (!p5 && p6 ? 1 : 0) +
              (!p6 && p7 ? 1 : 0) +
              (!p7 && p8 ? 1 : 0) +
              (!p8 && p9 ? 1 : 0) +
              (!p9 && p2 ? 1 : 0);
            if (A !== 1) continue;
            if (step === 0) {
              if (p2 * p4 * p6 !== 0 || p4 * p6 * p8 !== 0) continue;
            } else {
              if (p2 * p4 * p8 !== 0 || p2 * p6 * p8 !== 0) continue;
            }
            toClear.push(idx);
          }
        }
        if (toClear.length > 0) {
          changed = true;
          for (const idx of toClear) skel[idx] = 0;
        }
      }
      if (!changed) break;
    }
    return skel;
  }

  /**
   * Traces connected 1px skeleton chains and detects topological junctions.
   */
  traceSkeletonChainsAndJunctions(skel, width, height) {
    const visited = new Uint8Array(width * height);
    const junctions = [];
    const chains = [];
    const dirs = [
      [-1, -1], [0, -1], [1, -1],
      [-1, 0],           [1, 0],
      [-1, 1],  [0, 1],  [1, 1],
    ];

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        if (!skel[y * width + x]) continue;
        let deg = 0;
        for (const [dx, dy] of dirs) {
          if (skel[(y + dy) * width + (x + dx)]) deg++;
        }
        if (deg >= 3) junctions.push([x, y]);
      }
    }

    for (let y = 1; y < height - 1; y++) {
      for (let x = 1; x < width - 1; x++) {
        const startIdx = y * width + x;
        if (!skel[startIdx] || visited[startIdx]) continue;
        const chain = [[x, y]];
        visited[startIdx] = 1;
        let cx = x;
        let cy = y;
        while (true) {
          let moved = false;
          for (const [dx, dy] of dirs) {
            const nx = cx + dx;
            const ny = cy + dy;
            if (nx <= 0 || nx >= width - 1 || ny <= 0 || ny >= height - 1) continue;
            const nIdx = ny * width + nx;
            if (skel[nIdx] && !visited[nIdx]) {
              visited[nIdx] = 1;
              chain.push([nx, ny]);
              cx = nx;
              cy = ny;
              moved = true;
              break;
            }
          }
          if (!moved) break;
        }
        if (chain.length >= 3) chains.push(chain);
      }
    }
    return { chains, junctions };
  }

  /**
   * Groups strictly colinear short dashed segments while rejecting 2D water-hatch clusters.
   */
  groupColinearDashes(dashedCenters) {
    const n = dashedCenters.length;
    if (n < 2) return [];
    const used = new Uint8Array(n);
    // Reject 2D stipple/water-hatch clusters (>= 4 neighbors within 30px)
    for (let i = 0; i < n; i++) {
      let nearCount = 0;
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        if (Math.hypot(dashedCenters[j][0] - dashedCenters[i][0], dashedCenters[j][1] - dashedCenters[i][1]) <= 30.0) {
          nearCount++;
        }
      }
      if (nearCount >= 4) used[i] = 1;
    }

    const dashedOut = [];
    for (let i = 0; i < n; i++) {
      if (used[i]) continue;
      used[i] = 1;
      const group = [dashedCenters[i]];
      while (true) {
        const last = group[group.length - 1];
        let bestJ = -1;
        let bestD = this.cfg.dashGapMax;
        for (let j = 0; j < n; j++) {
          if (used[j]) continue;
          const vx = dashedCenters[j][0] - last[0];
          const vy = dashedCenters[j][1] - last[1];
          const d = Math.hypot(vx, vy);
          if (d > 2.0 && d < bestD) {
            if (group.length >= 2) {
              const prev = group[group.length - 2];
              const v1x = last[0] - prev[0];
              const v1y = last[1] - prev[1];
              const n1 = Math.hypot(v1x, v1y);
              if (n1 > 1e-3) {
                const cosA = (v1x * vx + v1y * vy) / (n1 * d);
                if (cosA < 0.92) continue;
              }
              const vcx = dashedCenters[j][0] - group[0][0];
              const vcy = dashedCenters[j][1] - group[0][1];
              const nc = Math.hypot(vcx, vcy);
              if (nc > 1e-3) {
                const cosC = (vcx * vx + vcy * vy) / (nc * d);
                if (cosC < 0.90) continue;
              }
            }
            bestD = d;
            bestJ = j;
          }
        }
        if (bestJ >= 0) {
          used[bestJ] = 1;
          group.push(dashedCenters[bestJ]);
        } else {
          break;
        }
      }
      const totalSpan = Math.hypot(
        group[group.length - 1][0] - group[0][0],
        group[group.length - 1][1] - group[0][1]
      );
      if (group.length >= 3 || (group.length === 2 && totalSpan >= 18.0)) {
        dashedOut.push({
          type: group.length === 2 ? "line" : "polyline",
          x1: round2(group[0][0]),
          y1: round2(group[0][1]),
          x2: round2(group[group.length - 1][0]),
          y2: round2(group[group.length - 1][1]),
          points: douglasPeucker(group, 2.5).map((p) => [round2(p[0]), round2(p[1])]),
          dasharray: "6,5",
        });
      }
    }
    return dashedOut;
  }

  /**
   * Compiles pre-extracted chains into crisp interactive SVG.
   */
  compileFromChains({
    width,
    height,
    strokeChains = [],
    dashedCenters = [],
    arrowChains = [],
    fillContours = [],
    junctions = [],
    labels = [],
    metadata = {},
  }) {
    const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
    const preferOrg = metadata.domain === "biology";
    const fillsOut = [];
    for (const fPts of fillContours) {
      const prim = classifyContourPrimitive(fPts, junctions, true);
      if (prim.type !== "empty") fillsOut.push(prim);
    }

    const strokesOut = [];
    let rawNodes = 0;
    let compiledNodes = 0;
    for (const chain of strokeChains) {
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
      for (const [px, py] of chain) {
        if (px < minX) minX = px;
        if (px > maxX) maxX = px;
        if (py < minY) minY = py;
        if (py > maxY) maxY = py;
      }
      if (maxX - minX < 10.0 && maxY - minY < 10.0) continue;

      rawNodes += chain.length;
      const prim = classifyContourPrimitive(chain, junctions, preferOrg);
      if (prim.type === "empty") continue;
      if (prim.type === "circle" || prim.type === "ellipse" || prim.type === "line") {
        compiledNodes += 2;
      } else if (prim.type === "polyline" || prim.type === "polygon") {
        compiledNodes += (prim.points || []).length;
      } else {
        compiledNodes += prim.node_count || 4;
      }
      strokesOut.push(prim);
    }

    const arrowsOut = [];
    for (const aChain of arrowChains) {
      const snapped = douglasPeucker(aChain, 2.2);
      if (snapped.length >= 2) {
        arrowsOut.push({
          type: "line",
          x1: round2(snapped[0][0]),
          y1: round2(snapped[0][1]),
          x2: round2(snapped[snapped.length - 1][0]),
          y2: round2(snapped[snapped.length - 1][1]),
        });
      }
    }

    const dashedOut = this.groupColinearDashes(dashedCenters);
    const [labelsLinked, arrowsLinked] = associateCalloutsToLabels(labels, arrowsOut);
    const svg = buildInteractiveSvg({
      width,
      height,
      fills: fillsOut,
      strokes: strokesOut,
      dashed: dashedOut,
      arrows: arrowsLinked,
      labels: labelsLinked,
      themeName: this.cfg.theme,
      metadata,
    });

    const t1 = typeof performance !== "undefined" ? performance.now() : Date.now();
    return {
      svg,
      width,
      height,
      fills: fillsOut,
      strokes: strokesOut,
      dashed: dashedOut,
      arrows: arrowsLinked,
      labels: labelsLinked,
      latency_ms: round2(t1 - t0),
      raw_nodes: rawNodes,
      compiled_nodes: compiledNodes,
      node_reduction_ratio: round2(Math.max(1.0, rawNodes / Math.max(1, compiledNodes))),
    };
  }

  /**
   * Synchronous end-to-end vectorization from raw RGBA pixel array in browser.
   */
  vectorizeRGBA(rgba, width, height, labels = [], metadata = {}) {
    const { ink } = this.normalizeIlluminationRGBA(rgba, width, height);
    const activeLabels = Array.isArray(labels) ? labels.slice() : [];
    if (activeLabels.length > 0) {
      for (const lb of activeLabels) {
        const [bx, by, bw, bh] = lb.bbox || [lb.x || 0, lb.y || 0, 50, 18];
        const x0 = Math.max(0, Math.floor(bx - 3));
        const y0 = Math.max(0, Math.floor(by - 3));
        const x1 = Math.min(width, Math.ceil(bx + bw + 3));
        const y1 = Math.min(height, Math.ceil(by + bh + 3));
        for (let y = y0; y < y1; y++) {
          for (let x = x0; x < x1; x++) {
            ink[y * width + x] = 0;
          }
        }
      }
    } else {
      // Automatically isolate printed words/chars so they are never traced into jagged stick paths
      this.extractWordBoxesAndFilterInk(ink, width, height);
    }

    const skel = this.thinMask(ink, width, height);
    const { chains, junctions } = this.traceSkeletonChainsAndJunctions(skel, width, height);
    const strokeChains = [];
    const dashedCenters = [];
    for (const ch of chains) {
      if (ch.length < this.cfg.minStrokeLen) {
        const mid = ch[Math.floor(ch.length / 2)];
        dashedCenters.push(mid);
      } else {
        strokeChains.push(ch);
      }
    }

    return this.compileFromChains({
      width,
      height,
      strokeChains,
      dashedCenters,
      arrowChains: metadata.arrowChains || [],
      fillContours: metadata.fillContours || [],
      junctions,
      labels: activeLabels,
      metadata,
    });
  }

  /**
   * Async neural-geometric pipeline using Rekha-Net + Rekha-Text INT8 ONNX in browser WASM.
   */
  async vectorizeRGBAAsync(rgba, width, height, labels = [], metadata = {}) {
    if (!this.onnxNetSession || !this.ort) {
      return this.vectorizeRGBA(rgba, width, height, labels, metadata);
    }
    const { ink, gray } = this.normalizeIlluminationRGBA(rgba, width, height);
    const floatRgb = new Float32Array(3 * width * height);
    const hw = width * height;
    for (let i = 0; i < hw; i++) {
      floatRgb[i] = rgba[i * 4] / 255.0;
      floatRgb[hw + i] = rgba[i * 4 + 1] / 255.0;
      floatRgb[2 * hw + i] = rgba[i * 4 + 2] / 255.0;
    }

    const imgTensor = new this.ort.Tensor("float32", floatRgb, [1, 3, height, width]);
    const netOut = await this.onnxNetSession.run({ image: imgTensor });
    const logits = netOut.logits_7ch.data;

    const strokeMask = new Uint8Array(hw);
    const textMask = new Uint8Array(hw);
    for (let i = 0; i < hw; i++) {
      const pStroke = 1.0 / (1.0 + Math.exp(-logits[1 * hw + i]));
      const pText = 1.0 / (1.0 + Math.exp(-logits[5 * hw + i]));
      if (pText >= 0.40) textMask[i] = 1;
      else if (pStroke >= 0.40 || ink[i]) strokeMask[i] = 1;
    }

    // Dilate text exclusion mask by 3px
    for (let y = 2; y < height - 2; y++) {
      for (let x = 2; x < width - 2; x++) {
        if (textMask[y * width + x]) {
          for (let dy = -2; dy <= 2; dy++) {
            for (let dx = -2; dx <= 2; dx++) {
              strokeMask[(y + dy) * width + (x + dx)] = 0;
            }
          }
        }
      }
    }

    let activeLabels = Array.isArray(labels) && labels.length > 0 ? labels.slice() : [];
    const wordBoxes = this.extractWordBoxesAndFilterInk(strokeMask, width, height);
    if (activeLabels.length === 0 && this.onnxTextSession && wordBoxes.length > 0) {
      for (let idx = 0; idx < wordBoxes.length; idx++) {
        const [bx, by, bw, bh] = wordBoxes[idx];
        const cropData = new Float32Array(32 * 192).fill(1.0);
        const scale = 28.0 / Math.max(1, bh);
        const newW = Math.max(8, Math.min(188, Math.round(bw * scale)));
        for (let cy = 0; cy < 28; cy++) {
          const sy = Math.min(height - 1, Math.max(0, Math.floor(by + cy / scale)));
          for (let cx = 0; cx < newW; cx++) {
            const sx = Math.min(width - 1, Math.max(0, Math.floor(bx + cx / scale)));
            cropData[(cy + 2) * 192 + (cx + 2)] = gray[sy * width + sx] / 255.0;
          }
        }
        const cropTensor = new this.ort.Tensor("float32", cropData, [1, 1, 32, 192]);
        const txtOut = await this.onnxTextSession.run({ crop: cropTensor });
        const lp = txtOut.log_probs.data;
        const vocabSize = REKHA_CHARSET.length;
        const chars = [];
        let prev = 0;
        for (let t = 0; t < 48; t++) {
          let bestIdx = 0;
          let bestVal = -Infinity;
          const base = t * vocabSize;
          for (let v = 0; v < vocabSize; v++) {
            if (lp[base + v] > bestVal) {
              bestVal = lp[base + v];
              bestIdx = v;
            }
          }
          if (bestIdx !== 0 && bestIdx !== prev && bestIdx < vocabSize) {
            chars.push(REKHA_CHARSET[bestIdx]);
          }
          prev = bestIdx;
        }
        const textPred = chars.join("").trim();
        if (textPred.length > 0) {
          const cid = `callout-${activeLabels.length + 1}`;
          activeLabels.push({
            id: cid,
            callout_id: cid,
            text: textPred,
            lang: /[\u0900-\u097F]/.test(textPred) ? "ne" : "en",
            bbox: [bx, by, bw, bh],
            x: bx + 2,
            y: by + bh - 3,
            font_size: 14.0,
          });
        }
      }
    }

    const skel = this.thinMask(strokeMask, width, height);
    const { chains, junctions } = this.traceSkeletonChainsAndJunctions(skel, width, height);
    const strokeChains = [];
    const dashedCenters = [];
    for (const ch of chains) {
      if (ch.length < this.cfg.minStrokeLen) {
        dashedCenters.push(ch[Math.floor(ch.length / 2)]);
      } else {
        strokeChains.push(ch);
      }
    }

    return this.compileFromChains({
      width,
      height,
      strokeChains,
      dashedCenters,
      arrowChains: metadata.arrowChains || [],
      fillContours: metadata.fillContours || [],
      junctions,
      labels: activeLabels,
      metadata,
    });
  }
}
