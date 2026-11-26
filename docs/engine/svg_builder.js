/**
 * RekhaSVG — 1:1 Browser JavaScript Mirror of rekhasvg/svg_builder.py
 * Emits semantic interactive SVG layers (<g id="layer-fills">, <g id="layer-strokes">,
 * <g id="layer-dashed">, <g id="layer-arrows">, <g id="layer-labels">).
 */

export const THEMES = {
  print: {
    name: "print",
    background: "#FFFFFF",
    stroke_color: "#1D1B20",
    dashed_color: "#49454F",
    arrow_color: "#4F378B",
    label_color: "#1D1B20",
    fill_default: "#E8DEF8",
    fill_opacity: 0.55,
    stroke_width: 2.2,
  },
  m3x_dark: {
    name: "m3x_dark",
    background: "#141218",
    stroke_color: "#E6E0E9",
    dashed_color: "#CAC4D0",
    arrow_color: "#D0BCFF",
    label_color: "#F5EFF7",
    fill_default: "#4F378B",
    fill_opacity: 0.55,
    stroke_width: 2.2,
  },
  blueprint: {
    name: "blueprint",
    background: "#0D233A",
    stroke_color: "#E3F2FD",
    dashed_color: "#90CAF9",
    arrow_color: "#64B5F6",
    label_color: "#FFFFFF",
    fill_default: "#1E4976",
    fill_opacity: 0.55,
    stroke_width: 2.2,
  },
  color_coded: {
    name: "color_coded",
    background: "#FEF7FF",
    stroke_color: "#1D1B20",
    dashed_color: "#B3261E",
    arrow_color: "#6750A4",
    label_color: "#006A60",
    fill_default: "#FFD8E4",
    fill_opacity: 0.55,
    stroke_width: 2.2,
  },
};

function escapeXml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderPrimitiveEl(prim, attrs) {
  const ptype = prim.type || "polyline";
  if (ptype === "line") {
    return `<line x1="${Number(prim.x1).toFixed(2)}" y1="${Number(prim.y1).toFixed(2)}" x2="${Number(prim.x2).toFixed(2)}" y2="${Number(prim.y2).toFixed(2)}" ${attrs} />`;
  }
  if (ptype === "circle") {
    return `<circle cx="${Number(prim.cx).toFixed(2)}" cy="${Number(prim.cy).toFixed(2)}" r="${Number(prim.r).toFixed(2)}" ${attrs} />`;
  }
  if (ptype === "ellipse") {
    const ang = Number(prim.angle || 0.0);
    const rot = Math.abs(ang) > 0.5 ? ` transform="rotate(${ang.toFixed(2)} ${Number(prim.cx).toFixed(2)} ${Number(prim.cy).toFixed(2)})"` : "";
    return `<ellipse cx="${Number(prim.cx).toFixed(2)}" cy="${Number(prim.cy).toFixed(2)}" rx="${Number(prim.rx).toFixed(2)}" ry="${Number(prim.ry).toFixed(2)}"${rot} ${attrs} />`;
  }
  if (ptype === "polyline" || ptype === "polygon") {
    const pts = prim.points || [];
    if (!pts.length) return "";
    const ptsStr = pts.map((p) => `${Number(p[0]).toFixed(2)},${Number(p[1]).toFixed(2)}`).join(" ");
    const tag = ptype === "polygon" ? "polygon" : "polyline";
    return `<${tag} points="${ptsStr}" ${attrs} />`;
  }
  if (ptype === "bezier") {
    const d = prim.d || "";
    if (!d) return "";
    return `<path d="${d}" ${attrs} />`;
  }
  return "";
}

export function buildInteractiveSvg({
  width,
  height,
  fills = [],
  strokes = [],
  dashed = [],
  arrows = [],
  labels = [],
  themeName = "print",
  metadata = {},
}) {
  const theme = THEMES[themeName] || THEMES.print;
  const metaDomain = escapeXml(metadata.domain || "stem");
  const metaTitle = escapeXml(metadata.title || "RekhaSVG Textbook Diagram");

  const lines = [
    `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" data-rekhasvg-version="1.0" data-domain="${metaDomain}" data-theme="${theme.name}" style="background:${theme.background};font-family:'Poppins','Noto Sans Devanagari',sans-serif;">`,
    `  <title>${metaTitle}</title>`,
    `  <defs>`,
    `    <marker id="rekha-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">`,
    `      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="${theme.arrow_color}" />`,
    `    </marker>`,
    `  </defs>`,
    `  <rect id="rekha-bg" width="${width}" height="${height}" fill="${theme.background}" />`,
  ];

  // 1. Fills
  lines.push(`  <g id="layer-fills" data-layer="fills">`);
  fills.forEach((fItem, idx) => {
    const fillCol = fItem.fill || theme.fill_default;
    const opacity = fItem.opacity ?? theme.fill_opacity;
    const attrs = `fill="${fillCol}" fill-opacity="${Number(opacity).toFixed(2)}" stroke="none" data-fill-id="fill-${idx + 1}"`;
    const el = renderPrimitiveEl(fItem, attrs);
    if (el) lines.push("    " + el);
  });
  lines.push(`  </g>`);

  // 2. Strokes
  lines.push(`  <g id="layer-strokes" data-layer="strokes">`);
  strokes.forEach((sItem, idx) => {
    const sw = sItem.stroke_width ?? theme.stroke_width;
    const sc = sItem.stroke || theme.stroke_color;
    const attrs = `fill="none" stroke="${sc}" stroke-width="${Number(sw).toFixed(2)}" stroke-linecap="round" stroke-linejoin="round" data-stroke-id="stroke-${idx + 1}"`;
    const el = renderPrimitiveEl(sItem, attrs);
    if (el) lines.push("    " + el);
  });
  lines.push(`  </g>`);

  // 3. Dashed
  lines.push(`  <g id="layer-dashed" data-layer="dashed">`);
  dashed.forEach((dItem, idx) => {
    const sw = dItem.stroke_width ?? Math.max(1.6, theme.stroke_width * 0.8);
    const dc = dItem.stroke || theme.dashed_color;
    const dashPat = dItem.dasharray || "6,5";
    const attrs = `fill="none" stroke="${dc}" stroke-width="${Number(sw).toFixed(2)}" stroke-dasharray="${dashPat}" stroke-linecap="round" data-dashed-id="dashed-${idx + 1}"`;
    const el = renderPrimitiveEl(dItem, attrs);
    if (el) lines.push("    " + el);
  });
  lines.push(`  </g>`);

  // 4. Arrows
  lines.push(`  <g id="layer-arrows" data-layer="arrows">`);
  arrows.forEach((aItem, idx) => {
    const sw = aItem.stroke_width ?? Math.max(1.8, theme.stroke_width * 0.85);
    const ac = aItem.stroke || theme.arrow_color;
    const cid = escapeXml(aItem.callout_id || `callout-${idx + 1}`);
    const attrs = `fill="none" stroke="${ac}" stroke-width="${Number(sw).toFixed(2)}" stroke-linecap="round" marker-end="url(#rekha-arrow)" data-callout="${cid}" class="rekha-callout-arrow"`;
    const el = renderPrimitiveEl(aItem, attrs);
    if (el) lines.push("    " + el);
  });
  lines.push(`  </g>`);

  // 5. Labels
  lines.push(`  <g id="layer-labels" data-layer="labels">`);
  labels.forEach((lb, idx) => {
    const textVal = escapeXml(lb.text || "");
    if (!textVal) return;
    const cid = escapeXml(lb.callout_id || `callout-${idx + 1}`);
    const bbox = lb.bbox || [lb.x || 0, lb.y || 0, 64, 20];
    const [bx, by, bw, bh] = bbox.map(Number);
    const tx = Number(lb.x ?? bx + 2);
    const ty = Number(lb.y ?? by + bh - 4);
    const fsize = Number(lb.font_size ?? 14.0);
    const lang = escapeXml(lb.lang || "en");

    lines.push(
      `    <g class="rekha-label-group" data-callout="${cid}" data-quiz-answer="${textVal}" data-lang="${lang}" data-bbox="${bx.toFixed(1)},${by.toFixed(1)},${bw.toFixed(1)},${bh.toFixed(1)}" style="cursor:pointer;">`
    );
    lines.push(
      `      <rect class="rekha-label-chip" x="${(bx - 3).toFixed(1)}" y="${(by - 2).toFixed(1)}" width="${(bw + 6).toFixed(1)}" height="${(bh + 4).toFixed(1)}" rx="5" fill="${theme.background}" fill-opacity="0.82" stroke="none" />`
    );
    lines.push(
      `      <text x="${tx.toFixed(1)}" y="${ty.toFixed(1)}" fill="${theme.label_color}" font-size="${fsize.toFixed(1)}" font-weight="600" data-callout="${cid}">${textVal}</text>`
    );
    lines.push(`    </g>`);
  });
  lines.push(`  </g>`);
  lines.push(`</svg>`);
  return lines.join("\n");
}
