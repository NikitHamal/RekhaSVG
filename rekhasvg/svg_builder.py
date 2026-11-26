"""
Semantic Interactive SVG DOM Builder for RekhaSVG.
Emits clean, accessible, interactive layered SVG with:
  - <g id="layer-fills">
  - <g id="layer-strokes">
  - <g id="layer-dashed">
  - <g id="layer-arrows">
  - <g id="layer-labels"> (with data-callout, data-quiz-answer, data-bbox)
Mirrored 1:1 in docs/engine/svg_builder.js.
"""
from __future__ import annotations
from dataclasses import dataclass
import html
from typing import Dict, List, Optional


@dataclass
class SVGTheme:
    name: str = "print"
    background: str = "#FFFFFF"
    stroke_color: str = "#1D1B20"
    dashed_color: str = "#49454F"
    arrow_color: str = "#6750A4"
    label_color: str = "#1D1B20"
    fill_default: str = "#E8DEF8"
    fill_opacity: float = 0.55
    stroke_width: float = 2.2


THEMES: Dict[str, SVGTheme] = {
    "print": SVGTheme(
        name="print",
        background="#FFFFFF",
        stroke_color="#1D1B20",
        dashed_color="#49454F",
        arrow_color="#4F378B",
        label_color="#1D1B20",
        fill_default="#E8DEF8",
    ),
    "m3x_dark": SVGTheme(
        name="m3x_dark",
        background="#141218",
        stroke_color="#E6E0E9",
        dashed_color="#CAC4D0",
        arrow_color="#D0BCFF",
        label_color="#F5EFF7",
        fill_default="#4F378B",
    ),
    "blueprint": SVGTheme(
        name="blueprint",
        background="#0D233A",
        stroke_color="#E3F2FD",
        dashed_color="#90CAF9",
        arrow_color="#64B5F6",
        label_color="#FFFFFF",
        fill_default="#1E4976",
    ),
    "color_coded": SVGTheme(
        name="color_coded",
        background="#FEF7FF",
        stroke_color="#1D1B20",
        dashed_color="#B3261E",
        arrow_color="#6750A4",
        label_color="#006A60",
        fill_default="#FFD8E4",
    ),
}


def _render_primitive_el(prim: Dict, attrs: str) -> str:
    ptype = prim.get("type", "polyline")
    if ptype == "line":
        return (
            f'<line x1="{prim["x1"]:.2f}" y1="{prim["y1"]:.2f}" '
            f'x2="{prim["x2"]:.2f}" y2="{prim["y2"]:.2f}" {attrs} />'
        )
    if ptype == "circle":
        return (
            f'<circle cx="{prim["cx"]:.2f}" cy="{prim["cy"]:.2f}" '
            f'r="{prim["r"]:.2f}" {attrs} />'
        )
    if ptype == "ellipse":
        ang = prim.get("angle", 0.0)
        rot = f' transform="rotate({ang:.2f} {prim["cx"]:.2f} {prim["cy"]:.2f})"' if abs(ang) > 0.5 else ""
        return (
            f'<ellipse cx="{prim["cx"]:.2f}" cy="{prim["cy"]:.2f}" '
            f'rx="{prim["rx"]:.2f}" ry="{prim["ry"]:.2f}"{rot} {attrs} />'
        )
    if ptype in ("polyline", "polygon"):
        pts = prim.get("points", [])
        if not pts:
            return ""
        pts_str = " ".join(f"{p[0]:.2f},{p[1]:.2f}" for p in pts)
        tag = "polygon" if ptype == "polygon" else "polyline"
        return f'<{tag} points="{pts_str}" {attrs} />'
    if ptype == "bezier":
        d = prim.get("d", "")
        if not d:
            return ""
        return f'<path d="{d}" {attrs} />'
    return ""


def build_interactive_svg(
    width: int,
    height: int,
    fills: List[Dict],
    strokes: List[Dict],
    dashed: List[Dict],
    arrows: List[Dict],
    labels: List[Dict],
    theme_name: str = "print",
    metadata: Optional[Dict] = None,
) -> str:
    """
    Constructs a crisp, interactive, multi-layer SVG document with semantic layers
    and embedded callout linkage attributes.
    """
    theme = THEMES.get(theme_name, THEMES["print"])
    meta_domain = html.escape(str((metadata or {}).get("domain", "stem")))
    meta_title = html.escape(str((metadata or {}).get("title", "RekhaSVG Textbook Diagram")))

    lines: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" data-rekhasvg-version="1.0" '
        f'data-domain="{meta_domain}" data-theme="{theme.name}" '
        f'style="background:{theme.background};font-family:\'Poppins\',\'Noto Sans Devanagari\',sans-serif;">',
        f"  <title>{meta_title}</title>",
        "  <defs>",
        f'    <marker id="rekha-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" '
        f'markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">',
        f'      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="{theme.arrow_color}" />',
        "    </marker>",
        "  </defs>",
        f'  <rect id="rekha-bg" width="{width}" height="{height}" fill="{theme.background}" />',
    ]

    # 1. Layer: Fills
    lines.append('  <g id="layer-fills" data-layer="fills">')
    for idx, f_item in enumerate(fills):
        fill_col = f_item.get("fill", theme.fill_default)
        opacity = f_item.get("opacity", theme.fill_opacity)
        attrs = f'fill="{fill_col}" fill-opacity="{opacity:.2f}" stroke="none" data-fill-id="fill-{idx + 1}"'
        el = _render_primitive_el(f_item, attrs)
        if el:
            lines.append("    " + el)
    lines.append("  </g>")

    # 2. Layer: Solid Strokes
    lines.append('  <g id="layer-strokes" data-layer="strokes">')
    for idx, s_item in enumerate(strokes):
        sw = s_item.get("stroke_width", theme.stroke_width)
        sc = s_item.get("stroke", theme.stroke_color)
        attrs = (
            f'fill="none" stroke="{sc}" stroke-width="{sw:.2f}" '
            f'stroke-linecap="round" stroke-linejoin="round" data-stroke-id="stroke-{idx + 1}"'
        )
        el = _render_primitive_el(s_item, attrs)
        if el:
            lines.append("    " + el)
    lines.append("  </g>")

    # 3. Layer: Dashed / Construction Lines
    lines.append('  <g id="layer-dashed" data-layer="dashed">')
    for idx, d_item in enumerate(dashed):
        sw = d_item.get("stroke_width", max(1.6, theme.stroke_width * 0.8))
        dc = d_item.get("stroke", theme.dashed_color)
        dash_pat = d_item.get("dasharray", "6,5")
        attrs = (
            f'fill="none" stroke="{dc}" stroke-width="{sw:.2f}" '
            f'stroke-dasharray="{dash_pat}" stroke-linecap="round" data-dashed-id="dashed-{idx + 1}"'
        )
        el = _render_primitive_el(d_item, attrs)
        if el:
            lines.append("    " + el)
    lines.append("  </g>")

    # 4. Layer: Arrows & Callout Pointers
    lines.append('  <g id="layer-arrows" data-layer="arrows">')
    for idx, a_item in enumerate(arrows):
        sw = a_item.get("stroke_width", max(1.8, theme.stroke_width * 0.85))
        ac = a_item.get("stroke", theme.arrow_color)
        cid = html.escape(str(a_item.get("callout_id", f"callout-{idx + 1}")))
        attrs = (
            f'fill="none" stroke="{ac}" stroke-width="{sw:.2f}" '
            f'stroke-linecap="round" marker-end="url(#rekha-arrow)" '
            f'data-callout="{cid}" class="rekha-callout-arrow"'
        )
        el = _render_primitive_el(a_item, attrs)
        if el:
            lines.append("    " + el)
    lines.append("  </g>")

    # 5. Layer: Editable & Quiz-Ready Semantic Labels (English + Devanagari + STEM)
    lines.append('  <g id="layer-labels" data-layer="labels">')
    for idx, lb in enumerate(labels):
        text_val = html.escape(str(lb.get("text", "")))
        if not text_val:
            continue
        cid = html.escape(str(lb.get("callout_id", f"callout-{idx + 1}")))
        bx, by, bw, bh = lb.get("bbox", [lb.get("x", 0), lb.get("y", 0), 64, 20])
        tx = float(lb.get("x", bx + 2))
        ty = float(lb.get("y", by + bh - 4))
        fsize = float(lb.get("font_size", 14.0))
        lang = html.escape(str(lb.get("lang", "en")))
        lines.append(
            f'    <g class="rekha-label-group" data-callout="{cid}" '
            f'data-quiz-answer="{text_val}" data-lang="{lang}" '
            f'data-bbox="{bx:.1f},{by:.1f},{bw:.1f},{bh:.1f}" style="cursor:pointer;">'
        )
        lines.append(
            f'      <rect class="rekha-label-chip" x="{bx - 3:.1f}" y="{by - 2:.1f}" '
            f'width="{bw + 6:.1f}" height="{bh + 4:.1f}" rx="5" '
            f'fill="{theme.background}" fill-opacity="0.82" stroke="none" />'
        )
        lines.append(
            f'      <text x="{tx:.1f}" y="{ty:.1f}" fill="{theme.label_color}" '
            f'font-size="{fsize:.1f}" font-weight="600" data-callout="{cid}">{text_val}</text>'
        )
        lines.append("    </g>")
    lines.append("  </g>")
    lines.append("</svg>")
    return "\n".join(lines)
