"""
RekhaSVG — On-Device Textbook Diagram to Crisp Interactive SVG Converter
Three-Stage Neural-Geometric Hybrid Pipeline:
  1. Rekha-Net (~6-8 MB INT8 ONNX, 512x512 MobileConvNeXt/NAFNet-Lite 6-channel semantic decomposer)
  2. Rekha-Text (~4-5 MB INT8 ONNX, SVTR-Tiny/CRNN bilingual English + Devanagari + STEM reader)
  3. Deterministic Geometric Primitive + Cubic Bezier + Interactive SVG Compiler
"""

from .primitives import (
    douglas_peucker,
    snap_orthogonal_polyline,
    fit_taubin_circle,
    fit_ellipse_algebraic,
    fit_schneider_cubic_bezier,
    classify_contour_primitive,
    associate_callouts_to_labels,
)
from .svg_builder import build_interactive_svg, SVGTheme
from .vectorizer import RekhaVectorizer, VectorizerConfig

__version__ = "1.0.0"
__all__ = [
    "RekhaVectorizer",
    "VectorizerConfig",
    "SVGTheme",
    "build_interactive_svg",
    "douglas_peucker",
    "snap_orthogonal_polyline",
    "fit_taubin_circle",
    "fit_ellipse_algebraic",
    "fit_schneider_cubic_bezier",
    "classify_contour_primitive",
    "associate_callouts_to_labels",
]
