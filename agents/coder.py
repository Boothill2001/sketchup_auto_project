"""
Agent 5 — Ruby Coder (upgraded)
Phase 5: Generate LOD 300 SketchUp Ruby script from 3D-mapped members.

For each member with start_point + end_point + rotation_degrees:
  1. Compute extrusion vector
  2. Build cross-section face on a plane perpendicular to the vector
  3. Rotate face by rotation_degrees
  4. PushPull by member length
  5. Tag with IFC attributes + correct layer

Unmapped members (confidence="unmapped") go to layer LOD300_UNMAPPED_NEEDS_REVIEW.
"""

import json
import math
import re
from pathlib import Path
from rich import print as rprint

from config import MAPPED_OUTPUT_FILE, CODER_OUTPUT_FILE, SPATIAL_OUTPUT_FILE
from core.llm_wrapper import call_llm, call_llm_with_feedback


# ============================================================================
# STEEL SECTION LOOKUP TABLE
# Source: AS/NZS 3679.1, AS/NZS 1163, TCVN 7571, JIS G3192, Vina-One catalogue
# ============================================================================
STEEL_SECTIONS: dict[str, dict] = {
    # ── UB — Universal Beam (Australian old format) ──────────────────────────
    "UB36b": {"type": "UB", "d": 359, "bf": 172, "tf": 13.0, "tw": 8.0},
    "UB36a": {"type": "UB", "d": 356, "bf": 171, "tf": 11.5, "tw": 7.0},
    "UB30a": {"type": "UB", "d": 303, "bf": 165, "tf":  9.9, "tw": 6.1},
    "UB25b": {"type": "UB", "d": 256, "bf": 146, "tf": 10.9, "tw": 6.4},
    "UB25a": {"type": "UB", "d": 254, "bf": 146, "tf":  8.6, "tw": 5.8},
    "UB20b": {"type": "UB", "d": 206, "bf": 133, "tf":  9.6, "tw": 6.2},
    "UB20a": {"type": "UB", "d": 203, "bf": 133, "tf":  7.8, "tw": 5.8},
    "UB15a": {"type": "UB", "d": 155, "bf": 102, "tf":  7.0, "tw": 4.5},
    # ── UB — Universal Beam (metric) ─────────────────────────────────────────
    "360UB56.7": {"type": "UB", "d": 359, "bf": 172, "tf": 13.0, "tw": 8.0},
    "310UB40.4": {"type": "UB", "d": 304, "bf": 165, "tf":  8.7, "tw": 6.1},
    "250UB37.3": {"type": "UB", "d": 256, "bf": 146, "tf": 10.9, "tw": 6.4},
    "200UB29.8": {"type": "UB", "d": 207, "bf": 134, "tf":  9.6, "tw": 6.3},
    "150UB18.0": {"type": "UB", "d": 155, "bf": 102, "tf":  7.0, "tw": 4.5},

    # ── UC — Universal Column (Australian old + metric) ──────────────────────
    "UC155d": {"type": "UC", "d": 158, "bf": 153, "tf": 10.0, "tw": 6.6},
    "UC155c": {"type": "UC", "d": 152, "bf": 152, "tf":  8.0, "tw": 5.0},
    "UC100a": {"type": "UC", "d":  97, "bf":  99, "tf":  7.8, "tw": 4.6},
    "150UC30.0": {"type": "UC", "d": 158, "bf": 153, "tf": 10.0, "tw": 6.6},
    "100UC14.8": {"type": "UC", "d":  97, "bf":  99, "tf":  7.8, "tw": 4.6},

    # ── PFC — Parallel Flange Channel ────────────────────────────────────────
    "CH35a":     {"type": "PFC", "d": 380, "bf": 100, "tf": 17.5, "tw": 10.0},
    "CH30a":     {"type": "PFC", "d": 310, "bf":  86, "tf": 14.8, "tw":  9.1},
    "CH13c":     {"type": "PFC", "d": 125, "bf":  65, "tf":  9.5, "tw":  6.5},
    "380PFC":    {"type": "PFC", "d": 380, "bf": 100, "tf": 17.5, "tw": 10.0},
    "310PFC":    {"type": "PFC", "d": 310, "bf":  86, "tf": 14.8, "tw":  9.1},
    "125PFC":    {"type": "PFC", "d": 125, "bf":  65, "tf":  9.5, "tw":  6.5},

    # ── RHS / SHS — Hollow Sections ──────────────────────────────────────────
    "SH30b":         {"type": "RHS", "d": 300, "b": 100, "t": 6.0},
    "SH20a":         {"type": "RHS", "d": 200, "b":  75, "t": 4.0},
    "SH08d":         {"type": "SHS", "d":  89, "b":  89, "t": 3.5},
    "SH07g":         {"type": "SHS", "d":  75, "b":  75, "t": 3.0},
    "150x50x5RHS":   {"type": "RHS", "d": 150, "b":  50, "t": 5.0},
    "100x50x4RHS":   {"type": "RHS", "d": 100, "b":  50, "t": 4.0},

    # ── FB — Flat Bar ────────────────────────────────────────────────────────
    "FB": {"type": "FB", "d": None, "b": None, "t": None},

    # ── AS4100 UB series (nominal depth prefix, no mass suffix) ─────────────
    "150UB": {"type": "UB", "d": 155, "bf":  75, "tf":  7.0, "tw": 4.5},
    "180UB": {"type": "UB", "d": 173, "bf":  90, "tf":  8.0, "tw": 5.5},
    "200UB": {"type": "UB", "d": 203, "bf": 100, "tf":  8.0, "tw": 5.0},
    "250UB": {"type": "UB", "d": 256, "bf": 146, "tf": 10.9, "tw": 6.4},
    "310UB": {"type": "UB", "d": 302, "bf": 133, "tf":  9.6, "tw": 6.1},
    "360UB": {"type": "UB", "d": 359, "bf": 148, "tf": 13.0, "tw": 8.0},
    "410UB": {"type": "UB", "d": 403, "bf": 178, "tf": 12.8, "tw": 8.0},
    "460UB": {"type": "UB", "d": 457, "bf": 191, "tf": 14.5, "tw": 9.0},
    "530UB": {"type": "UB", "d": 533, "bf": 209, "tf": 15.6, "tw": 10.2},

    # ── AS4100 PFC series ────────────────────────────────────────────────────
    "75PFC":  {"type": "PFC", "d":  75, "bf":  40, "tf":  6.1, "tw": 4.5},
    "100PFC": {"type": "PFC", "d": 100, "bf":  50, "tf":  6.7, "tw": 5.0},
    "150PFC": {"type": "PFC", "d": 150, "bf":  75, "tf":  9.5, "tw": 6.0},
    "200PFC": {"type": "PFC", "d": 200, "bf":  75, "tf": 12.5, "tw": 6.5},
    "250PFC": {"type": "PFC", "d": 250, "bf":  90, "tf": 13.0, "tw": 7.0},
    "300PFC": {"type": "PFC", "d": 300, "bf":  90, "tf": 16.5, "tw": 8.0},

    # ── AS4100 SHS series ────────────────────────────────────────────────────
    "50SHS4":  {"type": "SHS", "d":  50, "b":  50, "t": 4.0},
    "65SHS4":  {"type": "SHS", "d":  65, "b":  65, "t": 4.0},
    "75SHS4":  {"type": "SHS", "d":  75, "b":  75, "t": 4.0},
    "89SHS4":  {"type": "SHS", "d":  89, "b":  89, "t": 4.0},
    "100SHS4": {"type": "SHS", "d": 100, "b": 100, "t": 4.0},
    "100SHS5": {"type": "SHS", "d": 100, "b": 100, "t": 5.0},
    "150SHS5": {"type": "SHS", "d": 150, "b": 150, "t": 5.0},
    "150SHS6": {"type": "SHS", "d": 150, "b": 150, "t": 6.0},
    "200SHS5": {"type": "SHS", "d": 200, "b": 200, "t": 5.0},
    "200SHS6": {"type": "SHS", "d": 200, "b": 200, "t": 6.0},
    "250SHS6": {"type": "SHS", "d": 250, "b": 250, "t": 6.0},

    # ── AS4100 RHS series ────────────────────────────────────────────────────
    "75x50RHS4":   {"type": "RHS", "d":  75, "b":  50, "t": 4.0},
    "100x50RHS4":  {"type": "RHS", "d": 100, "b":  50, "t": 4.0},
    "125x75RHS4":  {"type": "RHS", "d": 125, "b":  75, "t": 4.0},
    "150x50RHS5":  {"type": "RHS", "d": 150, "b":  50, "t": 5.0},
    "150x100RHS5": {"type": "RHS", "d": 150, "b": 100, "t": 5.0},
    "200x100RHS5": {"type": "RHS", "d": 200, "b": 100, "t": 5.0},

    # ════════════════════════════════════════════════════════════════════════
    # TCVN / JIS / VINA-ONE — Vietnamese & Asian structural steel sections
    # ════════════════════════════════════════════════════════════════════════

    # ── I-shape (depth x flange-width x web x flange-thk) ────────────────────
    # Vina-One / Posco SS400 I-beams — common in VN structural drawings
    "I200x100x5.5x8":   {"type": "UB", "d": 200, "bf": 100, "tf": 8.0, "tw": 5.5},
    "I250x125x6x9":     {"type": "UB", "d": 250, "bf": 125, "tf": 9.0, "tw": 6.0},
    "I300x150x6.5x9":   {"type": "UB", "d": 300, "bf": 150, "tf": 9.0, "tw": 6.5},
    "I350x175x7x11":    {"type": "UB", "d": 350, "bf": 175, "tf": 11.0, "tw": 7.0},
    "I400x200x8x13":    {"type": "UB", "d": 400, "bf": 200, "tf": 13.0, "tw": 8.0},
    "I450x200x9x14":    {"type": "UB", "d": 450, "bf": 200, "tf": 14.0, "tw": 9.0},
    "I500x200x10x16":   {"type": "UB", "d": 500, "bf": 200, "tf": 16.0, "tw": 10.0},
    "I600x200x11x17":   {"type": "UB", "d": 600, "bf": 200, "tf": 17.0, "tw": 11.0},

    # ── H-shape (depth x width x web x flange-thk) ───────────────────────────
    # Wide-flange beams/columns widely used in VN
    "H100x100x6x8":     {"type": "UC", "d": 100, "bf": 100, "tf": 8.0, "tw": 6.0},
    "H150x150x7x10":    {"type": "UC", "d": 150, "bf": 150, "tf": 10.0, "tw": 7.0},
    "H200x200x8x12":    {"type": "UC", "d": 200, "bf": 200, "tf": 12.0, "tw": 8.0},
    "H250x250x9x14":    {"type": "UC", "d": 250, "bf": 250, "tf": 14.0, "tw": 9.0},
    "H300x300x10x15":   {"type": "UC", "d": 300, "bf": 300, "tf": 15.0, "tw": 10.0},
    "H350x350x12x19":   {"type": "UC", "d": 350, "bf": 350, "tf": 19.0, "tw": 12.0},
    "H400x400x13x21":   {"type": "UC", "d": 400, "bf": 400, "tf": 21.0, "tw": 13.0},

    # ── C/U-channel (depth x flange-width x web x flange-thk) ────────────────
    "C100x50x5x7.5":    {"type": "PFC", "d": 100, "bf": 50, "tf": 7.5, "tw": 5.0},
    "C150x75x5.5x7.5":  {"type": "PFC", "d": 150, "bf": 75, "tf": 7.5, "tw": 5.5},
    "C200x80x6x9":      {"type": "PFC", "d": 200, "bf": 80, "tf": 9.0, "tw": 6.0},
    "C250x90x7x10":     {"type": "PFC", "d": 250, "bf": 90, "tf": 10.0, "tw": 7.0},
    "C300x90x9x11":     {"type": "PFC", "d": 300, "bf": 90, "tf": 11.0, "tw": 9.0},
    "U100x50x5x7.5":    {"type": "PFC", "d": 100, "bf": 50, "tf": 7.5, "tw": 5.0},
    "U150x75x5.5x7.5":  {"type": "PFC", "d": 150, "bf": 75, "tf": 7.5, "tw": 5.5},
    "U200x80x6x9":      {"type": "PFC", "d": 200, "bf": 80, "tf": 9.0, "tw": 6.0},

    # ── L-shape / V-shape (equal leg angle) ──────────────────────────────────
    "L50x50x5":    {"type": "angle", "d": 50,  "bf": 50,  "tf": 5.0, "tw": 5.0},
    "L75x75x6":    {"type": "angle", "d": 75,  "bf": 75,  "tf": 6.0, "tw": 6.0},
    "L100x100x8":  {"type": "angle", "d": 100, "bf": 100, "tf": 8.0, "tw": 8.0},
    "L120x120x10": {"type": "angle", "d": 120, "bf": 120, "tf": 10.0, "tw": 10.0},

    # ── Box/Rectangular hollow (width x depth x thickness) ───────────────────
    "RHS100x50x4":   {"type": "RHS", "d": 100, "b": 50,  "t": 4.0},
    "RHS150x100x5":  {"type": "RHS", "d": 150, "b": 100, "t": 5.0},
    "RHS200x100x6":  {"type": "RHS", "d": 200, "b": 100, "t": 6.0},
    "SHS50x50x3":    {"type": "SHS", "d":  50, "b":  50, "t": 3.0},
    "SHS100x100x5":  {"type": "SHS", "d": 100, "b": 100, "t": 5.0},
    "SHS150x150x6":  {"type": "SHS", "d": 150, "b": 150, "t": 6.0},
}


# ============================================================================
# SECTION PATTERN PARSER — handles Vietnamese/Asian naming conventions
#   I200x100x5.5x8 -> d=200, bf=100, tw=5.5, tf=8
#   H250x250x9x14  -> d=250, bf=250, tw=9, tf=14
#   C200x75x6x9    -> d=200, bf=75, tw=6, tf=9
#   L75x75x6       -> equal angle, leg=75, leg=75, t=6
#   RHS150x100x5   -> d=150, b=100, t=5
#   SHS100x100x5   -> d=100, b=100, t=5
#   200UC46        -> d=200, UC type (UB/UC old format without full dims)
#   310UB40.4      -> d=310, UB type
# ============================================================================

_SECTION_PATTERNS = [
    # I-shape: I<d>x<bf>x<tw>x<tf>
    (re.compile(r'^I\s*(\d+)\s*[xX]\s*(\d+)\s*[xX]\s*([\d.]+)\s*[xX]\s*([\d.]+)\s*$', re.IGNORECASE),
     lambda m: {"type": "UB", "d": int(m[1]), "bf": int(m[2]),
                "tw": float(m[3]), "tf": float(m[4])}),

    # H-shape: H<d>x<bf>x<tw>x<tf>
    (re.compile(r'^H\s*(\d+)\s*[xX]\s*(\d+)\s*[xX]\s*([\d.]+)\s*[xX]\s*([\d.]+)\s*$', re.IGNORECASE),
     lambda m: {"type": "UC", "d": int(m[1]), "bf": int(m[2]),
                "tw": float(m[3]), "tf": float(m[4])}),

    # C/U-channel: [CU]\s*<d>x<bf>x<tw>x<tf>
    (re.compile(r'^[CU]\s*(\d+)\s*[xX]\s*(\d+)\s*[xX]\s*([\d.]+)\s*[xX]\s*([\d.]+)\s*$', re.IGNORECASE),
     lambda m: {"type": "PFC", "d": int(m[1]), "bf": int(m[2]),
                "tw": float(m[3]), "tf": float(m[4])}),

    # L/V angle: [LV]\s*<d>x<bf>x<t>  (equal leg)
    (re.compile(r'^[LV]\s*(\d+)\s*[xX]\s*(\d+)\s*[xX]\s*([\d.]+)\s*$', re.IGNORECASE),
     lambda m: {"type": "angle", "d": int(m[1]), "bf": int(m[2]),
                "tw": float(m[3]), "tf": float(m[3])}),

    # RHS: RHS<d>x<b>x<t>  (rectangular hollow)
    (re.compile(r'^RHS\s*(\d+)\s*[xX]\s*(\d+)\s*[xX]\s*([\d.]+)\s*$', re.IGNORECASE),
     lambda m: {"type": "RHS", "d": int(m[1]), "b": int(m[2]), "t": float(m[3])}),

    # SHS: SHS<d>x<d>x<t>  (square hollow)
    (re.compile(r'^SHS\s*(\d+)\s*[xX]\s*\1\s*[xX]\s*([\d.]+)\s*$', re.IGNORECASE),
     lambda m: {"type": "SHS", "d": int(m[1]), "b": int(m[1]), "t": float(m[2])}),
    # SHS variant: SHS<d>x<t>
    (re.compile(r'^SHS\s*(\d+)\s*[xX]\s*([\d.]+)\s*$', re.IGNORECASE),
     lambda m: {"type": "SHS", "d": int(m[1]), "b": int(m[1]), "t": float(m[2])}),

    # UB/UC metric with mass: <depth>UB<mass> or <depth>UC<mass>
    (re.compile(r'^(\d+)\s*UB\s*([\d.]+)\s*$', re.IGNORECASE),
     lambda m: {"type": "UB", "d": int(m[1]), "bf": int(m[1]) // 2, "tf": 8.0, "tw": 5.0}),
    (re.compile(r'^(\d+)\s*UC\s*([\d.]+)\s*$', re.IGNORECASE),
     lambda m: {"type": "UC", "d": int(m[1]), "bf": int(m[1]) // 2, "tf": 8.0, "tw": 5.0}),
    # UB/UC short: <depth>UB or <depth>UC (no mass suffix)
    (re.compile(r'^(\d+)\s*UB\s*$', re.IGNORECASE),
     lambda m: {"type": "UB", "d": int(m[1]), "bf": int(m[1]) // 2, "tf": 8.0, "tw": 5.0}),
    (re.compile(r'^(\d+)\s*UC\s*$', re.IGNORECASE),
     lambda m: {"type": "UC", "d": int(m[1]), "bf": int(m[1]) // 2, "tf": 8.0, "tw": 5.0}),
    # PFC short: <depth>PFC
    (re.compile(r'^(\d+)\s*PFC\s*$', re.IGNORECASE),
     lambda m: {"type": "PFC", "d": int(m[1]), "bf": max(40, int(m[1]) // 3), "tf": 7.0, "tw": 5.0}),
    # SHS short: <depth>SHS<t> or <depth>SHS
    (re.compile(r'^(\d+)\s*SHS\s*([\d.]+)\s*$', re.IGNORECASE),
     lambda m: {"type": "SHS", "d": int(m[1]), "b": int(m[1]), "t": float(m[2])}),
    (re.compile(r'^(\d+)\s*SHS\s*$', re.IGNORECASE),
     lambda m: {"type": "SHS", "d": int(m[1]), "b": int(m[1]), "t": 5.0}),

    # UB/UC old format: UB<depth><suffix> or UC<depth><suffix>
    (re.compile(r'^UB\s*(\d+)\s*([a-zA-Z]*)\s*$', re.IGNORECASE),
     lambda m: {"type": "UB", "d": int(m[1]) * 10, "bf": int(m[1]) * 5, "tf": 8.0, "tw": 5.0}),
    (re.compile(r'^UC\s*(\d+)\s*([a-zA-Z]*)\s*$', re.IGNORECASE),
     lambda m: {"type": "UC", "d": int(m[1]) * 10, "bf": int(m[1]) * 5, "tf": 8.0, "tw": 5.0}),

    # CH old format: CH<num><suffix> (depth ≈ num*10)
    (re.compile(r'^CH\s*(\d+)\s*([a-zA-Z]*)\s*$', re.IGNORECASE),
     lambda m: {"type": "PFC", "d": int(m[1]) * 10, "bf": int(m[1]) * 2.5, "tf": 6.0, "tw": 4.0}),

    # Generic: just digits like "200UC46" pre-parsed but failed exact match
    # -> use approximate: d ≈ first number, infer rest
    (re.compile(r'^(\d+)\s*UB([\d.]+)$', re.IGNORECASE),
     lambda m: {"type": "UB", "d": int(m[1]), "bf": int(m[1]) * 2 // 3, "tf": 8.0, "tw": 5.0}),
]


def _parse_section_pattern(section_str: str) -> dict | None:
    """Try to parse a section string using common Vietnamese/Asian patterns."""
    if not section_str:
        return None
    key = section_str.strip().replace(" ", "")
    for pattern, resolver in _SECTION_PATTERNS:
        m = pattern.match(key)
        if m:
            return resolver(m)
    return None


def lookup_section(section_str: str) -> dict | None:
    """
    Look up a section designation in STEEL_SECTIONS.
    Falls back to pattern-based parsing for Vietnamese/Asian naming conventions.
    Returns the dims dict if found/parsed, else None.
    """
    if not section_str:
        return None

    key = section_str.strip()

    # 1. Exact match
    result = STEEL_SECTIONS.get(key)
    if result is not None:
        return result

    # 2. Strip trailing dots
    key2 = key.rstrip(".")
    result = STEEL_SECTIONS.get(key2)
    if result is not None:
        return result

    # 3. Try normalizing spaces (e.g., "I 200 x 100 x 5.5 x 8" -> "I200x100x5.5x8")
    normalized = key.replace(" ", "")
    result = STEEL_SECTIONS.get(normalized)
    if result is not None:
        return result

    # 4. Pattern-based parsing (Vietnamese/Asian conventions)
    result = _parse_section_pattern(key)
    if result is not None:
        return result

    # 5. Fallback: try normalized version with pattern parser
    result = _parse_section_pattern(normalized)
    if result is not None:
        return result

    return None


# ============================================================================
# TEMPLATE-BASED RUBY GENERATOR — no LLM needed for standard geometry
# ============================================================================

def _estimate_depth(section_str: str) -> int:
    """Extract first integer from a section string as depth in mm."""
    m = re.search(r'(\d+)', section_str or "")
    return int(m.group(1)) if m else 100


def _section_rect(dims: dict, mtype: str, section_str: str) -> tuple[list, str]:
    """
    Return (pts_2d, log) for a solid rectangular cross-section.
    pts_2d is a list of (x, y) tuples in mm, centred on origin.
    All members use solid rect for LOD300 (no hollow shell needed).
    """
    sec_type = dims.get("type", "")

    if sec_type in ("SHS", "RHS"):
        b = dims.get("b", dims.get("d", 100))
        d = dims.get("d", 100)
        half_b, half_d = b / 2, d / 2
        pts = [(-half_b, -half_d), (half_b, -half_d), (half_b, half_d), (-half_b, half_d)]
        log = f"{sec_type} {b}×{d}mm"

    elif sec_type in ("UB", "UC"):
        d  = dims.get("d", 200)
        bf = dims.get("bf", max(100, d // 2))
        pts = [(-bf / 2, 0), (bf / 2, 0), (bf / 2, d), (-bf / 2, d)]
        log = f"{sec_type} solid rect {bf}×{d}mm"

    elif sec_type == "PFC":
        d  = dims.get("d", 150)
        bf = dims.get("bf", 75)
        pts = [(0, 0), (bf, 0), (bf, d), (0, d)]
        log = f"PFC {bf}×{d}mm"

    elif sec_type == "angle":
        d  = dims.get("d", 75)
        bf = dims.get("bf", d)
        pts = [(0, 0), (bf, 0), (bf, d), (0, d)]
        log = f"angle {bf}×{d}mm"

    elif sec_type == "FB":
        width = dims.get("b") or 100
        thk   = dims.get("t") or 10
        pts = [(0, 0), (width, 0), (width, thk), (0, thk)]
        log = f"FB {width}×{thk}mm"

    else:
        depth = _estimate_depth(section_str)
        width = depth if mtype in ("column", "strut") else max(50, depth // 2)
        pts = [(-width / 2, 0), (width / 2, 0), (width / 2, depth), (-width / 2, depth)]
        log = f"estimated {width}×{depth}mm (section not in library)"

    return pts, log


def _pts_to_ruby(pts: list) -> str:
    return "[" + ", ".join(f"[{x:g}, {y:g}]" for x, y in pts) + "]"


def _layer_for(mtype: str, conf: str, sz: float, ez: float) -> str:
    if conf == "unmapped" or sz == -9999 or ez == -9999:
        return "LOD300_UNMAPPED_NEEDS_REVIEW"
    mt = mtype.lower()
    if mt == "column":
        return "STR-Columns"
    if mt in ("beam", "purlin", "rafter", "girder"):
        return "STR-Beams"
    if mt == "brace":
        return "STR-Braces"
    if mt in ("plate", "flatbar", "gusset", "baseplat"):
        return "STR-Plates"
    return "STR-Beams"


# ── Spatial helpers (levels + grids loaded once per build_ruby_script call) ──
_spatial_cache: tuple | None = None


def _load_spatial() -> tuple[list, list, list]:
    """Return (sorted_z_levels, sorted_x_grids, sorted_y_grids) in mm. Cached per run."""
    global _spatial_cache
    if _spatial_cache is not None:
        return _spatial_cache
    try:
        data = json.loads(Path(SPATIAL_OUTPUT_FILE).read_text(encoding="utf-8"))
        levels = sorted(lv["z_mm"] for lv in data.get("levels", []) if "z_mm" in lv)
        gx = sorted(g["x_mm"] for g in data.get("grids_x", []) if "x_mm" in g)
        gy = sorted(g["y_mm"] for g in data.get("grids_y", []) if "y_mm" in g)
        _spatial_cache = (levels, gx, gy)
    except Exception:
        _spatial_cache = ([0, 3500, 7000, 10500, 13500], [], [])
    return _spatial_cache


def _next_level_z(sz: float) -> float:
    """Return z_mm of the next floor level strictly above sz, or sz+3500 if none."""
    levels, _, _ = _load_spatial()
    for z in levels:
        if z > sz + 1:
            return float(z)
    return float(sz) + 3500.0


def _beam_extension(sx: float, sy: float) -> tuple[float, float]:
    """
    For a zero-length beam/other at (sx, sy), return (ex, ey) by extending to the
    nearest adjacent grid line — whichever gap is smallest in X or Y direction.
    Falls back to (sx+4000, sy) if no grid data available.
    """
    _, gx, gy = _load_spatial()
    dx_cands = [abs(x - sx) for x in gx if abs(x - sx) > 1]
    dy_cands = [abs(y - sy) for y in gy if abs(y - sy) > 1]
    dx_min = min(dx_cands, default=4000.0)
    dy_min = min(dy_cands, default=None)

    if dy_min is None or dx_min <= dy_min:
        # next grid in X direction (East-West beam)
        x_targets = [x for x in gx if x > sx + 1]
        ex = float(x_targets[0]) if x_targets else sx + dx_min
        return ex, sy
    else:
        # next grid in Y direction (North-South beam)
        y_targets = [y for y in gy if y > sy + 1]
        ey = float(y_targets[0]) if y_targets else sy + dy_min
        return sx, ey


def _build_member_ruby(member: dict) -> str:
    """Generate Ruby code for one structural member — pure template, no LLM call."""
    mark    = (member.get("mark")    or "M?").replace('"', '\\"')
    section = (member.get("section") or "").replace('"', '\\"')
    mtype   = (member.get("type")    or "beam").lower()
    conf    = member.get("confidence", "high")
    dims    = member.get("_section_dims") or {}

    sp = member.get("start_point") or {}
    ep = member.get("end_point")   or {}
    sx = float(sp.get("x", 0)); sy = float(sp.get("y", 0)); sz = float(sp.get("z", 0))
    ex = float(ep.get("x", 0)); ey = float(ep.get("y", 0)); ez = float(ep.get("z", 0))

    layer = _layer_for(mtype, conf, sz, ez)
    if layer == "LOD300_UNMAPPED_NEEDS_REVIEW":
        sx, sy, sz, ex, ey, ez = 0.0, 0.0, 0.0, 0.0, 0.0, 3000.0

    # ── Fix zero-length members ───────────────────────────────────────────────
    sec_type = dims.get("type", "")
    _is_col_section = sec_type in ("SHS",) or (
        sec_type in ("RHS",) and abs(dims.get("d", 0) - dims.get("b", 1)) < 2
    )
    _len3d = math.sqrt((ex - sx) ** 2 + (ey - sy) ** 2 + (ez - sz) ** 2)

    if mtype == "column":
        if abs(ez - sz) < 1:
            ez = _next_level_z(sz)
            rprint(f"  Coder fix: column {mark!r} no height -> ez={ez:.0f}mm")
    elif _len3d < 1:
        if _is_col_section:
            # Square hollow section -> treat as column
            ez = _next_level_z(sz)
            rprint(f"  Coder fix: {mtype} {mark!r} SHS->column -> ez={ez:.0f}mm")
        else:
            # Beam / channel / other — extend to nearest adjacent grid
            ex, ey = _beam_extension(sx, sy)
            rprint(f"  Coder fix: {mtype} {mark!r} zero-length -> beam to ({ex:.0f},{ey:.0f})")

    # ── Snap horizontal beams to cardinal direction (no diagonal in plan) ─────
    if abs(ez - sz) < 1 and mtype != "column":
        dx, dy = ex - sx, ey - sy
        if abs(dx) > 1 and abs(dy) > 1:
            if abs(dx) >= abs(dy):
                ey = sy  # East-West dominant
            else:
                ex = sx  # North-South dominant
            rprint(f"  Coder fix: {mark!r} diagonal snapped to cardinal axis")

    pts, log_msg = _section_rect(dims, mtype, section)
    if not dims:
        rprint(f"  Section {section!r} not in library — {log_msg}")
    pts_ruby = _pts_to_ruby(pts)

    return (
        f"# ---- {mark} | {section} | {mtype} ----\n"
        f"begin\n"
        f"  _sp  = Geom::Point3d.new({sx:g}.mm, {sy:g}.mm, {sz:g}.mm)\n"
        f"  _ep  = Geom::Point3d.new({ex:g}.mm, {ey:g}.mm, {ez:g}.mm)\n"
        f"  _vec = _sp.vector_to(_ep)\n"
        f"  _len = _sp.distance(_ep)\n"
        f"  if _len < 1.mm || !_vec.valid?\n"
        f"    puts \"SKIP {mark}: zero-length (check mapper coords)\"\n"
        f"  else\n"
        f"    _grp = ents.add_group\n"
        f"    _ge  = _grp.entities\n"
        f"    _t   = Geom::Transformation.new(_sp, _vec)\n"
        f"    _raw = {pts_ruby}\n"
        f"    _face = _ge.add_face(_raw.map {{ |p| _t * Geom::Point3d.new(p[0].mm, p[1].mm, 0) }})\n"
        f"    _face.pushpull(_len) if _face\n"
        f"    _grp.layer = get_or_create_layer(layers, \"{layer}\")\n"
        f"    _grp.material = _steel_mat\n"
        f"    _grp.set_attribute(\"IFC\", \"Mark\",    \"{mark}\")\n"
        f"    _grp.set_attribute(\"IFC\", \"Section\", \"{section}\")\n"
        f"    _grp.set_attribute(\"IFC\", \"Type\",    \"{mtype}\")\n"
        f"    _grp.name = \"{mark}\"\n"
        f"  end\n"
        f"rescue => e\n"
        f"  puts \"SKIP {mark}: #{{e.message}}\"\n"
        f"end\n"
    )


RUBY_HEADER = """\
# =============================================================================
# LOD 300 Structural Model — Auto-Generated by PDF-to-SketchUp Pipeline
# Model: gemini-2.5-flash  |  DO NOT EDIT — regenerate from mapped_members.json
# =============================================================================

model  = Sketchup.active_model
model.name        = "LOD300 Structural Model"
model.description = "Generated by PDF-to-SketchUp pipeline"
ents   = model.active_entities
layers = model.layers
model.start_operation('LOD300 Structural Import', true)

def get_or_create_layer(layers, name)
  layers[name] || layers.add(name)
end

# ── Layers ────────────────────────────────────────────────────────────────────
["STR-Columns", "STR-Beams", "STR-Braces", "STR-Plates",
 "STR-Foundations", "STR-Slabs", "LOD300_UNMAPPED_NEEDS_REVIEW"].each do |lname|
  get_or_create_layer(layers, lname)
end

# ── Materials ─────────────────────────────────────────────────────────────────
_mats = model.materials
_steel_mat = _mats["Structural_Steel"] || _mats.add("Structural_Steel")
_steel_mat.color = Sketchup::Color.new(160, 160, 175)
_concrete_mat = _mats["Concrete"] || _mats.add("Concrete")
_concrete_mat.color = Sketchup::Color.new(180, 175, 165)

"""

RUBY_FOOTER = """\

model.commit_operation
puts "=== LOD 300 Import Complete ==="

# Auto-save model as .skp alongside this script
_skp_path = File.expand_path(File.join(File.dirname(__FILE__), 'lod300_model.skp'))
_saved = Sketchup.active_model.save(_skp_path)
puts _saved ? "Model saved: #{_skp_path}" : "Save FAILED: #{_skp_path}"
"""


CODER_PROMPT = """You are an Expert SketchUp Ruby API Developer generating LOD 300 structural elements.

Steel section lookup is pre-loaded — when you see designations like UB36b, UC155d, I200x100x5.5x8, H250x250x9x14, use these exact dimensions:
{section_lookup_hint}

Member data (JSON):
{member_json}

Write Ruby code that PRECISELY places this member in 3D space.

STRICT RULES — follow every one or the script will fail:
1. All dimensions in SketchUp inches. Convert mm with `.mm` suffix (e.g. `6000.mm`).
2. Compute start and end:
   start_pt = Geom::Point3d.new({sx}.mm, {sy}.mm, {sz}.mm)
   end_pt   = Geom::Point3d.new({ex}.mm, {ey}.mm, {ez}.mm)
3. Extrusion vector:
   vec = start_pt.vector_to(end_pt)
   length = start_pt.distance(end_pt)
4. Create a local group:
   grp = ents.add_group
   g_ents = grp.entities
5. Build the 2D cross-section face AT start_pt on a plane NORMAL to `vec`:
   - I/UB/UC sections: draw top flange, web, bottom flange as one closed polygon.
     Use actual flange width (bf), web height (d - 2*tf), flange thickness (tf), web thickness (tw).
   - RHS/SHS: outer rect minus inner rect (draw solid, SketchUp will show as box).
   - CHS: 24-sided polygon approximation using Math::PI and radius.
   - PL/FB: simple rectangle (width x thickness).
   Use `Geom::Transformation.new` with a local coordinate system aligned to `vec`.
   Transform face points into world space before drawing.
6. Get the face: `face = g_ents.grep(Sketchup::Face).first`
7. Extrude: `face.pushpull(length)`
8. Layer assignment:
   layer_name = {layer_name}
   grp.layer = get_or_create_layer(layers, layer_name)
9. IFC attributes:
   grp.set_attribute("IFC", "Mark",    "{mark}")
   grp.set_attribute("IFC", "Section", "{section}")
   grp.set_attribute("IFC", "Type",    "{type}")
10. Name the group: `grp.name = "{mark}"`

Output ONLY valid Ruby code. No markdown, no prose, no method definitions outside the snippet."""


BATCH_CODER_PROMPT = """You are an Expert SketchUp Ruby API Developer.
Generate SketchUp Ruby API code to place {count} structural members in 3D space.

IMPORTANT — OUTPUT FORMAT:
- Output ONLY executable Ruby code. No markdown, no prose, no explanations.
- Do NOT define any constants or variables for the input data.
- Do NOT echo back or embed the JSON data in your output.
- For EACH member, output Ruby code that calls SketchUp API methods directly.

OUTER SCOPE variables already defined (DO NOT redefine them):
  model, ents, layers, get_or_create_layer

MEMBER DATA:
{members_json}

FOR EACH MEMBER, generate Ruby code in this EXACT pattern:

# ---- <MARK> | <SECTION> | <TYPE> ----
begin
  _sp  = Geom::Point3d.new(<start_x>.mm, <start_y>.mm, <start_z>.mm)
  _ep  = Geom::Point3d.new(<end_x>.mm,   <end_y>.mm,   <end_z>.mm)
  _vec = _sp.vector_to(_ep)
  _len = _sp.distance(_ep)
  _grp = ents.add_group
  _ge  = _grp.entities
  _t   = Geom::Transformation.new(_sp, _vec)
  # Cross-section at origin, extruded along _vec
  # I/UB/UC: bf=<bf>mm, d=<d>mm, tf=<tf>mm, tw=<tw>mm
  _pts = [<2D cross-section points in LOCAL coords, mm units>]
  _face = _ge.add_face(_pts.map { |p| _t * Geom::Point3d.new(p[0].mm, p[1].mm, 0) })
  _face.pushpull(_len)
  _grp.layer = get_or_create_layer(layers, "<LAYER_NAME>")
  _grp.set_attribute("IFC", "Mark",    "<MARK>")
  _grp.set_attribute("IFC", "Section", "<SECTION>")
  _grp.set_attribute("IFC", "Type",    "<TYPE>")
  _grp.name = "<MARK>"
rescue => e
  puts "SKIP <MARK>: #{e.message}"
end

CROSS-SECTION RULES:
- I/UB/UC section (bf, d, tf, tw known): draw H-shape polygon:
  half_bf = bf/2; half_tw = tw/2; web_h = d - 2*tf
  points (clockwise from bottom-left of bottom flange):
  [-half_bf,0], [half_bf,0], [half_bf,tf], [half_tw,tf],
  [half_tw,tf+web_h], [half_bf,tf+web_h], [half_bf,d], [-half_bf,d],
  [-half_bf,tf+web_h], [-half_tw,tf+web_h], [-half_tw,tf], [-half_bf,tf]

- PFC/Channel (bf, d, tf, tw known): draw C-shape polygon:
  points: [0,0],[bf,0],[bf,tf],[tw,tf],[tw,d-tf],[bf,d-tf],[bf,d],[0,d]

- RHS/SHS (b, d, t known): solid rectangle (b x d):
  points: [0,0],[b,0],[b,d],[0,d]

- CHS (outer_radius r, thickness t): 24-sided polygon approximation:
  (1..24).map {{ |i| a = 2*Math::PI*i/24; [r*Math::cos(a), r*Math::sin(a)] }}

- PL/FB (width w, thickness t): rectangle [0,0],[w,0],[w,t],[0,t]

- ANGLE (V/L shape, leg1 bf, leg2 d, thickness t):
  points: [0,0],[bf,0],[bf,t],[t,t],[t,d],[0,d]

- UNKNOWN section: use rectangle 100mm x 100mm as placeholder

LAYER NAME RULES:
- confidence == "unmapped" OR start_z == -9999 OR end_z == -9999:
    layer = "LOD300_UNMAPPED_NEEDS_REVIEW"
    AND override start_pt = (0,0,0), end_pt = (0,0,3000) so it doesn't crash
- type == "beam"   -> "LOD300_BEAM"
- type == "column" -> "LOD300_COLUMN"
- type == "slab"   -> "LOD300_SLAB"
- type == "brace"  -> "LOD300_BRACE"
- type == "wall"   -> "LOD300_WALL"
- other            -> "LOD300_OTHER"

Generate code for ALL {count} members. Output ONLY Ruby code."""


def _fill_prompt(member: dict) -> str:
    sp      = member.get("start_point") or {"x": 0, "y": 0, "z": 0}
    ep      = member.get("end_point")   or {"x": 0, "y": 0, "z": 3000}
    mark    = member.get("mark")    or "M?"
    section = member.get("section") or "UB200x100x20"
    mtype   = member.get("type")    or "beam"
    conf    = member.get("confidence") or "high"
    layer_name = (
        "LOD300_UNMAPPED_NEEDS_REVIEW" if conf == "unmapped"
        else f"LOD300_{mtype.upper()}"
    )

    sx = str(sp.get("x") or 0); sy = str(sp.get("y") or 0); sz = str(sp.get("z") or 0)
    ex = str(ep.get("x") or 0); ey = str(ep.get("y") or 0); ez = str(ep.get("z") or 3000)

    dims = member.get("_section_dims")
    hint = json.dumps(dims, indent=2) if dims else "(not in lookup table — infer from section string)"

    return (
        CODER_PROMPT
        .replace("{member_json}", json.dumps(member, indent=2))
        .replace("{section_lookup_hint}", hint)
        .replace("{sx}", sx).replace("{sy}", sy).replace("{sz}", sz)
        .replace("{ex}", ex).replace("{ey}", ey).replace("{ez}", ez)
        .replace("{layer_name}", f'"{layer_name}"')
        .replace("{mark}", mark)
        .replace("{section}", section)
        .replace("{type}", mtype)
    )


def _fill_batch_prompt(batch: list[dict]) -> str:
    return (
        BATCH_CODER_PROMPT
        .replace("{count}", str(len(batch)))
        .replace("{members_json}", json.dumps(batch, indent=2))
    )


def generate_ruby_for_member(member: dict, error_feedback: str | None = None, previous_ruby: str | None = None) -> str:
    """Single-member call — used only for error-feedback retries."""
    if error_feedback:
        base = _fill_batch_prompt([member])
        prompt = (
            base
            + "\n\n--- AUDITOR FEEDBACK (fix these issues) ---\n"
            + error_feedback
            + "\n\nOutput ONLY corrected Ruby code using the begin/rescue pattern shown above."
        )
        raw = call_llm(prompt)
    else:
        raw = _generate_ruby_for_batch([member])
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 1)[1]
        if raw.startswith("ruby"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    return raw


def _generate_ruby_for_batch(batch: list[dict]) -> str:
    """Send one batch of members to the LLM, return raw Ruby text for the whole batch."""
    prompt = _fill_batch_prompt(batch)
    raw = call_llm(prompt)
    # Strip accidental markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 1)[1]
        if raw.startswith("ruby"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    return raw


def build_ruby_script(mapped_members: list[dict], error_feedback_map: dict | None = None) -> str:
    """
    Build the full Ruby script.
    Normal members: template-generated in Python (no LLM call, deterministic).
    error_feedback retries: single LLM call per member to apply auditor corrections.
    """
    global _spatial_cache
    _spatial_cache = None  # reload spatial data fresh each pipeline run

    error_feedback_map = error_feedback_map or {}
    total = len(mapped_members)

    # ── Section lookup: inject _section_dims before generation ───────────────
    hit_count = 0
    for member in mapped_members:
        dims = lookup_section(member.get("section", ""))
        if dims is not None:
            member["_section_dims"] = dims
            hit_count += 1
    rprint(f"  Section lookup: {hit_count}/{total} members resolved from table (no API needed)")

    retry_members  = [m for m in mapped_members if error_feedback_map.get(m.get("mark", ""))]
    normal_members = [m for m in mapped_members if not error_feedback_map.get(m.get("mark", ""))]

    rprint(f"[bold blue]Coder:[/] {total} members -> "
           f"{len(normal_members)} template-generated + {len(retry_members)} LLM retry call(s)")

    # ---- Template-based generation for normal members (zero LLM calls) ----
    normal_parts: list[str] = []
    for member in normal_members:
        normal_parts.append(_build_member_ruby(member))

    # ---- LLM retries for auditor-flagged members ----
    retry_parts: list[tuple[str, dict, str]] = []
    for member in retry_members:
        mark = member.get("mark", "?")
        rprint(f"  Retry (LLM): {mark}")
        feedback = error_feedback_map[mark]
        ruby_block = generate_ruby_for_member(member, feedback, None)
        ruby_block = ruby_block.strip()
        if ruby_block.startswith("```"):
            ruby_block = ruby_block.split("```", 1)[1]
            if ruby_block.startswith("ruby"):
                ruby_block = ruby_block[4:]
            ruby_block = ruby_block.rsplit("```", 1)[0].strip()
        retry_parts.append((mark, member, ruby_block))

    # ---- Assemble final script ----
    blocks = [RUBY_HEADER]
    for part in normal_parts:
        blocks.append(part)
    for mark, member, ruby_block in retry_parts:
        blocks.append(
            f"\n# ---- {mark} | {member.get('section','')} | {member.get('type','')} ----\n"
            f"{ruby_block}\n"
        )
    blocks.append(RUBY_FOOTER)
    return "\n".join(blocks)


def save_ruby_script(script: str) -> None:
    Path(CODER_OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(CODER_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(script)
    rprint(f"\n[bold green]Coder complete.[/] Ruby script -> {CODER_OUTPUT_FILE}")


if __name__ == "__main__":
    with open(MAPPED_OUTPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    members = data.get("mapped_members", [])
    script = build_ruby_script(members)
    save_ruby_script(script)