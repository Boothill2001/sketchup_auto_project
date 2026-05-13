"""
Centralized PDF analysis context builder.
Every downstream agent calls `build_context_string()` to get a prompt-ready
block that describes the PDF convention detected in Phase 0.

This ELIMINATES hardcoded English/Australian assumptions from every agent.
"""

import json
from pathlib import Path
from config import OUTPUT_JSON_DIR


def load_analysis_dict() -> dict:
    """Load pdf_analysis.json, return dict with safe defaults."""
    path = Path(OUTPUT_JSON_DIR) / "pdf_analysis.json"
    if not path.exists():
        return _defaults()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _defaults()


def _defaults() -> dict:
    return {
        "steel_standard": "australian",
        "unit_system": "mm",
        "grid_convention": "letter_x_number_y",
        "grid_labels_x": ["A","B","C","D","E","F","G","H"],
        "grid_labels_y": ["1","2","3","4","5","6","7","8","9"],
        "grid_count_x": 8,
        "grid_count_y": 9,
        "language": "english",
        "building_type": "unknown",
        "pdf_type": "structural_only",
        "floor_count": 2,
        "drawing_scale": "unknown",
    }


STEEL_STANDARD_DESCRIPTIONS = {
    "australian": "Australian AS/NZS 3679 — shapes: UB, UC, PFC, CH, RHS, SHS, EA (angles), UA (unequal angles), TFB (tee), WPB (welded plate beam). Sections in mm.",
    "aisc": "American AISC 360 — shapes: W, S, HP, C, L, HSS round/rect, WT, MT, ST, Pipe. Sections in inches or mm.",
    "eurocode": "European EN 1993 — shapes: HEA, HEB, HEM, IPE, IPN, UPE, UPN, CHS, RHS, L. Sections in mm.",
    "jis": "Japanese JIS G 3192 — shapes: H (wide), B (narrow), C (channel), L (angle). Sections in mm.",
    "chinese_gb": "Chinese GB/T 11263 — shapes: HN (narrow), HW (wide), HM (medium), HT, HP. Metric mm.",
    "vietnamese": "Vietnamese TCVN — shapes: I, H, U, V, L (angles), C (channels), steel pipe (ống thép), thép hộp (box/hollow sections), thép tấm (plate). Columns may use Vietnamese headers: KÍ HIỆU (mark), QUY CÁCH/KÍCH THƯỚC (size/dimensions), CHIỀU DÀI (length), SỐ LƯỢNG/SL (quantity), VẬT LIỆU (material), THÉP (steel). Sections in mm.",
    "custom": "Custom/non-standard designations — pay careful attention to shapes used. Sections in mm unless otherwise noted.",
    "none": "No structural steel detected — likely RC or other material.",
}


GRID_CONVENTION_DESCRIPTIONS = {
    "letter_x_number_y": "Letters along X-axis (columns), numbers along Y-axis (rows). Example: grid intersection at (B, 3).",
    "number_x_number_y": "Numbers on both axes. Example: grid intersection at (2, 5).",
    "number_x_letter_y": "Numbers on X-axis, letters on Y-axis. Example: grid intersection at (3, C).",
    "axis_labels": "Explicit axis labels (e.g. 'Trục 1', 'Trục A', 'Axis 1').",
    "northing_easting": "Coordinate pair system (N123-E456).",
    "custom": "Non-standard grid labels — check labels_x/labels_y arrays for actual values.",
    "none_detected": "No grid system visible.",
}


LANGUAGE_HINTS = {
    "english": "Column headers are likely in English: MARK, SIZE, SECTION, QTY, LENGTH, MATERIAL, GRADE, WEIGHT.",
    "vietnamese": "Column headers may be in Vietnamese: KÍ HIỆU (mark), QUY CÁCH/KÍCH THƯỚC (size), SỐ LƯỢNG/SL (quantity), CHIỀU DÀI (length), VẬT LIỆU (material), CẤP ĐỘ BỀN (grade), KHỐI LƯỢNG (weight), DIỄN GIẢI/GHI CHÚ (notes). Grid labels may use 'Trục' prefix.",
    "chinese": "Column headers may be in Chinese: 编号 (mark), 规格 (specification), 数量 (quantity), 长度 (length), 材质 (material).",
    "japanese": "Column headers may be in Japanese: 記号 (mark), サイズ (size), 数量 (quantity), 長さ (length).",
    "korean": "Column headers may be in Korean: 기호 (mark), 규격 (spec), 수량 (quantity), 길이 (length).",
    "mixed": "Multiple languages present — check actual column headers.",
    "other": "Unknown language — analyze column headers visually.",
}


def build_context_string(analysis: dict = None) -> str:
    """
    Build a self-contained context block that every agent prepends to its
    existing prompt. This makes the agent ADAPT to the actual PDF convention
    instead of assuming Australian/English.
    """
    if analysis is None:
        analysis = load_analysis_dict()

    steel = analysis.get("steel_standard", "australian")
    units = analysis.get("unit_system", "mm")
    grid_conv = analysis.get("grid_convention", "letter_x_number_y")
    lang = analysis.get("language", "english")
    labels_x = analysis.get("grid_labels_x", [])
    labels_y = analysis.get("grid_labels_y", [])
    nx = analysis.get("grid_count_x", 8)
    ny = analysis.get("grid_count_y", 9)
    floors = analysis.get("floor_count", 2)
    scale = analysis.get("drawing_scale", "unknown")
    bldg = analysis.get("building_type", "unknown")
    pdf_type = analysis.get("pdf_type", "unknown")

    steel_desc = STEEL_STANDARD_DESCRIPTIONS.get(steel, STEEL_STANDARD_DESCRIPTIONS["australian"])
    grid_desc = GRID_CONVENTION_DESCRIPTIONS.get(grid_conv, GRID_CONVENTION_DESCRIPTIONS["letter_x_number_y"])
    lang_hint = LANGUAGE_HINTS.get(lang, LANGUAGE_HINTS["english"])

    block = f"""
╔══════════════════════════════════════════════════════════════╗
║ PDF ANALYSIS — ADAPT TO THIS CONVENTION (DO NOT ASSUME)      ║
╚══════════════════════════════════════════════════════════════╝

STEEL STANDARD: {steel}
  → {steel_desc}

UNIT SYSTEM: {units} (all dimensions in {units}, produce output in {units})

GRID CONVENTION: {grid_conv}
  → {grid_desc}
  → X-axis labels (columns): {labels_x}   (count: {nx})
  → Y-axis labels (rows):    {labels_y}   (count: {ny})

LANGUAGE: {lang}
  → {lang_hint}

FLOOR COUNT: {floors}
BUILDING TYPE: {bldg}
DRAWING SCALE: {scale}
PDF TYPE: {pdf_type}

CRITICAL: Use the EXACT grid labels, column headers, and section names found in
the PDF. DO NOT assume English/Australian/American conventions. Read what is
actually on the page. Adapt your parsing to the detected convention above.
"""
    return block.strip()


def build_schedule_context(analysis: dict = None) -> str:
    """Context specifically for schedule parsing agents."""
    base = build_context_string(analysis)
    if analysis is None:
        analysis = load_analysis_dict()
    lang = analysis.get("language", "english")
    steel = analysis.get("steel_standard", "australian")

    extra = """
SCHEDULE TABLE PARSING INSTRUCTIONS:
- Identify column headers by their ACTUAL text (not hardcoded English names).
- Vietnamese headers to look for: KÍ HIỆU, QUY CÁCH, KÍCH THƯỚC, CHIỀU DÀI, SỐ LƯỢNG, SL, VẬT LIỆU, THÉP, DIỄN GIẢI, GHI CHÚ, CẤP ĐỘ BỀN, KHỐI LƯỢNG.
- For each row, extract: mark (unique ID), size/section designation, quantity, length (if given), material/grade (if given).
- Section designations follow the detected steel standard (see above).
- If a row spans multiple lines (merged cells), combine them.
- Ignore header/footer rows, page titles, and summary rows.
"""
    return base + "\n" + extra.strip()


def build_plan_context(analysis: dict = None) -> str:
    """Context specifically for plan/spatial parsing agents."""
    base = build_context_string(analysis)
    if analysis is None:
        analysis = load_analysis_dict()
    labels_x = analysis.get("grid_labels_x", [])
    labels_y = analysis.get("grid_labels_y", [])

    extra = f"""
PLAN VIEW / GRID PARSING INSTRUCTIONS:
- Grid lines are labeled: X = {labels_x}, Y = {labels_y}
- Grid intersections should be noted as (X_label, Y_label), e.g. (B, 3)
- Column locations are typically at grid intersections.
- Beam spans between adjacent grid intersections, e.g. beam on grid B from 1 to 2 is "B/1-2".
- Dimensions are in the detected unit system (see above).
- Look for grid bubbles/circles with these exact labels — do NOT substitute English letters.
"""
    return base + "\n" + extra.strip()


def build_coder_context(analysis: dict = None) -> str:
    """Context specifically for the Ruby script coder."""
    base = build_context_string(analysis)
    if analysis is None:
        analysis = load_analysis_dict()
    steel = analysis.get("steel_standard", "australian")
    units = analysis.get("unit_system", "mm")

    extra = f"""
CODING INSTRUCTIONS:
- Use the EXACT member marks (section designations) as extracted from the schedule.
- All coordinates are in {units}. Convert to inches ONLY if the standard demands it.
- Steel standard: {steel} — use appropriate material properties for this standard.
- SketchUp Ruby API: use model.entities.add_edges, group entities by member mark.
- Follow LOD 300 requirements: members positioned correctly, correct section profiles,
  connections modeled as simple endpoints (not detailed bolts).
"""
    return base + "\n" + extra.strip()