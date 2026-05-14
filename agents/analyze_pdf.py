"""
Agent 0 — PDF Analysis (CRITICAL — runs BEFORE Scanner)
Phase 0: Analyze PDF convention, unit system, grid style, steel standard.
Output: data/output_json/pdf_analysis.json
This feeds all downstream agents so they adapt to ANY PDF standard.
"""

import json
from pathlib import Path
from rich import print as rprint

from config import OUTPUT_JSON_DIR
from core.llm_wrapper import call_llm_json
from core.pdf_utils import render_page_fast, extract_text_from_page, get_page_count


PDF_ANALYSIS_OUTPUT = str(Path(OUTPUT_JSON_DIR) / "pdf_analysis.json")

ANALYZE_PROMPT = """You are a Principal Structural Engineer analyzing an unknown structural/architectural PDF drawing set.
Your job is to DETECT the drawing convention, NOT to extract members. This analysis drives ALL downstream extraction.

Examine these pages from the PDF and determine:

1. **PDF_TYPE**: "structural_only" | "architectural_only" | "combined" | "unknown"

2. **TEXT_LAYER**: "digital" (text can be selected/copied) | "scanned" (image-only, needs OCR)

3. **STEEL_STANDARD** (if structural): 
   - "australian" (UB/UC/PFC/CH/RHS/SHS designations, AS/NZS 3679)
   - "aisc" (W/S/HP/C/L/HSS designations, AISC 360)
   - "eurocode" (HEA/HEB/HEM/IPE/UPN/CHS/RHS designations, EN 1993)
   - "jis" (H/B/C/L designations, JIS G 3192)
   - "chinese_gb" (HN/HW/HM designations, GB/T 11263)
   - "vietnamese" (TCVN steel or custom Vietnamese shop drawings)
   - "custom" (non-standard designations)
   - "none" (no structural steel detected)

4. **UNIT_SYSTEM**: "mm" | "inch" | "meter" | "mixed" | "unknown"

5. **GRID_CONVENTION** (how grid axes are labeled):
   - "letter_x_number_y" (A-B-C × 1-2-3 — most common)
   - "number_x_number_y" (1-2-3 × 1-2-3)
   - "number_x_letter_y" (1-2-3 × A-B-C)
   - "axis_labels" (e.g. "Axis 1", "Axis A", "Trục 1")
   - "northing_easting" (N123-E456 coordinate pairs)
   - "custom" (non-standard labels like N1-N4, S1-S4)
   - "none_detected" (no grid system visible)
   
   List ALL grid labels you can identify (e.g. ["A","B","C","D","E","F","G","H"] for X, ["1","2","3","4","5","6","7","8","9"] for Y)
   If found, also estimate the NUMBER of grid lines in each direction.

6. **FLOOR_COUNT**: Estimate number of floor levels visible (0 if only plan views, N if elevations show N levels above ground)

7. **DRAWING_SCALE** (if visible): "1:100" | "1:50" | "1:200" | "1:250" | "1:500" | "not_to_scale" | "multiple" | "unknown"

8. **LANGUAGE**: primary language of annotations — "english" | "vietnamese" | "chinese" | "japanese" | "korean" | "mixed" | "other"

9. **BUILDING_TYPE** (best guess): "portal_frame" | "multi_storey_rc" | "steel_warehouse" | "residential" | "commercial" | "industrial" | "bridge" | "unknown"

10. **ARCHITECTURAL_ELEMENTS_DETECTED**: list what architectural features are visible:
    ["walls", "doors", "windows", "stairs", "slabs", "roof", "finishes", "dimensions", "grid_lines", "none"]

Return JSON:
{
  "pdf_type": "...",
  "text_layer": "...",
  "steel_standard": "...",
  "unit_system": "...",
  "grid_convention": "...",
  "grid_labels_x": [...],
  "grid_labels_y": [...],
  "grid_count_x": <number>,
  "grid_count_y": <number>,
  "floor_count": <number>,
  "drawing_scale": "...",
  "language": "...",
  "building_type": "...",
  "architectural_elements": [...],
  "confidence": "high" | "medium" | "low",
  "notes": "<any important observations>"
}
"""


def analyze_pdf(pdf_path: str) -> dict:
    """
    Analyze the PDF to detect convention, unit, grid, steel standard.
    Feeds all downstream agents.
    Returns the analysis dict.
    """
    rprint("\n[bold cyan]" + "=" * 56)
    rprint("[bold cyan]AGENT 0 - PDF Analysis (Convention Detection)[/]")
    rprint("[bold cyan]" + "=" * 56 + "[/]")

    try:
        total_pages = get_page_count(pdf_path)
    except Exception:
        total_pages = 0
    rprint(f"  Total pages: {total_pages}")

    # Sample pages: first page + last page + a couple from the middle
    pages_to_sample = [0]  # first page always
    if total_pages > 1:
        pages_to_sample.append(total_pages - 1)  # last page
    if total_pages > 10:
        pages_to_sample.append(total_pages // 2)  # middle
    if total_pages > 20:
        pages_to_sample.append(total_pages // 3)
        pages_to_sample.append(2 * total_pages // 3)

    pages_to_sample = sorted(set(pages_to_sample))
    rprint(f"  Sampling pages: {[p+1 for p in pages_to_sample]}")

    # Render sampled pages at 150 DPI for the vision model
    images = []
    for page_idx in pages_to_sample:
        try:
            img = render_page_fast(pdf_path, page_idx, dpi=150)
            images.append(img)
            rprint(f"  [dim]→ rendered p{page_idx+1}[/]")
        except Exception as e:
            rprint(f"  [yellow]WARNING: could not render p{page_idx+1}: {e}[/]")

    if not images:
        rprint("[red]ERROR: No pages could be rendered. Cannot analyze PDF.[/]")
        return _fallback_analysis()

    # Also extract any text from first page to help detection
    text_snippet = ""
    try:
        text_snippet = extract_text_from_page(pdf_path, 0)
        if text_snippet:
            text_snippet = text_snippet[:3000]  # first 3000 chars
            rprint(f"  Text extracted from page 1: {len(text_snippet)} chars")
    except Exception:
        pass

    # Build enhanced prompt with text context
    enhanced_prompt = ANALYZE_PROMPT
    if text_snippet:
        enhanced_prompt += f"\n\n--- TEXT EXTRACTED FROM PAGE 1 (helps detection) ---\n{text_snippet}\n--- END TEXT ---"

    try:
        raw = call_llm_json(enhanced_prompt, image_parts=images)
        analysis = json.loads(raw)
    except Exception as e:
        rprint(f"[red]PDF analysis LLM error: {e}[/]")
        analysis = _fallback_analysis()

    # Ensure all required keys exist
    analysis = _validate_and_fill(analysis)
    analysis["total_pages"] = total_pages
    analysis["pages_sampled"] = [p + 1 for p in pages_to_sample]

    # Save
    Path(OUTPUT_JSON_DIR).mkdir(parents=True, exist_ok=True)
    with open(PDF_ANALYSIS_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    rprint(f"\n[bold green]PDF Analysis complete.[/] → {PDF_ANALYSIS_OUTPUT}")
    rprint(f"  Type:       {analysis['pdf_type']}")
    rprint(f"  Standard:   {analysis['steel_standard']}")
    rprint(f"  Units:      {analysis['unit_system']}")
    rprint(f"  Grid:       {analysis['grid_convention']} ({analysis['grid_count_x']}x{analysis['grid_count_y']})")
    rprint(f"  Floors:     {analysis['floor_count']}")
    rprint(f"  Text layer: {analysis['text_layer']}")
    rprint(f"  Language:   {analysis['language']}")
    rprint(f"  Building:   {analysis['building_type']}")

    return analysis


def load_analysis() -> dict:
    """Load previously saved analysis (for downstream agents)."""
    path = Path(PDF_ANALYSIS_OUTPUT)
    if not path.exists():
        rprint("[yellow]pdf_analysis.json not found — using default Australian assumptions.[/]")
        return _fallback_analysis()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _fallback_analysis()


def _validate_and_fill(analysis: dict) -> dict:
    """Ensure all required keys are present with sensible defaults."""
    defaults = {
        "pdf_type": "unknown",
        "text_layer": "digital",
        "steel_standard": "australian",
        "unit_system": "mm",
        "grid_convention": "letter_x_number_y",
        "grid_labels_x": ["A", "B", "C", "D", "E", "F", "G", "H"],
        "grid_labels_y": ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "grid_count_x": 8,
        "grid_count_y": 9,
        "floor_count": 2,
        "drawing_scale": "unknown",
        "language": "english",
        "building_type": "unknown",
        "architectural_elements": ["none"],
        "confidence": "low",
        "notes": "",
    }
    for key, default_value in defaults.items():
        if key not in analysis or analysis[key] is None:
            analysis[key] = default_value
    return analysis


def _fallback_analysis() -> dict:
    """Fallback when PDF analysis fails completely."""
    return _validate_and_fill({
        "pdf_type": "unknown",
        "text_layer": "digital",
        "steel_standard": "australian",
        "unit_system": "mm",
        "grid_convention": "letter_x_number_y",
        "grid_labels_x": ["A","B","C","D","E","F","G","H"],
        "grid_labels_y": ["1","2","3","4","5","6","7","8","9"],
        "grid_count_x": 8,
        "grid_count_y": 9,
        "floor_count": 2,
        "drawing_scale": "unknown",
        "language": "english",
        "building_type": "unknown",
        "architectural_elements": ["none"],
        "confidence": "low",
        "notes": "Fallback — analysis failed, assuming Australian standard",
    })