"""
Agent 3 — Spatial Parser  (NEW)
Phase 3: Extract the spatial reference system from Plan and Elevation pages.

  - PLAN views  → Grid X (columns A,B,C...) and Grid Y (rows 1,2,3...) with distances
  - ELEVATION views → Z-level system (Base, FL1, Roof, etc.) with heights in mm

Output: data/output_json/spatial_data.json
"""

import json
from pathlib import Path
from rich import print as rprint

from config import SPATIAL_OUTPUT_FILE
from core.llm_wrapper import call_llm_json
from core.pdf_utils import render_page_as_image_part, segment_page_regions


PLAN_PROMPT = """You are a Senior Structural Detailer. This is a STRUCTURAL PLAN VIEW drawing.
Extract the complete grid system (column and row grid lines).

Rules:
- Grid X labels are typically letters: A, B, C, D... (left to right)
- Grid Y labels are typically numbers: 1, 2, 3, 4... (bottom to top)
- All distances are CUMULATIVE from the origin (0,0).
- Read dimension chains carefully — they may be incremental (e.g. 6000, 5000) not cumulative.
  Convert to cumulative: [0, 6000, 11000].
- If a dimension says "6000" between grids, the second grid is 6000mm from the first.

Return JSON:
{
  "view_type": "PLAN",
  "drawing_ref": "<drawing number or title if visible>",
  "origin": {"x": 0, "y": 0},
  "grids_x": [
    {"name": "A", "x_mm": 0},
    {"name": "B", "x_mm": 6000},
    {"name": "C", "x_mm": 11000}
  ],
  "grids_y": [
    {"name": "1", "y_mm": 0},
    {"name": "2", "y_mm": 5000},
    {"name": "3", "y_mm": 9500}
  ]
}
If no grid found, return: {"view_type": "PLAN", "grids_x": [], "grids_y": []}"""


ELEVATION_PROMPT = """You are a Senior Structural Detailer. This is a STRUCTURAL ELEVATION or SECTION VIEW drawing.
Extract EVERY floor level and datum height shown anywhere on this page (Z-axis).

Where to look:
- Datum triangles ▽ or ▼ next to a height label
- Horizontal dashed lines labelled with a level name and RL/EL/FFL value
- Portal frame elevations: look for GROUND, EAVE, HAUNCH, RIDGE, APEX height callouts
- Section views: look for FINISHED FLOOR LEVEL (FFL), TOP OF SLAB (TOS), TOP OF STEEL (TOS) markers
- Any text matching: BASE / GND / GROUND / RL / FFL / FL0 / FL1 / FL2 / FL3 / LEVEL 1 / LEVEL 2 / EAVE / HAUNCH / RIDGE / ROOF / PARAPET / TOP PLATE

Conversion rules:
- Express ALL heights in MILLIMETERS, relative to Base = 0.
- If the drawing uses metres (e.g. 4.500), multiply by 1000 → 4500 mm.
- If only incremental heights are shown (e.g. "3600 above FL1"), accumulate them from Base = 0.
- If Base/Ground is not explicitly labelled, treat the lowest level on the page as z_mm = 0.

Return JSON — include EVERY distinct level found, even if the list is long.
Use "ELEVATION" for full-height elevation views; use "SECTION" for cross-section cuts:
{
  "view_type": "ELEVATION",
  "drawing_ref": "<drawing number or title if visible>",
  "levels": [
    {"name": "GROUND",  "z_mm": 0},
    {"name": "FL1",     "z_mm": 3600},
    {"name": "EAVE",    "z_mm": 4200},
    {"name": "RIDGE",   "z_mm": 5800},
    {"name": "ROOF",    "z_mm": 6000}
  ]
}
If no level markers are visible anywhere on the page, return: {"view_type": "ELEVATION", "levels": []}"""


def _merge_spatial_results(all_results: list[dict]) -> dict:
    """Merge multiple plan/elevation extractions into one unified spatial dataset."""
    merged = {
        "grids_x": [],
        "grids_y": [],
        "levels": [],
    }
    seen_gx = set()
    seen_gy = set()
    seen_lv = set()

    for result in all_results:
        vtype = result.get("view_type", "")
        if vtype == "PLAN":
            for g in result.get("grids_x", []):
                if g["name"] not in seen_gx:
                    seen_gx.add(g["name"])
                    merged["grids_x"].append(g)
            for g in result.get("grids_y", []):
                if g["name"] not in seen_gy:
                    seen_gy.add(g["name"])
                    merged["grids_y"].append(g)
        elif vtype in ("ELEVATION", "SECTION"):
            for lv in result.get("levels", []):
                if lv["name"] not in seen_lv:
                    seen_lv.add(lv["name"])
                    merged["levels"].append(lv)

    # Sort grids and levels
    merged["grids_x"].sort(key=lambda g: g.get("x_mm", 0))
    merged["grids_y"].sort(key=lambda g: g.get("y_mm", 0))
    merged["levels"].sort(key=lambda lv: lv.get("z_mm", 0))

    # Guarantee a Base level exists
    if not merged["levels"]:
        merged["levels"] = [{"name": "Base", "z_mm": 0}]

    return merged


def parse_spatial_pages(
    pdf_path: str,
    plan_pages: list[int],
    elevation_pages: list[int],
) -> dict:
    all_results = []

    for page_num in plan_pages:
        rprint(f"[bold green]Spatial Parser:[/] Plan page {page_num + 1}...")
        regions = segment_page_regions(pdf_path, page_num)
        for img_part in regions:
            try:
                raw = call_llm_json(PLAN_PROMPT, image_parts=[img_part])
                parsed = json.loads(raw)
                if parsed.get("grids_x") or parsed.get("grids_y"):
                    all_results.append(parsed)
                    rprint(f"  [green]Grid X:{len(parsed.get('grids_x',[]))} Y:{len(parsed.get('grids_y',[]))}[/]")
            except Exception as e:
                rprint(f"  [red]Plan parse error: {e}[/]")

    for page_num in elevation_pages:
        rprint(f"[bold green]Spatial Parser:[/] Elevation page {page_num + 1}...")
        regions = segment_page_regions(pdf_path, page_num)
        for img_part in regions:
            try:
                raw = call_llm_json(ELEVATION_PROMPT, image_parts=[img_part])
                parsed = json.loads(raw)
                if parsed.get("levels"):
                    all_results.append(parsed)
                    rprint(f"  [green]Levels: {[l['name'] for l in parsed.get('levels', [])]}[/]")
            except Exception as e:
                rprint(f"  [red]Elevation parse error: {e}[/]")

    spatial = _merge_spatial_results(all_results)

    Path(SPATIAL_OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(SPATIAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(spatial, f, indent=2, ensure_ascii=False)

    rprint(f"\n[bold green]Spatial Parser complete.[/] "
           f"GridX:{len(spatial['grids_x'])} GridY:{len(spatial['grids_y'])} "
           f"Levels:{len(spatial['levels'])} → {SPATIAL_OUTPUT_FILE}")
    return spatial
