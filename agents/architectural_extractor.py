"""
Agent 1c — Architectural Element Extractor (NEW for LOD300/400)
Phase 1c: Extract walls, slabs, doors, windows, stairs from plan/elevation views.
Runs AFTER Scanner identifies which pages are architectural.

This fills the gap: the existing pipeline only handles structural steel with
schedule tables. This agent extracts architectural geometry from plan views
so that ANY PDF — structural only, architectural only, or combined — can
produce a 3D SketchUp model.

Inputs:
  - PDF path
  - Plan pages (from Scanner drawing_index.json)
  - Elevation pages (from Scanner drawing_index.json)
  - PDF analysis (convention detection)

Output: data/output_json/architectural_elements.json
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional

import pdfplumber
from rich import print as rprint

from config import OUTPUT_JSON_DIR
from core.llm_wrapper import call_llm_json, call_llm
from core.pdf_utils import render_page_as_image_part, extract_text_from_page
from core.analysis_context import build_context_string, load_analysis_dict

ARCHITECTURAL_OUTPUT_FILE = str(Path(OUTPUT_JSON_DIR) / "architectural_elements.json")

_LLM_CALL_TIMEOUT   = 90    # seconds per individual LLM call
_PHASE_TOTAL_TIMEOUT = 300  # seconds total budget for Phase 5b


def _call_with_timeout(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) with _LLM_CALL_TIMEOUT-second timeout.
    Uses shutdown(wait=False) so a timed-out thread doesn't block continuation.
    """
    ex = ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(fn, *args, **kwargs)
        return fut.result(timeout=_LLM_CALL_TIMEOUT)
    finally:
        ex.shutdown(wait=False)


# ── Prompt: Extract walls, openings, stairs from plan view ──────────────────
PLAN_EXTRACT_PROMPT = """You are a BIM architect analyzing an architectural plan view drawing.
Your job: extract ALL architectural elements visible on this plan with their DIMENSIONS.

{analysis_context}

Examine this plan view image carefully. Extract:

1. **WALLS** — every wall line visible:
   - type: "external" | "internal" | "shear" | "curtain"
   - thickness_mm: wall thickness in mm
   - start_point: {x_mm, y_mm} — wall start on plan
   - end_point: {x_mm, y_mm} — wall end on plan
   - height_mm: wall height (if visible in elevation/section) or estimate based on floor-to-floor
   - grids: which grid lines this wall aligns with (e.g. ["A", "1-2"])
   - doors_on_wall: list of door marks if openings visible

2. **DOORS** — every door symbol visible:
   - mark: door ID (e.g. "D1", "D01")
   - type: "single_swing" | "double_swing" | "sliding" | "folding" | "roller" | "other"
   - width_mm: door opening width
   - height_mm: door height (default 2100 if not shown)
   - position: {x_mm, y_mm} — center of door on plan
   - wall_grid_ref: which wall/grid this door is on

3. **WINDOWS** — every window symbol visible:
   - mark: window ID (e.g. "W1", "WIN01")
   - type: "fixed" | "sliding" | "casement" | "awning" | "louvre" | "other"
   - width_mm: window width
   - height_mm: window height
   - sill_height_mm: height from floor to window sill (default 900)
   - position: {x_mm, y_mm} — center of window on plan
   - wall_grid_ref: which wall/grid this window is on

4. **STAIRS** — any stair symbols:
   - mark: stair ID (e.g. "ST1")
   - type: "straight" | "l_shape" | "u_shape" | "spiral" | "other"
   - width_mm: stair width
   - rise_mm: total rise height
   - run_mm: total run length
   - position: {x_mm, y_mm} — bottom start point
   - direction: "up" | "down"
   - num_risers: number of steps (estimate from drawing)

5. **SLABS / FLOORS** — floor plate boundaries:
   - level_name: which floor level (e.g. "Floor 1", "Roof")
   - thickness_mm: slab thickness
   - boundary_points: list of {x_mm, y_mm} defining slab outline
   - openings: list of slab openings (stair voids, shafts)

6. **COLUMNS** (architectural — different from structural schedule):
   - mark: column ID if visible
   - position: {x_mm, y_mm}
   - width_mm, depth_mm: column dimensions
   - height_mm: column height

7. **GRID LINES** (confirm/refine):
   - x_axis_lines: [{label, x_mm}] — vertical grid lines
   - y_axis_lines: [{label, y_mm}] — horizontal grid lines
   - grid_spacing_mm: typical spacing

8. **ROOMS / SPACES** — named spaces:
   - name: room name (e.g. "Living Room", "Phòng khách")
   - area_m2: approximate area
   - boundary_points: list of {x_mm, y_mm}

Return JSON:
{
  "walls": [...],
  "doors": [...],
  "windows": [...],
  "stairs": [...],
  "slabs": [...],
  "columns": [...],
  "grid_lines": {
    "x_axis": [...],
    "y_axis": [...],
    "spacing_mm": 6000
  },
  "rooms": [...],
  "floor_level": "Floor 1",
  "floor_to_floor_height_mm": 3500,
  "confidence": "high" | "medium" | "low"
}
"""

# ── Prompt: Extract elevations and vertical dimensions ──────────────────────
ELEVATION_EXTRACT_PROMPT = """You are analyzing an architectural elevation/section drawing.
Extract VERTICAL dimensions and heights that plan views alone cannot show.

{analysis_context}

From this elevation/section drawing, extract:

1. **FLOOR LEVELS** — every horizontal level line:
   - name: level name (e.g. "Ground Floor", "Floor 1", "Roof")
   - z_mm: elevation in mm from ground (0 = ground floor finish level)
   - is_structural: true if this is a structural floor, false for finished floor

2. **WALL HEIGHTS** (any visible):
   - grid_ref: which grid/wall
   - top_z_mm: wall top elevation
   - parapet_height_mm: if parapet visible

3. **ROOF PROFILE**:
   - type: "flat" | "gable" | "hip" | "shed" | "other"
   - ridge_z_mm: highest point
   - eaves_z_mm: lowest edge
   - slope_degrees: roof pitch

4. **OPENING HEIGHTS** (doors/windows visible in elevation):
   - mark: matches plan door/window ID
   - head_height_mm: top of opening
   - sill_height_mm: bottom of opening (windows)

5. **BUILDING OUTLINE**:
   - overall_width_mm: total building width
   - overall_height_mm: total building height from ground to roof
   - overall_depth_mm: total building depth

Return JSON:
{
  "floor_levels": [...],
  "wall_heights": [...],
  "roof": {...},
  "opening_heights": [...],
  "building_outline": {...},
  "confidence": "high" | "medium" | "low"
}
"""


def extract_architectural_elements(
    pdf_path: str,
    plan_pages: list[int] | None = None,
    elevation_pages: list[int] | None = None,
    pdf_analysis: Optional[dict] = None,
) -> dict:
    """
    Main entry point. Extracts all architectural elements from plan + elevation pages.

    Args:
        pdf_path: Path to the PDF file
        plan_pages: List of 0-based page indices for plan views (auto-loaded if None)
        elevation_pages: List of 0-based page indices for elevation views (auto-loaded if None)
        pdf_analysis: Phase 0 analysis dict (convention detection)

    Returns:
        dict with keys: walls, doors, windows, stairs, slabs, columns, grid_lines,
                        rooms, floor_levels, roof, building_outline
    """
    if pdf_analysis is None:
        pdf_analysis = load_analysis_dict()

    # Auto-load plan/elevation pages from scanner output when not supplied
    if plan_pages is None or elevation_pages is None:
        try:
            from config import SCANNER_OUTPUT_FILE as _sf_path
            _scanner = json.loads(Path(_sf_path).read_text(encoding="utf-8"))
            _roles: dict = {}
            for _k, _v in _scanner.items():
                if isinstance(_v, dict) and "role" in _v:
                    _roles.setdefault(_v["role"], []).append(int(_k))
            if plan_pages is None:
                plan_pages = sorted(_roles.get("plan", []))
            if elevation_pages is None:
                elevation_pages = sorted(_roles.get("elevation", []))
        except Exception:
            plan_pages = plan_pages or []
            elevation_pages = elevation_pages or []

    arch_elements_flag = pdf_analysis.get("architectural_elements", [])
    pdf_type = pdf_analysis.get("pdf_type", "unknown")

    if pdf_type == "structural_only" and "walls" not in arch_elements_flag:
        rprint("[dim]Architectural extractor: structural-only PDF detected — skipping wall/slab extraction.[/]")
        return _empty_result()

    rprint("\n[bold magenta]══════════════════════════════════════════════════════")
    rprint("[bold magenta]AGENT 1c — Architectural Element Extraction[/]")
    rprint("[bold magenta]══════════════════════════════════════════════════════[/]")

    analysis_context = build_context_string(pdf_analysis)
    result = _empty_result()
    phase_start = time.time()

    # Step 1: Text extraction from plan pages (no LLM — no timeout needed)
    if plan_pages:
        rprint(f"  Processing {len(plan_pages)} plan page(s) via text...")
        text_elements = _extract_from_plan_text(pdf_path, plan_pages, pdf_analysis)
        result = _merge_results(result, text_elements)

    # Step 2: Visual extraction from plan views via LLM (up to 2 pages, 90s each)
    plan_batch = plan_pages[:2]
    if plan_batch:
        for idx, pg in enumerate(plan_batch):
            # Total phase timeout guard
            if time.time() - phase_start > _PHASE_TOTAL_TIMEOUT:
                rprint("[yellow]Phase 5b exceeded 300s — generating steel-only model (architectural skipped)[/]")
                return _empty_result()
            try:
                img = render_page_as_image_part(pdf_path, pg)
                prompt = PLAN_EXTRACT_PROMPT.replace("{analysis_context}", analysis_context)
                raw = _call_with_timeout(call_llm_json, prompt, image_parts=[img])
                try:
                    plan_data = json.loads(raw)
                except json.JSONDecodeError:
                    from json_repair import repair_json
                    plan_data = json.loads(repair_json(raw))
                result = _merge_results(result, plan_data)
                rprint(
                    f"  Arch extraction: page {idx + 1}/{len(plan_batch)} done "
                    f"(walls: {len(result['walls'])}, slabs: {len(result['slabs'])}, "
                    f"doors: {len(result['doors'])})"
                )
            except FuturesTimeoutError:
                rprint(f"  [yellow]Arch page {pg + 1} timed out — skipped[/]")
            except Exception as e:
                rprint(f"  [yellow]Plan page {pg + 1} extraction failed: {e}[/]")

    # Step 3: Visual extraction from elevation views via LLM (up to 1 page, 90s)
    if elevation_pages:
        if time.time() - phase_start > _PHASE_TOTAL_TIMEOUT:
            rprint("[yellow]Phase 5b exceeded 300s — generating steel-only model (architectural skipped)[/]")
            return _empty_result()
        for pg in elevation_pages[:1]:
            try:
                img = render_page_as_image_part(pdf_path, pg)
                prompt = ELEVATION_EXTRACT_PROMPT.replace("{analysis_context}", analysis_context)
                raw = _call_with_timeout(call_llm_json, prompt, image_parts=[img])
                try:
                    elev_data = json.loads(raw)
                except json.JSONDecodeError:
                    from json_repair import repair_json
                    elev_data = json.loads(repair_json(raw))
                rprint(f"  [green]Elevation page {pg + 1}:[/] "
                       f"{len(elev_data.get('floor_levels', []))} levels")
                result = _merge_results(result, elev_data)
            except FuturesTimeoutError:
                rprint(f"  [yellow]Arch page {pg + 1} (elevation) timed out — skipped[/]")
            except Exception as e:
                rprint(f"  [yellow]Elevation page {pg + 1} extraction failed: {e}[/]")

    # Step 4: Post-processing — rationalize coordinates
    result = _rationalize_coordinates(result, pdf_analysis)

    # Save
    Path(OUTPUT_JSON_DIR).mkdir(parents=True, exist_ok=True)
    with open(ARCHITECTURAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    _print_summary(result)
    rprint(f"Architectural elements → {ARCHITECTURAL_OUTPUT_FILE}")
    return result


def _extract_from_plan_text(
    pdf_path: str, plan_pages: list[int], pdf_analysis: dict
) -> dict:
    """Extract architectural elements using pdfplumber text extraction from plan pages."""
    result = _empty_result()
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            n_pages = len(pdf.pages)
            for pg_idx in plan_pages:
                if pg_idx >= n_pages:
                    continue
                page = pdf.pages[pg_idx]
                
                # Try to extract tables (for door/window schedules)
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    _parse_table_for_openings(table, result)
                
                # Extract text lines for wall dimensions
                text = page.extract_text()
                if text:
                    _parse_dimension_lines(text, result, pdf_analysis)
                    
    except Exception as e:
        rprint(f"  [yellow]Text extraction warning: {e}[/]")
    
    return result


def _parse_table_for_openings(table, result: dict):
    """Parse a table that may contain door/window schedules."""
    if not table or len(table) < 2:
        return
    
    headers = [str(h).strip().lower() if h else "" for h in table[0]]
    
    # Detect if this is a door schedule
    door_keywords = ["door", "cửa", "cua", "door type", "door no", "d no"]
    window_keywords = ["window", "cửa sổ", "cua so", "win type", "w no"]
    
    is_door = any(kw in " ".join(headers) for kw in door_keywords)
    is_window = any(kw in " ".join(headers) for kw in window_keywords)
    
    if not (is_door or is_window):
        return
    
    # Find column indices
    mark_col = _find_column(headers, ["mark", "no", "ký hiệu", "ki hieu", "door no", "win no", "type", "số"])
    width_col = _find_column(headers, ["width", "rộng", "rong", "w", "bề rộng"])
    height_col = _find_column(headers, ["height", "cao", "h", "chiều cao"])
    
    for row in table[1:]:
        if not row or all(c is None or str(c).strip() == "" for c in row):
            continue
        
        mark = str(row[mark_col]).strip() if mark_col < len(row) and row[mark_col] else ""
        if not mark or mark.lower() in ["mark", "no", "type", ""]:
            continue
        
        width_val = _parse_number(str(row[width_col])) if width_col < len(row) else 0
        height_val = _parse_number(str(row[height_col])) if height_col < len(row) else 0
        
        entry = {
            "mark": mark,
            "width_mm": width_val,
            "height_mm": height_val,
            "position": {"x": 0, "y": 0},
            "wall_grid_ref": "",
            "source": "schedule_table",
        }
        
        if is_door:
            entry["type"] = "single_swing"
            result["doors"].append(entry)
        elif is_window:
            entry["type"] = "fixed"
            entry["sill_height_mm"] = 900
            result["windows"].append(entry)


def _parse_dimension_lines(text: str, result: dict, pdf_analysis: dict):
    """Extract dimension strings like '6000' or '6.000' that indicate wall lengths."""
    dim_pattern = re.compile(r'(\d[\d\s.,]*)\s*(?:mm|m)?')
    
    # Look for grid spacing patterns
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Grid labels: A, B, C, 1, 2, 3 etc with dimensions between them
        matches = dim_pattern.findall(line)
        if len(matches) >= 2:
            dims = []
            for m in matches:
                try:
                    val = float(m.replace(' ', '').replace(',', '.'))
                    if 500 < val < 50000:  # sensible range in mm
                        dims.append(val)
                except ValueError:
                    pass
            
            if dims and not result["grid_lines"].get("spacing_mm"):
                result["grid_lines"]["spacing_mm"] = max(set(dims), key=dims.count)


def _rationalize_coordinates(result: dict, pdf_analysis: dict) -> dict:
    """Post-process: align coordinates to grid, fill missing dimensions."""
    grid = result.get("grid_lines", {})
    spacing = grid.get("spacing_mm", 6000)
    
    # If we have grid x_axis lines, use them
    x_axis = grid.get("x_axis", [])
    y_axis = grid.get("y_axis", [])
    
    # Snap all wall positions to nearest grid intersection
    for wall in result.get("walls", []):
        sp = wall.get("start_point", {})
        ep = wall.get("end_point", {})
        if sp.get("x"):
            sp["x"] = round(sp["x"] / spacing) * spacing
        if sp.get("y"):
            sp["y"] = round(sp["y"] / spacing) * spacing
        if ep.get("x"):
            ep["x"] = round(ep["x"] / spacing) * spacing
        if ep.get("y"):
            ep["y"] = round(ep["y"] / spacing) * spacing
        
        # Default wall height
        if not wall.get("height_mm"):
            wall["height_mm"] = result.get("floor_to_floor_height_mm", 3500)
    
    return result


def _merge_results(base: dict, new: dict) -> dict:
    """Merge two extraction results, avoiding duplicates by mark."""
    for key in ["walls", "doors", "windows", "stairs", "slabs", "columns", "rooms"]:
        existing_marks = {item.get("mark", "") for item in base.get(key, [])}
        for item in new.get(key, []):
            if item.get("mark", "") not in existing_marks:
                base[key].append(item)
    
    # Merge grid lines
    if new.get("grid_lines"):
        for axis in ["x_axis", "y_axis"]:
            existing_labels = {g.get("label", "") for g in base["grid_lines"].get(axis, [])}
            for g in new["grid_lines"].get(axis, []):
                if g.get("label", "") not in existing_labels:
                    base["grid_lines"][axis].append(g)
        if new["grid_lines"].get("spacing_mm"):
            base["grid_lines"]["spacing_mm"] = new["grid_lines"]["spacing_mm"]
    
    # Merge floor levels
    existing_levels = {l.get("name", "") for l in base.get("floor_levels", [])}
    for l in new.get("floor_levels", []):
        if l.get("name", "") not in existing_levels:
            base["floor_levels"].append(l)
    
    # Simple values (keep if not already set)
    for key in ["roof", "building_outline", "floor_level", "floor_to_floor_height_mm"]:
        if new.get(key) and not base.get(key):
            base[key] = new[key]
    
    return base


def _find_column(headers: list[str], keywords: list[str]) -> int:
    """Find column index matching any keyword."""
    for i, h in enumerate(headers):
        h_lower = h.lower()
        for kw in keywords:
            if kw in h_lower:
                return i
    return 0


def _parse_number(s: str) -> float:
    """Extract numeric value from a string like '900 x 2100' or '900mm'."""
    s = s.strip()
    # Handle '900 x 2100' — take first number
    s = s.split('x')[0] if 'x' in s.lower() else s
    # Remove non-numeric chars
    s = re.sub(r'[^\d.,]', '', s)
    try:
        return float(s.replace(',', '.'))
    except ValueError:
        return 0


def _print_summary(result: dict):
    """Print extraction summary."""
    walls = len(result.get("walls", []))
    doors = len(result.get("doors", []))
    windows = len(result.get("windows", []))
    stairs = len(result.get("stairs", []))
    slabs = len(result.get("slabs", []))
    cols = len(result.get("columns", []))
    rooms = len(result.get("rooms", []))
    levels = len(result.get("floor_levels", []))
    
    rprint(f"\n[bold green]Architectural Extraction Summary:[/]")
    rprint(f"  Walls:   {walls}")
    rprint(f"  Doors:   {doors}")
    rprint(f"  Windows: {windows}")
    rprint(f"  Stairs:  {stairs}")
    rprint(f"  Slabs:   {slabs}")
    rprint(f"  Columns: {cols}")
    rprint(f"  Rooms:   {rooms}")
    rprint(f"  Levels:  {levels}")


def _empty_result() -> dict:
    """Return an empty architectural elements dict."""
    return {
        "walls": [],
        "doors": [],
        "windows": [],
        "stairs": [],
        "slabs": [],
        "columns": [],
        "grid_lines": {
            "x_axis": [],
            "y_axis": [],
            "spacing_mm": 6000,
        },
        "rooms": [],
        "floor_levels": [],
        "roof": {},
        "building_outline": {},
        "floor_level": "Floor 1",
        "floor_to_floor_height_mm": 3500,
        "confidence": "low",
    }