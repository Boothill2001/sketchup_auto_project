"""
Agent 3 — Spatial Parser  (NEW)
Phase 3: Extract the spatial reference system from Plan and Elevation pages.

  - PLAN views  → Grid X (columns) and Grid Y (rows) with distances
  - ELEVATION views → Z-level system (Base, FL1, Roof, etc.) with heights in mm

Adapts to PDF convention via Phase 0 analysis.
Output: data/output_json/spatial_data.json
"""

import json
import re
from pathlib import Path
from rich import print as rprint

from config import SPATIAL_OUTPUT_FILE
from core.llm_wrapper import call_llm_json
from core.pdf_utils import render_page_as_image_part, segment_page_regions, extract_text_from_page
from core.analysis_context import build_plan_context


PLAN_PROMPT = """You are a Senior Structural Detailer. This is a STRUCTURAL PLAN VIEW drawing.
Extract the complete grid system (column and row grid lines).

{analysis_context}

Rules:
- Read grid labels EXACTLY as they appear on the drawing (use the convention above).
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
    {{"name": "A", "x_mm": 0}},
    {{"name": "B", "x_mm": 6000}},
    {{"name": "C", "x_mm": 11000}}
  ],
  "grids_y": [
    {{"name": "1", "y_mm": 0}},
    {{"name": "2", "y_mm": 5000}},
    {{"name": "3", "y_mm": 9500}}
  ]
}

NOTE: Grid lines may appear as:
 - Circles/bubbles with labels at the end of lines
 - Dimension chains showing cumulative distances
 - Column centerlines labeled in title block legend
 - "Trục" prefix (Vietnamese) for axis labels

If no grid found, return: {{"view_type": "PLAN", "grids_x": [], "grids_y": []}}"""


ELEVATION_PROMPT = """You are a Senior Structural Detailer. This is a STRUCTURAL ELEVATION or SECTION VIEW drawing.
Extract EVERY floor level and datum height shown anywhere on this page (Z-axis).

{analysis_context}

Where to look:
- Datum triangles ▽ or ▼ next to a height label
- Horizontal dashed lines labelled with a level name and RL/EL/FFL value
- Portal frame elevations: look for GROUND, EAVE, HAUNCH, RIDGE, APEX height callouts
- Section views: look for FINISHED FLOOR LEVEL (FFL), TOP OF SLAB (TOS), TOP OF STEEL (TOS) markers
- Any text matching: BASE / GND / GROUND / RL / FFL / FL0 / FL1 / FL2 / FL3 / LEVEL 1 / LEVEL 2 / EAVE / HAUNCH / RIDGE / ROOF / PARAPET / TOP PLATE / TẦNG / CỐT / MÁI (Vietnamese)

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
    {{"name": "GROUND",  "z_mm": 0}},
    {{"name": "FL1",     "z_mm": 3600}},
    {{"name": "EAVE",    "z_mm": 4200}},
    {{"name": "RIDGE",   "z_mm": 5800}},
    {{"name": "ROOF",    "z_mm": 6000}}
  ]
}
If no level markers are visible anywhere on the page, return: {{"view_type": "ELEVATION", "levels": []}}"""


FALLBACK_GRID_PROMPT = """This is a structural drawing. Looking at all visible column grid lines, identify the grid reference system.

{analysis_context}

Return JSON using the same schema:
{
  "view_type": "PLAN",
  "drawing_ref": "<drawing number or title if visible>",
  "origin": {"x": 0, "y": 0},
  "grids_x": [{"name": "A", "x_mm": 0}, {"name": "B", "x_mm": 6000}],
  "grids_y": [{"name": "1", "y_mm": 0}, {"name": "2", "y_mm": 5000}]
}
If still no grid reference system visible, return: {"view_type": "PLAN", "grids_x": [], "grids_y": []}"""


def _score_page(pdf_path: str, page: int) -> int:
    try:
        text = extract_text_from_page(pdf_path, page)
        return len(re.findall(
            r'\b[A-H]\b|\b[1-9]\b|GRID|LEVEL\s*\d|RL\s*[\d.]|FL\d|EL[\d.]'
            r'|FFL|TOS|EAVE|RIDGE|HAUNCH|PARAPET|ROOF|GND|GROUND'
            r'|TẦNG|CỐT|MÁI|TRỤC'            # Vietnamese
            r'|\b\d{3,5}\b',   # bare mm numbers often in elevation dims
            text, re.IGNORECASE,
        ))
    except Exception:
        return 0


def select_best_page(pages: list[int], pdf_path: str) -> int:
    """Return the single most information-rich page from a list of same-role pages."""
    if len(pages) == 1:
        return pages[0]
    return max(pages, key=lambda p: _score_page(pdf_path, p))


def select_top_pages(pages: list[int], pdf_path: str, max_n: int = 2) -> list[int]:
    """Return up to max_n pages ranked by structural grid/level reference density."""
    if len(pages) <= max_n:
        return list(pages)
    scored = sorted(pages, key=lambda p: _score_page(pdf_path, p), reverse=True)
    return scored[:max_n]


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

    # If only 0 or 1 levels found, apply multi-floor structural default
    # (typical 3-storey RC building: GF + 3 floors at 3500mm each)
    if len(merged["levels"]) <= 1:
        merged["levels"] = [
            {"name": "Base",  "z_mm":     0},
            {"name": "FL1",   "z_mm":  3500},
            {"name": "FL2",   "z_mm":  7000},
            {"name": "FL3",   "z_mm": 10500},
            {"name": "Roof",  "z_mm": 13500},
        ]

    return merged


import re as _re


def _extract_spatial_from_text(pdf_path: str, plan_pages: list, elev_pages: list) -> dict:
    """
    Phase 0: Extract grid dims + floor levels directly from pdfplumber text layer.
    Returns partial spatial dict (may be empty if PDF is pure image scan).
    """
    result = {"grids_x": [], "grids_y": [], "levels": []}
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            n = len(pdf.pages)

            # ── 1. Floor levels from elevation pages ──────────────────────────
            level_pos = {}   # {"LEVEL 01": y_pt, ...}
            elev_scale = None

            for pg_i in elev_pages[:4]:
                if pg_i >= n: continue
                page = pdf.pages[pg_i]
                words = page.extract_words()

                # Detect drawing scale (e.g. "1:100")
                for j, w in enumerate(words):
                    if w["text"] in ("1:100", "1:50", "1:200", "1:25"):
                        scale_map = {"1:100":100,"1:50":50,"1:200":200,"1:25":25}
                        elev_scale = scale_map[w["text"]]
                    # also "SCALE:" then ratio
                    if w["text"] == "SCALE:" and j+2 < len(words):
                        ratio = words[j+2]["text"]
                        try: elev_scale = int(ratio)
                        except ValueError: pass

                # Find LEVEL XX labels (also Vietnamese "TẦNG XX", "CỐT", "MÁI")
                for j, w in enumerate(words):
                    txt = w["text"].strip().upper()
                    # English: "LEVEL 01", "LEVEL 1"
                    if txt == "LEVEL" and j+1 < len(words):
                        num_w = words[j+1]
                        key = f"LEVEL {num_w['text']}"
                        if _re.match(r"^\d{1,2}$", num_w["text"]):
                            if key not in level_pos:
                                level_pos[key] = w["top"]
                    # Vietnamese: "TẦNG 1", "CỐT +0.00", "MÁI"
                    if txt in ("TẦNG", "CỐT", "MÁI"):
                        label = txt if txt == "MÁI" else f"{txt} {words[j+1]['text']}" if j+1 < len(words) else txt
                        if label not in level_pos:
                            level_pos[label] = w["top"]

            # Convert level Y-positions to Z heights using page scale
            if level_pos and elev_scale:
                MM_PER_PT = 25.4 / 72   # 1 PDF point = 0.3528 mm
                sorted_lvls = sorted(level_pos.items(), key=lambda x: x[1])
                ref_lvl, ref_y = sorted_lvls[-1]  # highest y-pt = lowest real level
                for lname, ly in sorted_lvls:
                    dy_pts = ref_y - ly          # positive = above ref
                    z_mm = round(dy_pts * MM_PER_PT * elev_scale / 50) * 50  # round to 50mm
                    result["levels"].append({"name": lname, "z_mm": int(z_mm)})
                result["levels"].sort(key=lambda l: l["z_mm"])

            # ── 2. Grid bay dims from plan pages ─────────────────────────────
            all_bays = []
            for pg_i in plan_pages[:4]:
                if pg_i >= n: continue
                page = pdf.pages[pg_i]
                words = page.extract_words()
                for w in words:
                    t = w["text"].strip()
                    # Typical structural bay dims: 1000-15000mm, round numbers
                    if _re.match(r"^\d{4,5}$", t):
                        val = int(t)
                        if 1000 <= val <= 15000 and val % 100 == 0:
                            all_bays.append(val)

            # Find most common bay dim
            if all_bays:
                from collections import Counter
                counts = Counter(all_bays)
                candidates = [v for v, c in counts.most_common(5) if c >= 2 and v >= 3000]
                if candidates:
                    bay = candidates[0]
                    result["grids_x"] = [
                        {"name": str(i+1), "x_mm": i * bay} for i in range(4)
                    ]
                    result["grids_y"] = [
                        {"name": chr(65+i), "y_mm": i * bay} for i in range(4)
                    ]

    except Exception:
        pass  # silently fall back to LLM

    return result


def parse_spatial_pages(
    pdf_path: str,
    plan_pages: list[int],
    elevation_pages: list[int],
) -> dict:
    all_results = []

    # ── Load PDF analysis context (adapts to convention) ────────────────────
    analysis_context = build_plan_context()
    plan_prompt = PLAN_PROMPT.replace("{analysis_context}", analysis_context)
    elev_prompt = ELEVATION_PROMPT.replace("{analysis_context}", analysis_context)
    fallback_prompt = FALLBACK_GRID_PROMPT.replace("{analysis_context}", analysis_context)

    # ── Phase 0: pdfplumber text extraction ──────────────────────────────────
    text_result = _extract_spatial_from_text(pdf_path, plan_pages, elevation_pages)
    if text_result["levels"] or text_result["grids_x"]:
        rprint(f"[green]Text layer:[/] {len(text_result['grids_x'])} grid lines, "
               f"{len(text_result['levels'])} levels extracted via pdfplumber")
        all_results.append(text_result)

    # ── Plan pages: up to 2 best pages, 1 API call each ──────────────────────
    top_plans: list[int] = []
    if plan_pages:
        top_plans = select_top_pages(plan_pages, pdf_path, max_n=2)
        rprint(
            f"[bold green]Spatial Parser:[/] Plan pages {[p+1 for p in top_plans]} "
            f"(top {len(top_plans)} of {len(plan_pages)})..."
        )
        for pg in top_plans:
            regions = segment_page_regions(pdf_path, pg)
            try:
                raw    = call_llm_json(plan_prompt, image_parts=[regions[0]])
                parsed = json.loads(raw)
                if parsed.get("grids_x") or parsed.get("grids_y"):
                    all_results.append(parsed)
                    rprint(
                        f"  [green]p{pg+1} Grid X:{len(parsed.get('grids_x', []))} "
                        f"Y:{len(parsed.get('grids_y', []))}[/]"
                    )
            except Exception as e:
                rprint(f"  [red]Plan parse error p{pg+1}: {e}[/]")

    # ── Validation: if top plan pages yielded no grids, try the rest ─────────
    _grids_found = any(
        r.get("grids_x") or r.get("grids_y")
        for r in all_results
        if r.get("view_type") == "PLAN"
    )
    if plan_pages and not _grids_found:
        rprint("[yellow]  WARNING: No grids found in top plan pages. Trying all plan pages...[/]")
        remaining_plans = [p for p in plan_pages if p not in top_plans]
        for pg in remaining_plans:
            regions = segment_page_regions(pdf_path, pg)
            try:
                raw    = call_llm_json(plan_prompt, image_parts=[regions[0]])
                parsed = json.loads(raw)
                if parsed.get("grids_x") or parsed.get("grids_y"):
                    all_results.append(parsed)
                    rprint(
                        f"  [green]p{pg+1} Grid X:{len(parsed.get('grids_x', []))} "
                        f"Y:{len(parsed.get('grids_y', []))}[/]"
                    )
                    _grids_found = True
            except Exception as e:
                rprint(f"  [red]Plan parse error p{pg+1}: {e}[/]")

    # ── Final fallback: send ALL plan pages in one call with the looser prompt ─
    if plan_pages and not _grids_found:
        rprint("[yellow]  WARNING: No grids after all plan pages. Running fallback grid extractor...[/]")
        fallback_images = []
        for pg in plan_pages:
            try:
                fallback_images.append(render_page_as_image_part(pdf_path, pg))
            except Exception:
                pass
        if fallback_images:
            try:
                raw    = call_llm_json(fallback_prompt, image_parts=fallback_images)
                parsed = json.loads(raw)
                if parsed.get("grids_x") or parsed.get("grids_y"):
                    all_results.append(parsed)
                    rprint(
                        f"  [green]Fallback Grid X:{len(parsed.get('grids_x', []))} "
                        f"Y:{len(parsed.get('grids_y', []))}[/]"
                    )
            except Exception as e:
                rprint(f"  [red]Fallback grid extractor error: {e}[/]")

    # ── Elevation pages: up to 2 best pages, 1 API call each ─────────────────
    if elevation_pages:
        top_elevs = select_top_pages(elevation_pages, pdf_path, max_n=4)
        rprint(
            f"[bold green]Spatial Parser:[/] Elevation pages {[p+1 for p in top_elevs]} "
            f"(top {len(top_elevs)} of {len(elevation_pages)})..."
        )
        for pg in top_elevs:
            regions = segment_page_regions(pdf_path, pg)
            try:
                raw    = call_llm_json(elev_prompt, image_parts=[regions[0]])
                parsed = json.loads(raw)
                if parsed.get("levels"):
                    all_results.append(parsed)
                    rprint(f"  [green]p{pg+1} Levels: {[l['name'] for l in parsed.get('levels', [])]}[/]")
            except Exception as e:
                rprint(f"  [red]Elevation parse error p{pg+1}: {e}[/]")

    spatial = _merge_spatial_results(all_results)

    Path(SPATIAL_OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(SPATIAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(spatial, f, indent=2, ensure_ascii=False)

    grids_x = spatial["grids_x"]
    grids_y = spatial["grids_y"]
    levels  = spatial["levels"]
    rprint(f"\n[bold green]Spatial Parser complete.[/] → {SPATIAL_OUTPUT_FILE}")
    rprint(f"  GridX: {len(grids_x)} lines found: {[g['name'] for g in grids_x[:5]]}")
    rprint(f"  GridY: {len(grids_y)} lines found: {[g['name'] for g in grids_y[:5]]}")
    rprint(f"  Levels: {len(levels)} found: {[l['name'] for l in levels]}")
    return spatial