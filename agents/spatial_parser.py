"""
Agent 3 — Spatial Parser  (2-pass grid extraction)
Phase 3: Extract the spatial reference system from Plan and Elevation pages.

  - PLAN views  → 2-pass:
      Pass 1 (symbolic, 100 DPI): list grid bubble labels
      Pass 2 (metric, 300 DPI):   read dimension chains, compute cumulative positions
      Combine + cross-page median consensus
  - ELEVATION views → Z-level system (Base, FL1, Roof, etc.) with heights in mm

FALLBACK RULE: if confidence < 0.5 on ALL pages → grids_x=[], grids_y=[]
  NEVER fabricate uniform spacing.

Output: data/output_json/spatial_data.json  (includes "confidence" field)
"""

import json
import re
import re as _re
import statistics
from pathlib import Path
from rich import print as rprint

from config import SPATIAL_OUTPUT_FILE
from core.llm_wrapper import call_llm_json
from core.pdf_utils import render_page_as_image_part, segment_page_regions, extract_text_from_page
from core.analysis_context import build_plan_context


# ── Pass 1: symbolic label extraction (100 DPI, fast) ───────────────────────
PASS1_LABELS_PROMPT = """You are a Senior Structural Detailer.
This is a structural plan drawing. Examine the BORDERS of the drawing carefully.

{analysis_context}

Grid lines in structural drawings appear as:
- Long dashed or dotted lines running across the full drawing width/height
- CIRCLES or BUBBLES with letters/numbers inside, placed at BOTH ends of each line
  (one at the top/bottom border for horizontal groups, one at left/right for vertical)
- Common conventions: X-axis labels = A, B, C, D... | Y-axis labels = 1, 2, 3, 4...
  OR X-axis = 1,2,3,4... | Y-axis = A,B,C,D... (depends on drawing)
- Vietnamese drawings may use "Trục A", "Trục 1" labels

Task: List ALL grid bubble labels visible on this drawing.

Return JSON ONLY — no prose:
{{
  "grids_x": ["A", "B", "C"],
  "grids_y": ["1", "2", "3"]
}}

Where:
- grids_x = labels running along the TOP or BOTTOM border (vertical grid lines, X coordinates)
- grids_y = labels running along the LEFT or RIGHT border (horizontal grid lines, Y coordinates)

Read labels EXACTLY as they appear. Include ALL visible grid labels in order.
If NO grid bubbles are visible anywhere on this page, return: {{"grids_x": [], "grids_y": []}}"""


# ── Pass 2: metric dimension chain extraction (300 DPI) ──────────────────────
PASS2_METRIC_PROMPT = """You are a Senior Structural Detailer measuring grid dimensions.
This is a structural plan drawing — examine it at full resolution.

{analysis_context}

Task: Find the DIMENSION CHAIN along the drawing edges that shows spacing between grid lines.

A dimension chain looks like one of these patterns:
  ←─6000─→←─7500─→←─6000─→   (with arrows between lines)
  |  6000  |  7500  |  6000  |  (numbers between tick marks)
  6000   7500   6000            (plain numbers along the border)

IMPORTANT — Units:
- Numbers in dimension chains may be in mm (e.g. 6000) or meters (e.g. 6.0 or 6.000).
- If all values are small (< 100), treat as METERS → multiply by 1000 to get mm.
- Most structural drawings use mm already. Do NOT confuse scale bar numbers with grid dims.

Look for:
1. Dimension chain along the BOTTOM or TOP edge → gives X-axis (horizontal) grid spacing
2. Dimension chain along the LEFT or RIGHT edge → gives Y-axis (vertical) grid spacing

Compute CUMULATIVE positions from 0 by summing intervals:
  If intervals_x = [6000, 7500, 6000]  →  positions_x = [0, 6000, 13500, 19500]
  (length of positions_x = length of intervals_x + 1)

Return JSON ONLY:
{{
  "intervals_x": [6000, 7500, 6000],
  "positions_x": [0, 6000, 13500, 19500],
  "intervals_y": [8000, 8000],
  "positions_y": [0, 8000, 16000],
  "scale_detected": "1:100",
  "unit": "mm",
  "confidence": 0.85
}}

Confidence guide:
- 0.9+  : clear dimension chain with arrows, all numbers clearly readable
- 0.7-0.9: numbers visible, minor ambiguity in extent
- 0.5-0.7: partial chain, some numbers estimated
- 0.3-0.5: possible chain, numbers uncertain
- < 0.3  : no clear dimension chain found

If NO dimension chain visible anywhere on this page, return:
{{"intervals_x": [], "positions_x": [], "intervals_y": [], "positions_y": [], "confidence": 0.0}}"""


# ── Elevation prompt (unchanged from original) ───────────────────────────────
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


# ── Page scoring ─────────────────────────────────────────────────────────────

def _score_page(pdf_path: str, page: int) -> int:
    try:
        text = extract_text_from_page(pdf_path, page)
        return len(re.findall(
            r'\b[A-H]\b|\b[1-9]\b|GRID|LEVEL\s*\d|RL\s*[\d.]|FL\d|EL[\d.]'
            r'|FFL|TOS|EAVE|RIDGE|HAUNCH|PARAPET|ROOF|GND|GROUND'
            r'|TẦNG|CỐT|MÁI|TRỤC'
            r'|\b\d{3,5}\b',
            text, re.IGNORECASE,
        ))
    except Exception:
        return 0


def select_best_page(pages: list[int], pdf_path: str) -> int:
    if len(pages) == 1:
        return pages[0]
    return max(pages, key=lambda p: _score_page(pdf_path, p))


def select_top_pages(pages: list[int], pdf_path: str, max_n: int = 2) -> list[int]:
    if len(pages) <= max_n:
        return list(pages)
    scored = sorted(pages, key=lambda p: _score_page(pdf_path, p), reverse=True)
    return scored[:max_n]


# ── 2-pass grid extraction ───────────────────────────────────────────────────

def _run_two_pass_grid(pdf_path: str, plan_pages: list[int], analysis_context: str) -> dict:
    """
    Two-pass grid extraction across up to 3 plan pages.
    Returns: {"grids_x": [...], "grids_y": [...], "confidence": float}
    NEVER fabricates spacing — returns empty lists if confidence < 0.5.
    """
    pass1_prompt = PASS1_LABELS_PROMPT.replace("{analysis_context}", analysis_context)
    pass2_prompt = PASS2_METRIC_PROMPT.replace("{analysis_context}", analysis_context)

    top_pages = select_top_pages(plan_pages, pdf_path, max_n=3)
    page_results = []  # list of per-page dicts

    for pg in top_pages:
        rprint(f"  [bold]Page {pg+1}:[/] Pass 1 — grid labels (100 DPI)...")

        # ── Pass 1: symbolic labels at 100 DPI ──────────────────────────────
        labels_x, labels_y = [], []
        try:
            img_low = render_page_as_image_part(pdf_path, pg, dpi=100)
            raw1 = call_llm_json(pass1_prompt, image_parts=[img_low])
            p1 = json.loads(raw1)
            labels_x = [str(v).strip() for v in p1.get("grids_x", []) if v]
            labels_y = [str(v).strip() for v in p1.get("grids_y", []) if v]
        except Exception as e:
            rprint(f"    [red]Pass 1 error page {pg+1}: {e}[/]")

        rprint(f"    Labels X: {labels_x} | Y: {labels_y}")

        if not labels_x and not labels_y:
            rprint(f"    [yellow]No grid labels found on page {pg+1} — skipping[/]")
            continue

        # ── Pass 2: metric dimension chain at 300 DPI ────────────────────────
        rprint(f"  [bold]Page {pg+1}:[/] Pass 2 — dimension chain (300 DPI)...")
        positions_x, positions_y, confidence = [], [], 0.0
        try:
            img_hi = render_page_as_image_part(pdf_path, pg, dpi=300)
            raw2 = call_llm_json(pass2_prompt, image_parts=[img_hi])
            p2 = json.loads(raw2)
            positions_x = [float(v) for v in p2.get("positions_x", [])]
            positions_y = [float(v) for v in p2.get("positions_y", [])]
            confidence = float(p2.get("confidence", 0.0))
            # Unit auto-correct: if values look like metres, convert to mm
            if positions_x and max(positions_x) < 200:
                rprint(f"    [yellow]X positions look like metres — multiplying by 1000[/]")
                positions_x = [v * 1000 for v in positions_x]
            if positions_y and max(positions_y) < 200:
                rprint(f"    [yellow]Y positions look like metres — multiplying by 1000[/]")
                positions_y = [v * 1000 for v in positions_y]
        except Exception as e:
            rprint(f"    [red]Pass 2 error page {pg+1}: {e}[/]")

        rprint(f"    Positions X: {[int(v) for v in positions_x[:6]]} | "
               f"Y: {[int(v) for v in positions_y[:6]]} | confidence={confidence:.2f}")

        # ── Align labels → positions ─────────────────────────────────────────
        pos_x_dict: dict[str, float] = {}
        pos_y_dict: dict[str, float] = {}

        def _align(labels: list, positions: list, axis: str) -> dict:
            if not labels or not positions:
                return {}
            if len(labels) == len(positions):
                return {labels[i]: positions[i] for i in range(len(labels))}
            # Off-by-one tolerance
            if abs(len(labels) - len(positions)) == 1:
                n = min(len(labels), len(positions))
                rprint(f"    [yellow]{axis} count mismatch ({len(labels)} labels vs "
                       f"{len(positions)} positions) — using {n}[/]")
                return {labels[i]: positions[i] for i in range(n)}
            rprint(f"    [yellow]{axis} count mismatch ({len(labels)} labels vs "
                   f"{len(positions)} positions) — skipping {axis}[/]")
            return {}

        pos_x_dict = _align(labels_x, positions_x, "X")
        pos_y_dict = _align(labels_y, positions_y, "Y")

        # Pass 1 found labels but Pass 2 found no positions → low conf, empty grids
        has_labels = bool(labels_x or labels_y)
        has_positions = bool(pos_x_dict or pos_y_dict)
        if has_labels and not has_positions:
            rprint(f"    [yellow]Grid labels found but no dimension chain readable — grids unknown[/]")
            confidence = min(confidence, 0.3)

        page_results.append({
            "page": pg,
            "labels_x": labels_x,
            "labels_y": labels_y,
            "pos_x_dict": pos_x_dict,
            "pos_y_dict": pos_y_dict,
            "confidence": confidence,
        })

    # ── No pages processed ───────────────────────────────────────────────────
    if not page_results:
        rprint("  [red]2-pass grid extraction: no plan pages produced results[/]")
        return {"grids_x": [], "grids_y": [], "confidence": 0.0}

    max_conf = max(r["confidence"] for r in page_results)
    if max_conf < 0.5:
        rprint(f"  [red]Grid extraction confidence too low on all pages "
               f"(max={max_conf:.2f}) — returning empty grids (no fabrication)[/]")
        # Still log in required format
        rprint(f"[yellow]Grid extraction:[/] confidence={max_conf:.2f} | X: [] | Y: []")
        return {"grids_x": [], "grids_y": [], "confidence": max_conf}

    # ── Cross-page median consensus ──────────────────────────────────────────
    # Only use pages with confidence >= 0.5
    good_results = [r for r in page_results if r["confidence"] >= 0.5]

    x_label_positions: dict[str, list[float]] = {}
    for r in good_results:
        for label, pos in r["pos_x_dict"].items():
            x_label_positions.setdefault(label, []).append(pos)

    y_label_positions: dict[str, list[float]] = {}
    for r in good_results:
        for label, pos in r["pos_y_dict"].items():
            y_label_positions.setdefault(label, []).append(pos)

    grids_x = sorted(
        [{"name": lbl, "x_mm": int(round(statistics.median(vals)))}
         for lbl, vals in x_label_positions.items()],
        key=lambda g: g["x_mm"],
    )
    grids_y = sorted(
        [{"name": lbl, "y_mm": int(round(statistics.median(vals)))}
         for lbl, vals in y_label_positions.items()],
        key=lambda g: g["y_mm"],
    )

    overall_conf = statistics.median([r["confidence"] for r in good_results])

    # ── Required log format ──────────────────────────────────────────────────
    x_str = ", ".join(f"{g['name']}={g['x_mm']}" for g in grids_x[:8])
    y_str = ", ".join(f"{g['name']}={g['y_mm']}" for g in grids_y[:8])
    rprint(f"[green]Grid extraction:[/] confidence={overall_conf:.2f} | "
           f"X: [{x_str}] | Y: [{y_str}]")

    return {"grids_x": grids_x, "grids_y": grids_y, "confidence": overall_conf}


# ── Text layer extraction (levels only — no fake grid) ──────────────────────

def _extract_spatial_from_text(pdf_path: str, plan_pages: list, elev_pages: list) -> dict:
    """
    Extract floor levels from pdfplumber text layer on elevation pages.
    Does NOT attempt to build grids from text — that is the 2-pass job.
    Returns partial spatial dict (levels only).
    """
    result = {"grids_x": [], "grids_y": [], "levels": []}
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            n = len(pdf.pages)

            level_pos = {}
            elev_scale = None

            for pg_i in elev_pages[:4]:
                if pg_i >= n:
                    continue
                page = pdf.pages[pg_i]
                words = page.extract_words()

                for j, w in enumerate(words):
                    if w["text"] in ("1:100", "1:50", "1:200", "1:25"):
                        scale_map = {"1:100": 100, "1:50": 50, "1:200": 200, "1:25": 25}
                        elev_scale = scale_map[w["text"]]
                    if w["text"] == "SCALE:" and j + 2 < len(words):
                        try:
                            elev_scale = int(words[j + 2]["text"])
                        except ValueError:
                            pass

                for j, w in enumerate(words):
                    txt = w["text"].strip().upper()
                    if txt == "LEVEL" and j + 1 < len(words):
                        num_w = words[j + 1]
                        key = f"LEVEL {num_w['text']}"
                        if _re.match(r"^\d{1,2}$", num_w["text"]):
                            if key not in level_pos:
                                level_pos[key] = w["top"]
                    if txt in ("TẦNG", "CỐT", "MÁI"):
                        label = (txt if txt == "MÁI"
                                 else f"{txt} {words[j+1]['text']}" if j + 1 < len(words)
                                 else txt)
                        if label not in level_pos:
                            level_pos[label] = w["top"]

            if level_pos and elev_scale:
                MM_PER_PT = 25.4 / 72
                sorted_lvls = sorted(level_pos.items(), key=lambda x: x[1])
                ref_lvl, ref_y = sorted_lvls[-1]
                for lname, ly in sorted_lvls:
                    dy_pts = ref_y - ly
                    z_mm = round(dy_pts * MM_PER_PT * elev_scale / 50) * 50
                    result["levels"].append({"name": lname, "z_mm": int(z_mm)})
                result["levels"].sort(key=lambda l: l["z_mm"])

    except Exception:
        pass

    return result


# ── Merge helper ─────────────────────────────────────────────────────────────

def _merge_spatial_results(all_results: list[dict]) -> dict:
    """
    Merge elevation results into one unified spatial dataset.
    Grids come exclusively from _run_two_pass_grid() — not from this function.
    """
    merged: dict = {
        "grids_x": [],
        "grids_y": [],
        "levels": [],
    }
    seen_lv: set[str] = set()

    for result in all_results:
        vtype = result.get("view_type", "")
        if vtype in ("ELEVATION", "SECTION"):
            for lv in result.get("levels", []):
                if lv["name"] not in seen_lv:
                    seen_lv.add(lv["name"])
                    merged["levels"].append(lv)

    merged["levels"].sort(key=lambda lv: lv.get("z_mm", 0))

    # Default levels only if elevation pages yielded nothing
    if len(merged["levels"]) <= 1:
        merged["levels"] = [
            {"name": "Base", "z_mm":     0},
            {"name": "FL1",  "z_mm":  3500},
            {"name": "FL2",  "z_mm":  7000},
            {"name": "FL3",  "z_mm": 10500},
            {"name": "Roof", "z_mm": 13500},
        ]

    return merged


# ── Main entry point ─────────────────────────────────────────────────────────

def parse_spatial_pages(
    pdf_path: str,
    plan_pages: list[int],
    elevation_pages: list[int],
) -> dict:
    analysis_context = build_plan_context()
    elev_prompt = ELEVATION_PROMPT.replace("{analysis_context}", analysis_context)

    # ── Text layer: levels from elevation pages (fast, no LLM) ──────────────
    text_result = _extract_spatial_from_text(pdf_path, plan_pages, elevation_pages)
    if text_result["levels"]:
        rprint(f"[green]Text layer:[/] {len(text_result['levels'])} levels extracted via pdfplumber")

    # ── 2-pass grid extraction ───────────────────────────────────────────────
    grid_conf = 0.0
    grids_x: list[dict] = []
    grids_y: list[dict] = []

    if plan_pages:
        rprint(f"\n[bold green]Spatial Parser:[/] 2-pass grid extraction on "
               f"{min(3, len(plan_pages))} of {len(plan_pages)} plan page(s)...")
        grid_result = _run_two_pass_grid(pdf_path, plan_pages, analysis_context)
        grids_x = grid_result["grids_x"]
        grids_y = grid_result["grids_y"]
        grid_conf = grid_result["confidence"]
    else:
        rprint("[yellow]  No plan pages found — grid extraction skipped[/]")

    # ── Elevation pages: LLM vision ──────────────────────────────────────────
    elev_results: list[dict] = []
    if text_result["levels"]:
        elev_results.append(text_result)

    if elevation_pages:
        top_elevs = select_top_pages(elevation_pages, pdf_path, max_n=4)
        rprint(f"[bold green]Spatial Parser:[/] Elevation pages {[p+1 for p in top_elevs]} "
               f"(top {len(top_elevs)} of {len(elevation_pages)})...")
        for pg in top_elevs:
            regions = segment_page_regions(pdf_path, pg)
            try:
                raw = call_llm_json(elev_prompt, image_parts=[regions[0]])
                parsed = json.loads(raw)
                if parsed.get("levels"):
                    elev_results.append(parsed)
                    rprint(f"  [green]p{pg+1} Levels: {[l['name'] for l in parsed.get('levels', [])]}[/]")
            except Exception as e:
                rprint(f"  [red]Elevation parse error p{pg+1}: {e}[/]")

    # ── Merge elevation results ───────────────────────────────────────────────
    merged = _merge_spatial_results(elev_results)
    merged["grids_x"] = grids_x
    merged["grids_y"] = grids_y
    merged["grid_confidence"] = round(grid_conf, 3)

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(SPATIAL_OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(SPATIAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    rprint(f"\n[bold green]Spatial Parser complete.[/] → {SPATIAL_OUTPUT_FILE}")
    rprint(f"  GridX: {len(grids_x)} lines: {[g['name'] for g in grids_x[:6]]}")
    rprint(f"  GridY: {len(grids_y)} lines: {[g['name'] for g in grids_y[:6]]}")
    rprint(f"  Levels: {len(merged['levels'])} found: {[l['name'] for l in merged['levels']]}")
    rprint(f"  Grid confidence: {grid_conf:.2f}")
    return merged
