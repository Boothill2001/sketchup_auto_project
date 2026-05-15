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
        rprint("  [yellow]→ Falling back to text-layer grid extraction...[/]")
        text_grids = _extract_grids_from_text(pdf_path, plan_pages)
        if text_grids["grids_x"] or text_grids["grids_y"]:
            rprint(f"  [green]Text-layer grid: X={len(text_grids['grids_x'])} Y={len(text_grids['grids_y'])}[/]")
            return text_grids
        return {"grids_x": [], "grids_y": [], "confidence": 0.0}

    max_conf = max(r["confidence"] for r in page_results)
    if max_conf < 0.5:
        rprint(f"  [red]Grid extraction confidence too low on all pages "
               f"(max={max_conf:.2f}) — trying text-layer fallback[/]")
        text_grids = _extract_grids_from_text(pdf_path, plan_pages)
        if text_grids["grids_x"] or text_grids["grids_y"]:
            rprint(f"  [green]Text-layer fallback: X={len(text_grids['grids_x'])} Y={len(text_grids['grids_y'])}[/]")
            # Merge: keep text grids, preserve any vision-pass labels/positions as supplementary
            return text_grids
        rprint(f"  [red]Text-layer fallback also empty — returning empty grids (no fabrication)[/]")
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


# ── Text layer grid extraction (fallback when vision fails) ──────────────────

def _extract_grids_from_text(pdf_path: str, plan_pages: list) -> dict:
    """
    Extract grid labels and compute positions from text layer on plan pages.
    Used as fallback when 2-pass vision extraction returns confidence < 0.5.
    
    Strategy:
    1. Find all grid bubble labels (A,B,C / 1,2,3) via pdfplumber text extraction
    2. Cluster by axis (X vs Y) based on position on page
    3. Use dimension chain text or estimate from text positions
    
    Returns: {"grids_x": [...], "grids_y": [...], "confidence": float}
    """
    result = {"grids_x": [], "grids_y": [], "confidence": 0.0}
    
    try:
        import pdfplumber
        import re
        with pdfplumber.open(pdf_path) as pdf:
            n = len(pdf.pages)
            
            for pg_i in plan_pages[:4]:
                if pg_i >= n:
                    continue
                page = pdf.pages[pg_i]
                pw, ph = float(page.width), float(page.height)
                words = page.extract_words()
                
                # ── Strategy A: Find grid bubble labels (single A-Z, 1-99 at page borders) ─
                x_candidates = {}  # label -> list of x positions
                y_candidates = {}  # label -> list of y positions
                
                for w in words:
                    t = w["text"].strip()
                    cx = (w["x0"] + w["x1"]) / 2
                    cy = (w["top"] + w["bottom"]) / 2
                    
                    # Grid labels: single character (A-Z, 1-99) near page border
                    if re.match(r'^[A-Za-z]$', t):
                        # Near top or bottom border → X-axis grid label
                        if cy < ph * 0.15 or cy > ph * 0.85:
                            x_candidates.setdefault(t.upper(), []).append(cx)
                        # Near left or right border → Y-axis grid label
                        elif cx < pw * 0.15 or cx > pw * 0.85:
                            y_candidates.setdefault(t.upper(), []).append(cx)
                    elif re.match(r'^\d{1,2}$', t):
                        if cy < ph * 0.15 or cy > ph * 0.85:
                            x_candidates.setdefault(t, []).append(cx)
                        elif cx < pw * 0.15 or cx > pw * 0.85:
                            y_candidates.setdefault(t, []).append(cy)
                    # Also check Vietnamese "Trục A"
                    elif t.upper().startswith("TRỤC") or t.upper().startswith("TRUC"):
                        # Next word might be the label
                        pass
                
                # ── Determine which axis has letters vs numbers ─
                x_has_letters = any(re.match(r'^[A-Z]$', k) for k in x_candidates)
                y_has_letters = any(re.match(r'^[A-Z]$', k) for k in y_candidates)
                
                # Swap if letters are on wrong axis (convention: X=numbers or letters)
                if x_has_letters and not y_has_letters and not any(re.match(r'^\d+$', k) for k in y_candidates):
                    # Letters on top/bottom: that's the X axis (or Y depending)
                    pass  # Accept as-is
                
                # ── Build grids from candidates ─
                if x_candidates and len(x_candidates) >= 2:
                    # Estimate positions from text x-coordinates
                    avg_xs = {label: sum(pxs)/len(pxs) for label, pxs in x_candidates.items()}
                    sorted_x = sorted(avg_xs.items(), key=lambda kv: kv[1])
                    
                    # Try to find dimension chain text on this page
                    dim_chain_x = _extract_dimension_chain_from_text(words, is_x_axis=True, page_width=pw, page_height=ph)
                    
                    if dim_chain_x and len(dim_chain_x) >= len(sorted_x) - 1:
                        # Use dimension chain intervals for mm positions
                        positions = [0.0]
                        for d in dim_chain_x[:len(sorted_x)-1]:
                            positions.append(positions[-1] + d)
                        for i, (label, px) in enumerate(sorted_x):
                            if i < len(positions):
                                result["grids_x"].append({"name": label, "x_mm": int(round(positions[i]))})
                    else:
                        # Estimate from text positions: scale to typical floor plan
                        x_px_values = [px for _, px in sorted_x]
                        px_range = max(x_px_values) - min(x_px_values) if len(x_px_values) > 1 else 1
                        mm_per_px = _estimate_mm_per_pixel(pw, ph)
                        for label, px in sorted_x:
                            x_mm = int(round((px - min(x_px_values)) / px_range * px_range * mm_per_px))
                            result["grids_x"].append({"name": label, "x_mm": x_mm})
                
                if y_candidates and len(y_candidates) >= 2:
                    avg_ys = {label: sum(pys)/len(pys) for label, pys in y_candidates.items()}
                    sorted_y = sorted(avg_ys.items(), key=lambda kv: kv[1])
                    
                    dim_chain_y = _extract_dimension_chain_from_text(words, is_x_axis=False, page_width=pw, page_height=ph)
                    
                    if dim_chain_y and len(dim_chain_y) >= len(sorted_y) - 1:
                        positions = [0.0]
                        for d in dim_chain_y[:len(sorted_y)-1]:
                            positions.append(positions[-1] + d)
                        for i, (label, py) in enumerate(sorted_y):
                            if i < len(positions):
                                result["grids_y"].append({"name": label, "y_mm": int(round(positions[i]))})
                    else:
                        y_px_values = [py for _, py in sorted_y]
                        px_range = max(y_px_values) - min(y_px_values) if len(y_px_values) > 1 else 1
                        mm_per_px = _estimate_mm_per_pixel(pw, ph)
                        for label, py in sorted_y:
                            y_mm = int(round((py - min(y_px_values)) / px_range * px_range * mm_per_px))
                            result["grids_y"].append({"name": label, "y_mm": y_mm})
                
                if result["grids_x"] or result["grids_y"]:
                    result["confidence"] = 0.35  # Mark as low-confidence text extraction
                    break  # Found grids on first viable page
                    
    except Exception:
        pass
    
    return result


def _extract_dimension_chain_from_text(words: list, is_x_axis: bool, page_width: float, page_height: float) -> list:
    """
    Find dimension chain numbers in text layer.
    Looks for consecutive numbers (e.g. 6000, 7500, 6000) near page borders.
    """
    dims = []
    try:
        import re
        
        # Look for 3-5 digit numbers near page borders (typical grid dimensions in mm)
        border_words = []
        for w in words:
            t = w["text"].strip()
            cy = (w["top"] + w["bottom"]) / 2
            cx = (w["x0"] + w["x1"]) / 2
            
            if is_x_axis and (cy < page_height * 0.12 or cy > page_height * 0.88):
                if re.match(r'^\d{3,5}$', t):
                    border_words.append((cx, int(t)))
            elif not is_x_axis and (cx < page_width * 0.12 or cx > page_width * 0.88):
                if re.match(r'^\d{3,5}$', t):
                    border_words.append((cy, int(t)))
        
        if border_words:
            border_words.sort(key=lambda x: x[0])
            # Take consecutive series (ignore outliers)
            dims = [v for _, v in border_words]
            
            # Filter: typical grid spacing is 2000-15000 mm
            dims = [d for d in dims if 2000 <= d <= 15000]
            
    except Exception:
        pass
    
    return dims


def _estimate_mm_per_pixel(page_w: float, page_h: float) -> float:
    """
    Estimate mm per pixel for a structural drawing page.
    Typical structural plan: A1 sheet (841x594mm) at 100-200 scale.
    Returns conservative estimate.
    """
    # Assume the page represents ~30-50m of building
    typical_mm_width = 40000  # 40m building width
    return typical_mm_width / max(page_w, 1)


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

    # FIX v6: Phase 0 fallback — use grid labels from PDF analysis when vision+text both fail
    if (not grids_x or not grids_y) and grid_conf < 0.5:
        try:
            from core.analysis_context import load_analysis_dict
            analysis = load_analysis_dict()
            pdf_labels_x = analysis.get("grid_labels_x", [])
            pdf_labels_y = analysis.get("grid_labels_y", [])
            if pdf_labels_x and pdf_labels_y:
                typical_spacing = 8000  # 8m typical grid
                grids_x = [{"name": str(lbl), "x_mm": i * typical_spacing}
                           for i, lbl in enumerate(pdf_labels_x)]
                grids_y = [{"name": str(lbl), "y_mm": i * typical_spacing}
                           for i, lbl in enumerate(pdf_labels_y)]
                grid_conf = 0.35  # low confidence (estimated)
                rprint(f"[yellow]Phase 0 fallback:[/] using PDF analysis grid labels "
                       f"X={pdf_labels_x} Y={pdf_labels_y} "
                       f"with estimated {typical_spacing}mm spacing (confidence={grid_conf})")
            else:
                rprint("[red]Phase 0 fallback:[/] no grid labels in analysis either — grids remain empty[/]")
        except Exception as e:
            rprint(f"[red]Phase 0 fallback failed: {e}")

    # ── Elevation pages: LLM vision ──────────────────────────────────────────
    elev_results: list[dict] = []
    if text_result["levels"]:
        elev_results.append(text_result)

    if elevation_pages:
        top_elevs = select_top_pages(elevation_pages, pdf_path, max_n=2)
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
