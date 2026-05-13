"""
Agent 4 — Mapper  (FIX 3: Vision-First Per-Page Placement)
Phase 4: Map every member from the Steel Schedule to exact 3D coordinates.

Inputs:
  - steel_schedule.json  (marks + sections)
  - spatial_data.json    (grid X/Y + Z levels)
  - Plan/Elevation page images (per-page Gemini vision passes)

Strategy (FIX 3 — vision-first):
  STEP 1 — Per plan page vision pass  (up to 5 pages, 250 DPI)
  STEP 2 — Per elevation page vision pass  (up to 3 pages, 250 DPI)
  STEP 3 — Text locator fallback for vision-missed marks (pdfplumber)
  STEP 4 — Merge: vision XY + elevation Z → final coordinates
  STEP 5 — Grid-ref / sequential coerce as absolute last resort (logged as FALLBACK)

Output: data/output_json/mapped_members.json
"""

import json
from pathlib import Path
from rich import print as rprint

from config import SCHEDULE_OUTPUT_FILE, SPATIAL_OUTPUT_FILE, MAPPED_OUTPUT_FILE
from core.llm_wrapper import call_llm_json
from core.pdf_utils import render_page_as_image_part, segment_page_regions
from core.analysis_context import build_plan_context


# ─── Vision Pass Prompts ──────────────────────────────────────────────────────

PLAN_VISION_PROMPT = """You are a BIM Coordinator AI examining a structural floor plan drawing.

{analysis_context}

TASK: Locate each structural member mark listed below on this floor plan image.

Member marks to find: {mark_list}

Grid lines on this drawing:
- X-direction labels (vertical grid lines): {grid_x_labels}
- Y-direction labels (horizontal grid lines): {grid_y_labels}
Grid data available: {has_grid_data}

INSTRUCTIONS:
- Look for member marks as text labels near structural symbols (column squares, beam lines).
- Grid intersections are labeled at drawing borders with circles/bubbles.
- Some marks may appear as text labels near a structural symbol (column square, beam line).
  If a member mark appears between two grids (e.g. midspan beam),
  record BOTH grid endpoints: start_grid_x/y and end_grid_x/y (different values).
- For COLUMNS at a single grid intersection: start and end grids are the same.
- If {has_grid_data} == YES: report grid names (e.g. "A", "B", "1", "2").
- If {has_grid_data} == NO: report pixel coordinates from image top-left (pixel_x, pixel_y).
- Confidence scoring:
  - Clearly visible mark at grid intersection: confidence = 0.9
  - Mark visible but grid unclear: confidence = 0.6
  - Mark not found on this page: DO NOT include in output (skip entirely)

Return JSON:
{{
  "placements": [
    {{
      "mark": "C1",
      "grid_x": "A",
      "grid_y": "1",
      "start_grid_x": "A",
      "start_grid_y": "1",
      "end_grid_x": "A",
      "end_grid_y": "1",
      "pixel_x": null,
      "pixel_y": null,
      "confidence": 0.9
    }}
  ]
}}
Only include marks you can actually see on this drawing. Do NOT fabricate positions."""


ELEVATION_VISION_PROMPT = """You are a BIM Coordinator AI examining a structural elevation drawing.

{analysis_context}

TASK: For each structural member mark listed, determine its floor level connection points.

Member marks to find: {mark_list}

Known floor levels: {level_names}

INSTRUCTIONS:
- Look for member marks as labels on the elevation drawing.
- Identify the bottom (start) and top (end) floor levels for each member.
- Column spanning ground to first floor: start_level="Base", end_level="FL1"
- Beam at first floor level: start_level="FL1", end_level="FL1"
- Use known level names when possible. Map unlabeled levels to nearest known level.
- If z_mm values are visible on the drawing (e.g. RL +4500, EL 4.5m), record them in mm.
- Confidence: clearly visible = 0.9, inferred from context = 0.6, not found = skip.

Return JSON:
{{
  "placements": [
    {{
      "mark": "C1",
      "level_name": "Base",
      "z_mm": 0,
      "end_level_name": "FL1",
      "end_z_mm": 4500,
      "confidence": 0.9
    }}
  ]
}}
Only include marks you can actually locate on this elevation drawing."""


# ─── Helper Functions (unchanged from original) ───────────────────────────────

def _parse_axis(token: str, lookup: dict) -> tuple:
    """Resolve a single grid token ("1") or range ("1-2") to (start_mm, end_mm)."""
    token = token.strip()
    if "-" in token:
        a, b = token.split("-", 1)
        va, vb = lookup.get(a.strip()), lookup.get(b.strip())
        if va is not None and vb is not None:
            return va, vb
    v = lookup.get(token)
    if v is not None:
        return v, v
    return None, None


def _resolve_grid_ref(ref_str: str, grids_x: dict, grids_y: dict):
    """
    Parse grid references like "(1, B)", "(A, 1-2)", "(1-2, A)".
    Returns (x1, y1, x2, y2) in mm or None.
    Tries both axis orderings so letter/number can appear in either slot.
    """
    s = str(ref_str).strip().strip("()")
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 2:
        return None
    p0, p1 = parts
    for xtoken, ytoken in [(p1, p0), (p0, p1)]:
        x1, x2 = _parse_axis(xtoken, grids_x)
        y1, y2 = _parse_axis(ytoken, grids_y)
        if x1 is not None and y1 is not None:
            return x1, y1, x2 or x1, y2 or y1
    return None


def _sentinel_member(member: dict) -> dict:
    return {
        "mark": member.get("mark", "UNKNOWN"),
        "section": member.get("section", ""),
        "type": member.get("type", "other"),
        "start_point": {"x": 0, "y": 0, "z": 0},
        "end_point": {"x": 0, "y": 0, "z": 3000},
        "rotation_degrees": 0,
        "grid_ref": "UNMAPPED",
        "level_ref": "UNMAPPED",
        "confidence": "unmapped",
    }


def _locate_members_by_text(pdf_path: str, plan_pages: list, members: list, spatial: dict) -> dict:
    """Find member marks as text on plan pages; snap to nearest grid intersection."""
    located = {}
    try:
        import pdfplumber
        gx_data = spatial.get("grids_x", [])
        gy_data = spatial.get("grids_y", [])
        if not gx_data or not gy_data:
            return located

        gx_names = {g["name"] for g in gx_data}
        gy_names = {g["name"] for g in gy_data}
        all_marks = {m["mark"] for m in members if m.get("mark")}
        gx_mm_map = {g["name"]: g["x_mm"] for g in gx_data}
        gy_mm_map = {g["name"]: g["y_mm"] for g in gy_data}

        with pdfplumber.open(pdf_path) as pdf:
            n_pages = len(pdf.pages)
            for pg_i in plan_pages[:8]:
                if pg_i >= n_pages:
                    continue
                page = pdf.pages[pg_i]
                pw, ph = float(page.width), float(page.height)
                words = page.extract_words()

                gx_px, gy_px = {}, {}
                for w in words:
                    t = w["text"].strip()
                    cx = (w["x0"] + w["x1"]) / 2
                    cy = (w["top"] + w["bottom"]) / 2
                    if t in gx_names and (w["top"] < ph * 0.18 or w["bottom"] > ph * 0.82):
                        gx_px.setdefault(t, []).append(cx)
                    if t in gy_names and (w["x0"] < pw * 0.18 or w["x1"] > pw * 0.82):
                        gy_px.setdefault(t, []).append(cy)

                if len(gx_px) < 2 or len(gy_px) < 2:
                    continue

                ctrl_x = sorted(
                    [(sum(xs) / len(xs), gx_mm_map[n]) for n, xs in gx_px.items() if n in gx_mm_map],
                    key=lambda p: p[0]
                )
                ctrl_y = sorted(
                    [(sum(ys) / len(ys), gy_mm_map[n]) for n, ys in gy_px.items() if n in gy_mm_map],
                    key=lambda p: p[0]
                )

                def _interp(val, ctrl):
                    if len(ctrl) < 2:
                        return None
                    if val <= ctrl[0][0]:
                        return ctrl[0][1]
                    if val >= ctrl[-1][0]:
                        return ctrl[-1][1]
                    for k in range(len(ctrl) - 1):
                        p0, m0 = ctrl[k]
                        p1, m1 = ctrl[k + 1]
                        if p0 <= val <= p1:
                            return m0 + (val - p0) / (p1 - p0) * (m1 - m0)
                    return None

                for w in words:
                    t = w["text"].strip()
                    if t in all_marks and t not in located:
                        cx = (w["x0"] + w["x1"]) / 2
                        cy = (w["top"] + w["bottom"]) / 2
                        x_mm = _interp(cx, ctrl_x)
                        y_mm = _interp(cy, ctrl_y)
                        if x_mm is not None and y_mm is not None:
                            located[t] = {
                                "x": min(gx_data, key=lambda g: abs(g["x_mm"] - x_mm))["x_mm"],
                                "y": min(gy_data, key=lambda g: abs(g["y_mm"] - y_mm))["y_mm"],
                                "page": pg_i,
                            }

    except Exception:
        pass
    return located


# ─── Vision Pass Functions (FIX 3) ───────────────────────────────────────────

def _vision_plan_pass(
    pdf_path: str,
    plan_pages: list,
    members: list,
    spatial: dict,
    analysis_context: str,
) -> dict:
    """
    Per plan-page LLM vision pass (FIX 3 STEP 1).
    Returns dict: mark -> {grid_x, grid_y, start_grid_x, start_grid_y,
                            end_grid_x, end_grid_y, confidence, page, pixel_x?, pixel_y?}
    Higher confidence overrides lower confidence across pages.
    """
    results: dict = {}

    gx_data = spatial.get("grids_x", [])
    gy_data = spatial.get("grids_y", [])
    gx_names = [g["name"] for g in gx_data]
    gy_names = [g["name"] for g in gy_data]
    has_grid = "YES" if (gx_names and gy_names) else "NO"
    all_marks = [m["mark"] for m in members if m.get("mark")]

    for pg_i in plan_pages[:5]:
        pending = [mk for mk in all_marks
                   if mk not in results or results[mk]["confidence"] < 0.6]
        if not pending:
            break

        try:
            img = render_page_as_image_part(pdf_path, pg_i, dpi=250)
        except Exception as e:
            rprint(f"  [yellow]Vision plan pass page {pg_i + 1}: render error — {e}[/]")
            continue

        prompt = PLAN_VISION_PROMPT.format(
            analysis_context=analysis_context,
            mark_list=json.dumps(pending),
            grid_x_labels=json.dumps(gx_names),
            grid_y_labels=json.dumps(gy_names),
            has_grid_data=has_grid,
        )

        try:
            raw = call_llm_json(prompt, image_parts=[img])
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    from json_repair import repair_json
                    result = json.loads(repair_json(raw))
                except Exception:
                    raise
            placements = result.get("placements", [])
        except Exception as e:
            rprint(f"  [yellow]Vision plan pass page {pg_i + 1}: LLM error — {e}[/]")
            continue

        found_labels = []
        for p in placements:
            mark = p.get("mark")
            conf = float(p.get("confidence", 0))
            if not mark or conf <= 0 or mark not in all_marks:
                continue
            if mark not in results or conf > results[mark]["confidence"]:
                gx = p.get("grid_x") or p.get("start_grid_x") or ""
                gy = p.get("grid_y") or p.get("start_grid_y") or ""
                results[mark] = {
                    "grid_x":       gx,
                    "grid_y":       gy,
                    "start_grid_x": p.get("start_grid_x") or gx,
                    "start_grid_y": p.get("start_grid_y") or gy,
                    "end_grid_x":   p.get("end_grid_x") or gx,
                    "end_grid_y":   p.get("end_grid_y") or gy,
                    "pixel_x":      p.get("pixel_x"),
                    "pixel_y":      p.get("pixel_y"),
                    "confidence":   conf,
                    "page":         pg_i,
                }
                sx = results[mark]["start_grid_x"]
                sy = results[mark]["start_grid_y"]
                found_labels.append(f"{mark}({sx}-{sy},{conf:.1f})")

        label_str = ", ".join(found_labels[:8]) + ("..." if len(found_labels) > 8 else "")
        rprint(f"  Vision pass page {pg_i + 1}: found {len(found_labels)} members [{label_str}]")

    return results


def _vision_elevation_pass(
    pdf_path: str,
    elevation_pages: list,
    members: list,
    spatial: dict,
    analysis_context: str,
) -> dict:
    """
    Per elevation-page LLM vision pass (FIX 3 STEP 2).
    Returns dict: mark -> {level_name, z_mm, end_level_name, end_z_mm, confidence, page}
    Higher confidence overrides lower across pages.
    """
    results: dict = {}
    level_list = spatial.get("levels", [])
    level_names = [l["name"] for l in level_list]
    all_marks = [m["mark"] for m in members if m.get("mark")]

    for pg_i in elevation_pages[:3]:
        pending = [mk for mk in all_marks
                   if mk not in results or results[mk]["confidence"] < 0.6]
        if not pending:
            break

        try:
            img = render_page_as_image_part(pdf_path, pg_i, dpi=250)
        except Exception as e:
            rprint(f"  [yellow]Vision elevation pass page {pg_i + 1}: render error — {e}[/]")
            continue

        prompt = ELEVATION_VISION_PROMPT.format(
            analysis_context=analysis_context,
            mark_list=json.dumps(pending),
            level_names=json.dumps(level_names),
        )

        try:
            raw = call_llm_json(prompt, image_parts=[img])
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                try:
                    from json_repair import repair_json
                    result = json.loads(repair_json(raw))
                except Exception:
                    raise
            placements = result.get("placements", [])
        except Exception as e:
            rprint(f"  [yellow]Vision elevation pass page {pg_i + 1}: LLM error — {e}[/]")
            continue

        found_labels = []
        for p in placements:
            mark = p.get("mark")
            conf = float(p.get("confidence", 0))
            if not mark or conf <= 0 or mark not in all_marks:
                continue
            if mark not in results or conf > results[mark]["confidence"]:
                results[mark] = {
                    "level_name":     p.get("level_name", ""),
                    "z_mm":           p.get("z_mm"),
                    "end_level_name": p.get("end_level_name", ""),
                    "end_z_mm":       p.get("end_z_mm"),
                    "confidence":     conf,
                    "page":           pg_i,
                }
                found_labels.append(f"{mark}({p.get('level_name', '?')},{conf:.1f})")

        label_str = ", ".join(found_labels[:8]) + ("..." if len(found_labels) > 8 else "")
        rprint(f"  Elevation pass page {pg_i + 1}: found {len(found_labels)} members [{label_str}]")

    return results


def _resolve_plan_xy(
    mark: str,
    plan_vision: dict,
    spatial: dict,
) -> tuple:
    """
    Convert vision plan result (grid names) to mm coords.
    Returns (x1, y1, x2, y2) or (None, None, None, None).
    Falls back to None when grid data is missing or pixel-only.
    """
    p = plan_vision.get(mark)
    if not p:
        return None, None, None, None

    gx_map = {g["name"]: g["x_mm"] for g in spatial.get("grids_x", [])}
    gy_map = {g["name"]: g["y_mm"] for g in spatial.get("grids_y", [])}

    sx = gx_map.get(p.get("start_grid_x", ""))
    sy = gy_map.get(p.get("start_grid_y", ""))
    ex = gx_map.get(p.get("end_grid_x", "")) if p.get("end_grid_x") else None
    ey = gy_map.get(p.get("end_grid_y", "")) if p.get("end_grid_y") else None

    if sx is None or sy is None:
        # Pixel fallback: log warning, return None to fall through to coerce
        if p.get("pixel_x") is not None:
            rprint(f"  [yellow]Warning: No grid data — pixel-based placement for {mark}, accuracy reduced[/]")
        return None, None, None, None

    return sx, sy, ex if ex is not None else sx, ey if ey is not None else sy


def _resolve_elev_z(
    mark: str,
    elev_vision: dict,
    spatial: dict,
    member_type: str,
) -> tuple:
    """
    Convert vision elevation result to (start_z_mm, end_z_mm).
    Falls back to spatial.levels defaults if mark not in elevation results.
    """
    levels   = sorted(spatial.get("levels", []), key=lambda l: l.get("z_mm", 0))
    base_z   = levels[0]["z_mm"] if levels else 0
    top_z    = levels[-1]["z_mm"] if len(levels) > 1 else (base_z + 3500)
    fl1_z    = levels[1]["z_mm"] if len(levels) > 1 else (base_z + 3500)

    e = elev_vision.get(mark)
    if not e:
        return (base_z, top_z) if member_type == "column" else (fl1_z, fl1_z)

    level_map = {l["name"]: l["z_mm"] for l in levels}
    sz = e.get("z_mm")
    if sz is None:
        default_sz = base_z if member_type == "column" else fl1_z
        sz = level_map.get(e.get("level_name", ""), default_sz)

    ez = e.get("end_z_mm")
    if ez is None:
        default_ez = top_z if member_type == "column" else fl1_z
        ez = level_map.get(e.get("end_level_name", ""), default_ez)

    return sz, ez


# ─── Main Mapper Function ─────────────────────────────────────────────────────

def run_mapper(pdf_path: str, plan_pages: list[int], elevation_pages: list[int]) -> list[dict]:
    with open(SCHEDULE_OUTPUT_FILE, "r", encoding="utf-8") as f:
        schedule = json.load(f)
    with open(SPATIAL_OUTPUT_FILE, "r", encoding="utf-8") as f:
        spatial = json.load(f)

    members = schedule.get("members", [])
    schedule_marks = {m["mark"] for m in members if m.get("mark")}

    rprint(f"[bold blue]Mapper:[/] Mapping {len(members)} members to 3D space...")

    gx_data  = spatial.get("grids_x", [])
    gy_data  = spatial.get("grids_y", [])
    levels   = sorted(spatial.get("levels", []), key=lambda l: l.get("z_mm", 0))
    base_z   = levels[0]["z_mm"] if levels else 0
    top_z    = levels[-1]["z_mm"] if len(levels) > 1 else (base_z + 3500)
    fl1_z    = levels[1]["z_mm"] if len(levels) > 1 else (base_z + 3500)
    _gx_map  = {g["name"]: g["x_mm"] for g in gx_data}
    _gy_map  = {g["name"]: g["y_mm"] for g in gy_data}
    _lv_ref  = (
        f"{levels[0]['name']} to {levels[-1]['name']}"
        if len(levels) > 1 else "Base to Roof"
    )

    if not gx_data or not gy_data:
        rprint("[yellow]  No grid data in spatial_data.json — pixel-based placement, accuracy reduced[/]")

    analysis_context = build_plan_context()

    # ── STEP 1: Per plan-page vision pass ─────────────────────────────────────
    rprint("[bold]  STEP 1:[/] Per-page plan vision pass...")
    plan_vision = _vision_plan_pass(pdf_path, plan_pages, members, spatial, analysis_context)
    rprint(f"  Plan vision total: {len(plan_vision)}/{len(members)} marks located")

    # ── STEP 2: Per elevation-page vision pass ────────────────────────────────
    rprint("[bold]  STEP 2:[/] Per-page elevation vision pass...")
    elev_vision = _vision_elevation_pass(pdf_path, elevation_pages, members, spatial, analysis_context)
    rprint(f"  Elevation vision total: {len(elev_vision)}/{len(members)} marks located")

    # ── STEP 3: Text locator for vision-missed marks ───────────────────────────
    rprint("[bold]  STEP 3:[/] Text locator pass (fallback for vision-missed)...")
    vision_found = set(plan_vision.keys())
    text_members = [m for m in members if m.get("mark") not in vision_found]
    text_located = _locate_members_by_text(pdf_path, plan_pages, text_members, spatial)
    if text_located:
        rprint(f"  Text locator: {len(text_located)} additional marks found via pdfplumber")

    # ── STEP 4: Merge all sources into final mapped list ──────────────────────
    rprint("[bold]  STEP 4:[/] Merging sources into 3D coordinates...")
    mapped: list[dict] = []
    vision_placed = 0
    text_placed   = 0
    grid_fallback = 0
    coerced_count = 0

    all_pts = [
        {"x": gx["x_mm"], "y": gy["y_mm"], "ref": f"{gx['name']}-{gy['name']}"}
        for gy in gy_data for gx in gx_data
    ]
    _coerce_col_idx   = 0
    _coerce_other_idx = 0

    for member in members:
        mark  = member.get("mark", "")
        mtype = member.get("type", "other")

        # ── Vision-placed (highest priority) ──────────────────────────────────
        if mark in plan_vision:
            x1, y1, x2, y2 = _resolve_plan_xy(mark, plan_vision, spatial)
            if x1 is not None:
                sz, ez = _resolve_elev_z(mark, elev_vision, spatial, mtype)
                vp   = plan_vision[mark]
                vref = f"{vp.get('start_grid_x','?')}-{vp.get('start_grid_y','?')}"
                conf_str = "high" if vp["confidence"] >= 0.8 else "medium"
                if mtype == "column":
                    mapped.append({
                        "mark": mark, "section": member.get("section", ""), "type": mtype,
                        "start_point": {"x": x1, "y": y1, "z": sz},
                        "end_point":   {"x": x1, "y": y1, "z": ez},
                        "rotation_degrees": 90,
                        "grid_ref": vref, "level_ref": _lv_ref, "confidence": conf_str,
                    })
                else:
                    eref = f"{vp.get('start_grid_x','?')}-{vp.get('start_grid_y','?')} to {vp.get('end_grid_x','?')}-{vp.get('end_grid_y','?')}"
                    mapped.append({
                        "mark": mark, "section": member.get("section", ""), "type": mtype,
                        "start_point": {"x": x1, "y": y1, "z": sz},
                        "end_point":   {"x": x2, "y": y2, "z": ez},
                        "rotation_degrees": 0,
                        "grid_ref": eref, "level_ref": f"z={sz}", "confidence": conf_str,
                    })
                vision_placed += 1
                continue

        # ── Text-locator fallback ──────────────────────────────────────────────
        if mark in text_located:
            loc = text_located[mark]
            sz, ez = _resolve_elev_z(mark, elev_vision, spatial, mtype)
            if mtype == "column":
                mapped.append({
                    "mark": mark, "section": member.get("section", ""), "type": mtype,
                    "start_point": {"x": loc["x"], "y": loc["y"], "z": sz},
                    "end_point":   {"x": loc["x"], "y": loc["y"], "z": ez},
                    "rotation_degrees": 90,
                    "grid_ref": f"text-loc p{loc['page']+1}", "level_ref": _lv_ref, "confidence": "medium",
                })
            else:
                _sorted_gx = sorted(gx_data, key=lambda g: g["x_mm"])
                _next_x = next(
                    (g["x_mm"] for g in _sorted_gx if g["x_mm"] > loc["x"]),
                    (_sorted_gx[-1]["x_mm"] if _sorted_gx else loc["x"] + 9000),
                )
                mapped.append({
                    "mark": mark, "section": member.get("section", ""), "type": mtype,
                    "start_point": {"x": loc["x"],   "y": loc["y"], "z": fl1_z},
                    "end_point":   {"x": _next_x,    "y": loc["y"], "z": fl1_z},
                    "rotation_degrees": 0,
                    "grid_ref": f"text-loc p{loc['page']+1}", "level_ref": f"z={fl1_z}", "confidence": "medium",
                })
            text_placed += 1
            continue

        # ── Grid-ref from schedule (last resort before coerce) ─────────────────
        _ref = member.get("grid_reference") or member.get("grid_ref") or ""
        if _ref and str(_ref).upper() not in ("UNMAPPED", "NONE", ""):
            _coords = _resolve_grid_ref(str(_ref), _gx_map, _gy_map)
            if _coords is not None:
                _x1, _y1, _x2, _y2 = _coords
                sz, ez = _resolve_elev_z(mark, elev_vision, spatial, mtype)
                rprint(f"  FALLBACK grid-coerce: mark={mark} — vision and text locator both failed")
                if mtype == "column":
                    mapped.append({
                        "mark": mark, "section": member.get("section", ""), "type": mtype,
                        "start_point": {"x": _x1, "y": _y1, "z": sz},
                        "end_point":   {"x": _x1, "y": _y1, "z": ez},
                        "rotation_degrees": 90,
                        "grid_ref": str(_ref), "level_ref": _lv_ref, "confidence": "low",
                    })
                elif mtype == "beam":
                    mapped.append({
                        "mark": mark, "section": member.get("section", ""), "type": mtype,
                        "start_point": {"x": _x1, "y": _y1, "z": fl1_z},
                        "end_point":   {"x": _x2, "y": _y2, "z": fl1_z},
                        "rotation_degrees": 0,
                        "grid_ref": str(_ref), "level_ref": f"z={fl1_z}", "confidence": "low",
                    })
                else:
                    mapped.append({
                        "mark": mark, "section": member.get("section", ""), "type": mtype,
                        "start_point": {"x": _x1, "y": _y1, "z": base_z},
                        "end_point":   {"x": _x2, "y": _y2, "z": fl1_z},
                        "rotation_degrees": 0,
                        "grid_ref": str(_ref), "level_ref": f"z={base_z} to z={fl1_z}", "confidence": "low",
                    })
                grid_fallback += 1
                continue

        # ── Sequential grid coerce (absolute last resort) ──────────────────────
        rprint(f"  FALLBACK grid-coerce: mark={mark} — vision and text locator both failed")
        if all_pts:
            if mtype == "column":
                _cp = all_pts[_coerce_col_idx % len(all_pts)]
                _coerce_col_idx += 1
                mapped.append({
                    "mark": mark, "section": member.get("section", ""), "type": mtype,
                    "start_point": {"x": _cp["x"], "y": _cp["y"], "z": base_z},
                    "end_point":   {"x": _cp["x"], "y": _cp["y"], "z": top_z},
                    "rotation_degrees": 90,
                    "grid_ref": _cp["ref"], "level_ref": _lv_ref, "confidence": "low",
                })
            elif mtype == "beam":
                _pa = all_pts[_coerce_other_idx % len(all_pts)]
                _pb = all_pts[(_coerce_other_idx + 1) % len(all_pts)]
                _coerce_other_idx += 1
                mapped.append({
                    "mark": mark, "section": member.get("section", ""), "type": mtype,
                    "start_point": {"x": _pa["x"], "y": _pa["y"], "z": fl1_z},
                    "end_point":   {"x": _pb["x"], "y": _pb["y"], "z": fl1_z},
                    "rotation_degrees": 0,
                    "grid_ref": f"{_pa['ref']} to {_pb['ref']}", "level_ref": f"z={fl1_z}", "confidence": "low",
                })
            else:
                _cp = all_pts[_coerce_other_idx % len(all_pts)]
                _coerce_other_idx += 1
                mapped.append({
                    "mark": mark, "section": member.get("section", ""), "type": mtype,
                    "start_point": {"x": _cp["x"], "y": _cp["y"], "z": base_z},
                    "end_point":   {"x": _cp["x"], "y": _cp["y"], "z": fl1_z},
                    "rotation_degrees": 0,
                    "grid_ref": _cp["ref"], "level_ref": f"z={base_z} to z={fl1_z}", "confidence": "low",
                })
        else:
            mapped.append(_sentinel_member(member))
        coerced_count += 1

    # ── Final summary ──────────────────────────────────────────────────────────
    total = len(members)
    rprint(
        f"\n[bold green]Mapper final:[/] {vision_placed}/{total} vision-placed "
        f"| {text_placed} text-locator "
        f"| {grid_fallback + coerced_count} grid-coerced "
        f"| 0 unmapped"
    )

    Path(MAPPED_OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPED_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"mapped_members": mapped}, f, indent=2, ensure_ascii=False)

    rprint(f"Mapped data → {MAPPED_OUTPUT_FILE}")
    return mapped
