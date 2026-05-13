"""
Agent 4 — Mapper  (NEW)
Phase 4: Map every member from the Steel Schedule to exact 3D coordinates.

Inputs:
  - steel_schedule.json  (marks + sections)
  - spatial_data.json    (grid X/Y + Z levels)
  - Plan/Elevation page images (for visual confirmation by Gemini)

Strategy:
  1. Give Gemini the schedule + spatial data + plan image.
  2. Ask it to assign each mark to a Start Point (x,y,z) and End Point (x,y,z).
  3. Detect any orphans (marks in schedule not placed on drawings).
  4. Output data/output_json/mapped_members.json

Output: data/output_json/mapped_members.json
"""

import json
from pathlib import Path
from rich import print as rprint

from config import SCHEDULE_OUTPUT_FILE, SPATIAL_OUTPUT_FILE, MAPPED_OUTPUT_FILE
from core.llm_wrapper import call_llm_json
from core.pdf_utils import render_page_as_image_part, segment_page_regions
from core.analysis_context import build_plan_context


MAPPER_PROMPT_TEMPLATE = """You are a BIM Coordinator AI with expert knowledge of structural framing.

{analysis_context}

You have two data sources:
1. STEEL SCHEDULE — lists member marks and section sizes
2. SPATIAL REFERENCE — grid system (X,Y in mm) and floor levels (Z in mm)

Your task: Locate EVERY member on the General Arrangement drawings and assign exact 3D coordinates.

STEEL SCHEDULE:
{schedule_json}

SPATIAL REFERENCE (Grid + Levels):
{spatial_json}

Rules for coordinate assignment:
- COLUMNS: start_point.z = lower level z_mm, end_point.z = upper level z_mm.
  X,Y = intersection of their grid lines (e.g. grid A, row 1 → x=grid_A_x, y=grid_1_y).
- BEAMS: z = the floor level they sit at (both start and end same Z).
  X,Y start/end = grid intersections they span between.
- BRACES: diagonal — start and end at different X,Y,Z.
- rotation_degrees: 0 = web in Y direction (default for beams); 90 = web in Z direction (default for columns).

For each member, return:
{
  "mark": "C1",
  "section": "200UC46",
  "type": "column",
  "start_point": {"x": 0, "y": 0, "z": 0},
  "end_point": {"x": 0, "y": 0, "z": 4500},
  "rotation_degrees": 90,
  "grid_ref": "A-1",
  "level_ref": "Base to FL1",
  "confidence": "high" | "medium" | "low"
}

Return ALL members from the schedule. If you cannot determine position for a member, set:
  start_point: {"x": 0, "y": 0, "z": -9999}  (sentinel for orphan detection)
  confidence: "unmapped"

Return JSON: {"mapped_members": [...]}"""


ORPHAN_RETRY_PROMPT = """You previously could not locate these structural members in the drawings:
{orphan_json}

Re-examine this drawing image carefully. Look for:
- These member marks written anywhere on the drawing (may be small text)
- Member marks in bubbles, callouts, or section cuts
- Members that may be hidden or on a different grid line

For each orphan, try to assign coordinates. If still impossible, return the same sentinel values.
Return JSON: {"mapped_members": [...]}"""


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

                # Collect grid bubble positions
                # Numbers (1,2,3,4) label vertical lines → near top/bottom edge → X matters
                # Letters (A,B,C,D) label horizontal lines → near left/right edge → Y matters
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


def run_mapper(pdf_path: str, plan_pages: list[int], elevation_pages: list[int]) -> list[dict]:
    with open(SCHEDULE_OUTPUT_FILE, "r", encoding="utf-8") as f:
        schedule = json.load(f)
    with open(SPATIAL_OUTPUT_FILE, "r", encoding="utf-8") as f:
        spatial = json.load(f)

    members = schedule.get("members", [])
    schedule_marks = {m["mark"] for m in members if m.get("mark")}

    # Phase 0: text-based member position detection
    _text_located = _locate_members_by_text(pdf_path, plan_pages, members, spatial)
    if _text_located:
        rprint(f"[green]Text locator:[/] {len(_text_located)} member positions found via pdfplumber")

    rprint(f"[bold blue]Mapper:[/] Mapping {len(members)} members to 3D space...")

    # Use first plan + first elevation page as visual context
    image_parts = []
    for pg in (plan_pages[:2] + elevation_pages[:2]):
        try:
            image_parts.append(render_page_as_image_part(pdf_path, pg))
        except Exception:
            pass

    # Debug: show first member to explain why it may be unmapped
    if members:
        first = members[0]
        rprint(f"[dim]Mapper debug — first member sample:[/]")
        rprint(f"  mark={first.get('mark')!r}  type={first.get('type')!r}  "
               f"section={first.get('section')!r}  grid_reference={first.get('grid_reference')!r}")
        if not first.get("grid_reference"):
            rprint(f"  [yellow]→ grid_reference is null — LLM must infer position from drawings alone[/]")

    schedule_json_str = json.dumps({"members": members}, indent=2)
    if len(schedule_json_str) > 4000:
        rprint(f"[yellow]Mapper warning:[/] schedule JSON is {len(schedule_json_str)} chars "
               f"(covering {len(members)} members) — sending full payload to LLM (no truncation).")

    analysis_context = build_plan_context()
    prompt = MAPPER_PROMPT_TEMPLATE.replace(
        "{analysis_context}", analysis_context
    ).replace(
        "{schedule_json}", schedule_json_str
    ).replace(
        "{spatial_json}", json.dumps(spatial, indent=2)
    )

    try:
        raw = call_llm_json(prompt, image_parts=image_parts)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json
                result = json.loads(repair_json(raw))
                rprint("[yellow]  JSON repaired (minor LLM formatting issue)[/]")
            except Exception:
                raise
        mapped = result.get("mapped_members", [])
        rprint(f"[dim]Mapper LLM returned {len(mapped)} members.[/]")
        if mapped:
            first_mapped = mapped[0]
            sp = first_mapped.get("start_point", {})
            rprint(f"  first returned: mark={first_mapped.get('mark')!r}  "
                   f"confidence={first_mapped.get('confidence')!r}  "
                   f"start_point.z={sp.get('z')}  "
                   f"grid_ref={first_mapped.get('grid_ref')!r}")
            if first_mapped.get("confidence") == "unmapped":
                rprint(f"  [yellow]→ LLM marked as unmapped — no grid/level info in drawings or schedule[/]")
    except Exception as e:
        rprint(f"[red]Mapper LLM error: {e}[/]")
        mapped = [_sentinel_member(m) for m in members]

    # ---- Orphan Detection ----
    mapped_marks = {m["mark"] for m in mapped}
    orphan_marks = schedule_marks - mapped_marks
    orphans_by_sentinel = [m for m in mapped if m.get("confidence") == "unmapped"]
    orphan_marks |= {m["mark"] for m in orphans_by_sentinel}

    if orphan_marks:
        rprint(f"[yellow]Orphan detection:[/] {len(orphan_marks)} unplaced members: {orphan_marks}")
        orphan_members = [m for m in members if m.get("mark") in orphan_marks]

        # Retry: feed Gemini the orphans + all images again
        retry_prompt = ORPHAN_RETRY_PROMPT.replace(
            "{orphan_json}", json.dumps(orphan_members, indent=2)
        )
        try:
            raw2 = call_llm_json(retry_prompt, image_parts=image_parts)
            try:
                retry_result = json.loads(raw2)
            except json.JSONDecodeError:
                try:
                    from json_repair import repair_json
                    retry_result = json.loads(repair_json(raw2))
                    rprint("[yellow]  JSON repaired (minor LLM formatting issue)[/]")
                except Exception:
                    raise
            retry_mapped = {m["mark"]: m for m in retry_result.get("mapped_members", [])}
            # Merge: update mapped list with retry results
            for i, m in enumerate(mapped):
                if m["mark"] in retry_mapped:
                    mapped[i] = retry_mapped[m["mark"]]
            # Add any marks that were completely missing
            existing_marks = {m["mark"] for m in mapped}
            for mark, m in retry_mapped.items():
                if mark not in existing_marks:
                    mapped.append(m)
        except Exception as e:
            rprint(f"[red]Orphan retry error: {e}[/]")

    # ---- Final Orphan Pass: sentinel remaining unknowns ----
    final_marks = {m["mark"] for m in mapped}
    still_orphan = schedule_marks - final_marks
    for mark in still_orphan:
        src = next((m for m in members if m["mark"] == mark), {"mark": mark})
        sentinel = _sentinel_member(src)
        mapped.append(sentinel)
        rprint(f"  [red]UNMAPPED:[/] {mark} → placed at origin, layer LOD300_UNMAPPED_NEEDS_REVIEW")

    # ---- Apply text-located X,Y positions (overrides LLM guesses) ----
    if _text_located:
        _tl_levels = sorted(spatial.get("levels", []), key=lambda l: l["z_mm"])
        _tl_base   = _tl_levels[0]["z_mm"] if _tl_levels else 0
        _tl_top    = _tl_levels[-1]["z_mm"] if len(_tl_levels) > 1 else _tl_base + 3500
        _tl_fl1    = _tl_levels[1]["z_mm"] if len(_tl_levels) > 1 else _tl_base + 3500
        _tl_fixed  = 0
        for _ti, _tm in enumerate(mapped):
            _mark = _tm.get("mark")
            if _mark not in _text_located:
                continue
            _loc   = _text_located[_mark]
            _mtype = _tm.get("type", "other")
            _sp    = _tm.get("start_point") or {}
            _good_z = isinstance(_sp.get("z"), (int, float)) and _sp.get("z", -9999) != -9999
            _good_conf = _tm.get("confidence") not in ("unmapped", None)
            if _mtype == "column":
                mapped[_ti].setdefault("start_point", {})["x"] = _loc["x"]
                mapped[_ti]["start_point"]["y"] = _loc["y"]
                if not (_good_z and _good_conf):
                    mapped[_ti]["start_point"]["z"] = _tl_base
                mapped[_ti].setdefault("end_point", {})["x"] = _loc["x"]
                mapped[_ti]["end_point"]["y"] = _loc["y"]
                if not (_good_z and _good_conf):
                    mapped[_ti]["end_point"]["z"] = _tl_top
                mapped[_ti]["rotation_degrees"] = 90
            else:
                _sp_x = _loc["x"]
                _sp_y = _loc["y"]
                mapped[_ti].setdefault("start_point", {})["x"] = _sp_x
                mapped[_ti]["start_point"]["y"] = _sp_y
                if not (_good_z and _good_conf):
                    mapped[_ti]["start_point"]["z"] = _tl_fl1
                    # Span to the next grid X so the member has non-zero length
                    _tl_gx = sorted(spatial.get("grids_x", []), key=lambda g: g["x_mm"])
                    _next_gx = next(
                        (g["x_mm"] for g in _tl_gx if g["x_mm"] > _sp_x),
                        _tl_gx[-1]["x_mm"] - _tl_gx[-2]["x_mm"] if len(_tl_gx) >= 2 else _sp_x + 9000
                    )
                    mapped[_ti].setdefault("end_point", {})["x"] = _next_gx
                    mapped[_ti]["end_point"]["y"] = _sp_y
                    mapped[_ti]["end_point"]["z"] = _tl_fl1
            mapped[_ti]["confidence"] = "high"
            mapped[_ti]["grid_ref"]   = f"text-loc p{_loc['page']+1}"
            _tl_fixed += 1
            rprint(f"  [green]Text override:[/] {_mark} ({_mtype}) → ({_loc['x']},{_loc['y']})")
        if _tl_fixed:
            rprint(f"  [green]Text locator placed {_tl_fixed} member(s) with exact grid positions.[/]")

    # ---- Deterministic grid-reference coordinate fallback ----------------------
    _gx = {g["name"]: g["x_mm"] for g in spatial.get("grids_x", [])}
    _gy = {g["name"]: g["y_mm"] for g in spatial.get("grids_y", [])}
    _levels  = sorted(spatial.get("levels", []), key=lambda l: l.get("z_mm", 0))
    _base_z  = _levels[0]["z_mm"] if _levels else 0
    _top_z   = _levels[-1]["z_mm"] if len(_levels) > 1 else (_base_z + 3500)

    _grid_fixed = 0
    for _i, _m in enumerate(mapped):
        _sp   = _m.get("start_point")
        _conf = _m.get("confidence")
        if _sp and _conf not in ("unmapped", None):
            continue   # already properly placed by LLM
        _ref = _m.get("grid_reference") or _m.get("grid_ref") or ""
        if not _ref or str(_ref).upper() in ("UNMAPPED", "NONE", ""):
            continue
        _coords = _resolve_grid_ref(str(_ref), _gx, _gy)
        if _coords is None:
            continue
        _x1, _y1, _x2, _y2 = _coords
        _mtype = _m.get("type", "other")
        if _mtype == "column":
            # Columns span full building height (Base → Roof)
            _m["start_point"]      = {"x": _x1, "y": _y1, "z": _base_z}
            _m["end_point"]        = {"x": _x1, "y": _y1, "z": _top_z}
            _m["rotation_degrees"] = 90
        elif _mtype == "beam":
            # Beams sit at FL1 (first level above base)
            _beam_z = _levels[1]["z_mm"] if len(_levels) > 1 else (_base_z + 3500)
            _m["start_point"]      = {"x": _x1, "y": _y1, "z": _beam_z}
            _m["end_point"]        = {"x": _x2, "y": _y2, "z": _beam_z}
            _m["rotation_degrees"] = 0
        else:
            # Braces / other: default single-storey placement
            _beam_z = _levels[1]["z_mm"] if len(_levels) > 1 else (_base_z + 3500)
            _m["start_point"]      = {"x": _x1, "y": _y1, "z": _base_z}
            _m["end_point"]        = {"x": _x2, "y": _y2, "z": _beam_z}
            _m["rotation_degrees"] = 0
        _m["confidence"] = "low"
        _m["grid_ref"]   = str(_ref)
        _m["level_ref"]  = f"{_levels[0]['name']} to {_levels[-1]['name']}"
        mapped[_i] = _m
        _grid_fixed += 1
        rprint(f"  [dim]Grid fallback:[/] {_m['mark']} {_ref} → ({_x1},{_y1})→({_x2},{_y2})")

    if _grid_fixed:
        rprint(f"  [green]Grid fallback placed {_grid_fixed} additional member(s).[/]")

    # ---- Coerce all members to have complete 3D coordinates ----
    # Members lacking a valid (x,y,z) start_point get sequentially
    # assigned to grid intersections so SketchUp shows a full building.
    _all_pts = [
        {"x": gx["x_mm"], "y": gy["y_mm"], "ref": f"{gx['name']}-{gy['name']}"}
        for gy in spatial.get("grids_y", [])
        for gx in spatial.get("grids_x", [])
    ]
    _coerce_col_idx   = 0
    _coerce_other_idx = 0
    _coerced_count    = 0

    for _ci, _cm in enumerate(mapped):
        _csp = _cm.get("start_point")
        _is_valid = (
            isinstance(_csp, dict)
            and isinstance(_csp.get("x"), (int, float))
            and isinstance(_csp.get("y"), (int, float))
            and isinstance(_csp.get("z"), (int, float))
            and _cm.get("confidence") not in ("unmapped", None)   # skip sentinel/unresolved members
        )
        if _is_valid:
            continue

        _ctype = _cm.get("type", "other")
        if _ctype == "column":
            _cp = _all_pts[_coerce_col_idx % len(_all_pts)]
            _coerce_col_idx += 1
            _cm["start_point"]      = {"x": _cp["x"], "y": _cp["y"], "z": _base_z}
            _cm["end_point"]        = {"x": _cp["x"], "y": _cp["y"], "z": _top_z}
            _cm["rotation_degrees"] = 90
            _cm["grid_ref"]         = _cp["ref"]
        elif _ctype == "beam":
            _pa = _all_pts[_coerce_other_idx % len(_all_pts)]
            _pb = _all_pts[(_coerce_other_idx + 1) % len(_all_pts)]
            _coerce_other_idx += 1
            _cbz = _levels[1]["z_mm"] if len(_levels) > 1 else (_base_z + 3500)
            _cm["start_point"]      = {"x": _pa["x"], "y": _pa["y"], "z": _cbz}
            _cm["end_point"]        = {"x": _pb["x"], "y": _pb["y"], "z": _cbz}
            _cm["rotation_degrees"] = 0
            _cm["grid_ref"]         = f"{_pa['ref']} to {_pb['ref']}"
        else:
            _cp = _all_pts[_coerce_other_idx % len(_all_pts)]
            _coerce_other_idx += 1
            _cbz = _levels[1]["z_mm"] if len(_levels) > 1 else (_base_z + 3500)
            _cm["start_point"]      = {"x": _cp["x"], "y": _cp["y"], "z": _base_z}
            _cm["end_point"]        = {"x": _cp["x"], "y": _cp["y"], "z": _cbz}
            _cm["rotation_degrees"] = 0
            _cm["grid_ref"]         = _cp["ref"]

        _cm["level_ref"]  = f"{_levels[0]['name']} to {_levels[-1]['name']}"
        _cm["confidence"] = "low"
        mapped[_ci]       = _cm
        _coerced_count    += 1
        rprint(f"  [cyan]Grid coerce:[/] {_cm['mark']} ({_ctype}) → "
               f"({_cm['start_point']['x']},{_cm['start_point']['y']},{_cm['start_point']['z']})")

    if _coerced_count:
        rprint(f"  [cyan]Sequential coercion complete:[/] {_coerced_count} member(s) assigned to grid positions.")

    # Confirm final counts
    unmapped_count = sum(1 for m in mapped if m.get("confidence") == "unmapped")
    rprint(f"\n[bold green]Mapper complete.[/] "
           f"{len(mapped)} total | {len(mapped)-unmapped_count} placed | {unmapped_count} unmapped")

    Path(MAPPED_OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPED_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"mapped_members": mapped}, f, indent=2, ensure_ascii=False)

    rprint(f"Mapped data → {MAPPED_OUTPUT_FILE}")
    return mapped
