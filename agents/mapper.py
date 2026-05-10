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


MAPPER_PROMPT_TEMPLATE = """You are a BIM Coordinator AI with expert knowledge of structural framing.

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


def run_mapper(pdf_path: str, plan_pages: list[int], elevation_pages: list[int]) -> list[dict]:
    with open(SCHEDULE_OUTPUT_FILE, "r", encoding="utf-8") as f:
        schedule = json.load(f)
    with open(SPATIAL_OUTPUT_FILE, "r", encoding="utf-8") as f:
        spatial = json.load(f)

    members = schedule.get("members", [])
    schedule_marks = {m["mark"] for m in members if m.get("mark")}

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

    prompt = MAPPER_PROMPT_TEMPLATE.replace(
        "{schedule_json}", schedule_json_str
    ).replace(
        "{spatial_json}", json.dumps(spatial, indent=2)
    )

    try:
        raw = call_llm_json(prompt, image_parts=image_parts)
        result = json.loads(raw)
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
            retry_result = json.loads(raw2)
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

    # Confirm final counts
    unmapped_count = sum(1 for m in mapped if m.get("confidence") == "unmapped")
    rprint(f"\n[bold green]Mapper complete.[/] "
           f"{len(mapped)} total | {len(mapped)-unmapped_count} placed | {unmapped_count} unmapped")

    Path(MAPPED_OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(MAPPED_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"mapped_members": mapped}, f, indent=2, ensure_ascii=False)

    rprint(f"Mapped data → {MAPPED_OUTPUT_FILE}")
    return mapped
