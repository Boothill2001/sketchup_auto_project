"""
Agent 1 — Scanner
Phase 1: Classify every PDF page to locate:
  - Drawing Index / Table of Contents
  - Steel Schedules (member marks + section sizes)
  - Plan views (Grid X/Y system)
  - Elevation / Section views (Z levels)
  - General Notes / Legend (abbreviation glossary)

Speed optimisations (new):
  - Composite thumbnail grid: ALL pages classified in 1-2 API calls
  - 72 DPI thumbnails — enough for layout-level classification
  - Fallback to sequential per-page if composite parse fails

Output: data/output_json/drawing_index.json
"""

import json
import math
import time
from pathlib import Path
from rich import print as rprint

from config import SCANNER_OUTPUT_FILE
from core.llm_wrapper import call_llm_json
from core.pdf_utils import get_page_count, render_page_fast, build_thumbnail_grid


# ── Per-page prompt (fallback) ────────────────────────────────────────────────

CLASSIFY_PROMPT = """You are a Senior Structural Detailer reviewing an engineering drawing sheet.
Classify this page and extract key metadata.

Return JSON with this exact schema:
{
  "type": "<one of: drawing_index | steel_schedule | plan_view | elevation_view | section_view | glossary | general_note | connection_detail | other>",
  "description": "<one sentence: key content on this page>",
  "has_schedule_table": <true|false>,
  "has_member_marks": <true|false>,
  "has_grid_lines": <true|false>,
  "has_level_markers": <true|false>,
  "has_legend_abbreviations": <true|false>,
  "drawing_number": "<drawing number if visible, else null>",
  "drawing_title": "<title from title block if visible, else null>"
}

Priority rules (apply the FIRST matching rule):
1. Table with columns MARK / SIZE / SECTION / QTY / LENGTH → steel_schedule (even if the page also has details)
2. List of abbreviations, symbols, or material specifications table → glossary
3. Grid bubbles (A,B,C / 1,2,3) with dimension chains and no vertical elevation markers → plan_view
4. Vertical section showing floor levels / datum triangles / RL values with heights → elevation_view
5. Cross-section cut through structural elements (slabs, walls, beams) → section_view
6. Detail drawings of bolted/welded connections, splice plates, base plates → connection_detail
7. Drawing number index, sheet list, or table of contents → drawing_index
8. General text notes / specification clauses → general_note"""


# ── Composite grid prompt ─────────────────────────────────────────────────────

CLASSIFY_GRID_PROMPT = """You are looking at a grid of PDF page thumbnails from a structural engineering drawing set.
Each thumbnail is labeled P0, P1, P2... in the top-left corner — the number is the 0-indexed PDF page number.

Classify EVERY thumbnail visible in the grid.

Return a JSON array with exactly one object per thumbnail:
[
  {"page_index": 0, "role": "plan_view", "confidence": "high"},
  {"page_index": 1, "role": "steel_schedule", "confidence": "high"},
  ...
]

Valid roles:
  drawing_index     — sheet list or table of contents
  steel_schedule    — table with MARK / SIZE / SECTION / QTY columns
  plan_view         — floor plan with grid bubbles (A,B,C / 1,2,3) and dimension chains
  elevation_view    — vertical view showing floor levels / datum triangles / RL values
  section_view      — cross-section cut through structural elements
  glossary          — list of abbreviations, symbols, or material specs
  general_note      — dense text paragraphs, specification clauses
  connection_detail — bolted/welded connections, splice plates, base plates
  other             — anything not matching above

For thumbnails too small to read clearly, infer from layout patterns:
  dense table rows          → steel_schedule
  grid lines + dim bubbles  → plan_view
  vertical stacking + levels → elevation_view
  dense text paragraphs     → general_note

The page_index in each response object MUST match the number shown on the thumbnail label
(e.g. thumbnail labeled "P5" → page_index: 5).
Return exactly one entry per thumbnail — no gaps, no duplicates."""


# ── Role → index-entry mapping ────────────────────────────────────────────────

_ROLE_FLAGS: dict[str, dict] = {
    "steel_schedule":    {"has_schedule_table": True,  "has_member_marks": True,  "has_grid_lines": False, "has_level_markers": False, "has_legend_abbreviations": False},
    "plan_view":         {"has_schedule_table": False, "has_member_marks": False, "has_grid_lines": True,  "has_level_markers": False, "has_legend_abbreviations": False},
    "elevation_view":    {"has_schedule_table": False, "has_member_marks": False, "has_grid_lines": False, "has_level_markers": True,  "has_legend_abbreviations": False},
    "section_view":      {"has_schedule_table": False, "has_member_marks": False, "has_grid_lines": False, "has_level_markers": True,  "has_legend_abbreviations": False},
    "glossary":          {"has_schedule_table": False, "has_member_marks": False, "has_grid_lines": False, "has_level_markers": False, "has_legend_abbreviations": True},
    "general_note":      {"has_schedule_table": False, "has_member_marks": False, "has_grid_lines": False, "has_level_markers": False, "has_legend_abbreviations": True},
    "drawing_index":     {"has_schedule_table": False, "has_member_marks": False, "has_grid_lines": False, "has_level_markers": False, "has_legend_abbreviations": False},
    "connection_detail": {"has_schedule_table": False, "has_member_marks": False, "has_grid_lines": False, "has_level_markers": False, "has_legend_abbreviations": False},
    "other":             {"has_schedule_table": False, "has_member_marks": False, "has_grid_lines": False, "has_level_markers": False, "has_legend_abbreviations": False},
}

_ERROR_PAGE = {
    "type": "parse_error",
    "description": "Classification failed",
    "has_schedule_table": False, "has_member_marks": False,
    "has_grid_lines": False, "has_level_markers": False,
    "has_legend_abbreviations": False,
    "drawing_number": None, "drawing_title": None,
}


def _role_to_index_entry(role: str) -> dict:
    flags = _ROLE_FLAGS.get(role, _ROLE_FLAGS["other"])
    return {
        "type": role,
        "description": f"Classified as {role} via composite grid scan",
        "drawing_number": None,
        "drawing_title": None,
        **flags,
    }


def _parse_grid_response(raw_json: str, expected_pages: list[int]) -> dict[int, dict]:
    """Parse LLM grid classification response → {page_num: index_entry}."""
    data = json.loads(raw_json)

    # Unwrap if LLM wrapped the array in an object
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                data = v
                break

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    valid_pages = set(expected_pages)
    result: dict[int, dict] = {}

    for item in data:
        if not isinstance(item, dict):
            continue
        raw_idx = item.get("page_index")
        role    = item.get("role", "other")
        if role not in _ROLE_FLAGS:
            role = "other"
        if raw_idx is not None and int(raw_idx) in valid_pages:
            result[int(raw_idx)] = _role_to_index_entry(role)

    # Mark any pages the LLM missed as "other"
    for p in expected_pages:
        if p not in result:
            result[p] = _role_to_index_entry("other")

    return result


# ── Sequential fallback (original per-page logic) ─────────────────────────────

_SLEEP_BETWEEN_PAGES = 13
PAGES_PER_RUN = 5


def _classify_page(pdf_path: str, page_num: int, total: int) -> tuple[int, dict]:
    try:
        img = render_page_fast(pdf_path, page_num)
        raw = call_llm_json(CLASSIFY_PROMPT, image_parts=[img])
        parsed = json.loads(raw)
    except Exception as e:
        parsed = {**_ERROR_PAGE, "description": str(e)}
    return page_num, parsed


def _scan_pages_sequential(
    pdf_path: str,
    pages_todo: list[int],
    index: dict,
    total_pages: int,
) -> None:
    """Classify pages one at a time — used as fallback when composite grid fails."""
    batch     = pages_todo[:PAGES_PER_RUN]
    remaining = len(pages_todo) - len(batch)

    for i, p in enumerate(batch):
        rprint(f"  [dim]→ classifying p{p+1} ({i+1}/{len(batch)})...[/]")
        page_num, result = _classify_page(pdf_path, p, total_pages)
        index[page_num] = result
        ptype = result.get("type", "?")
        desc  = result.get("description", "")[:60]
        rprint(
            f"  [cyan][{len(index):>2}/{total_pages}][/] "
            f"p{page_num+1:>2} → [green]{ptype}[/]  {desc}"
        )
        _save_index(index)
        if i < len(batch) - 1:
            rprint(f"  [dim]  sleeping {_SLEEP_BETWEEN_PAGES}s (RPM guard)...[/]")
            time.sleep(_SLEEP_BETWEEN_PAGES)

    if remaining > 0:
        rprint(
            f"\n[yellow]Scanner batch done.[/] {len(batch)} pages classified this run. "
            f"{remaining} pages still pending — re-run pipeline to continue."
        )
    else:
        rprint(f"\n[bold green]Scanner complete.[/] All {total_pages} pages classified.")


# ── Shared helpers ────────────────────────────────────────────────────────────

def _save_index(index: dict[int, dict]) -> None:
    sorted_idx = dict(sorted(index.items()))
    Path(SCANNER_OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(SCANNER_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted_idx, f, indent=2, ensure_ascii=False)


_MAX_PAGES_PER_GRID = 36


# ── Main entry point ─────────────────────────────────────────────────────────

def scan_pdf(pdf_path: str) -> dict:
    total_pages = get_page_count(pdf_path)

    # ── Resume from a previous partial run ───────────────────────────────────
    index: dict[int, dict] = {}
    if Path(SCANNER_OUTPUT_FILE).exists():
        try:
            with open(SCANNER_OUTPUT_FILE, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k, v in raw.items():
                if v.get("type") not in (None, "parse_error"):
                    index[int(k)] = v
            if index:
                rprint(f"[dim]Scanner: resuming — {len(index)}/{total_pages} pages already classified.[/]")
        except Exception:
            pass

    pages_todo = [p for p in range(total_pages) if p not in index]
    if not pages_todo:
        rprint("[green]Scanner: all pages already classified — skipping.[/]")
        return dict(sorted(index.items()))

    n_grids = math.ceil(len(pages_todo) / _MAX_PAGES_PER_GRID)
    rprint(
        f"[bold cyan]Scanner:[/] {total_pages} pages total | "
        f"{len(pages_todo)} to classify | "
        f"composite grid ({n_grids} API call{'s' if n_grids > 1 else ''})"
    )

    # ── Composite grid approach ───────────────────────────────────────────────
    try:
        grids = build_thumbnail_grid(
            pdf_path, dpi=72, max_per_grid=_MAX_PAGES_PER_GRID, pages=pages_todo
        )
        total_classified = 0

        for grid_idx, (grid_img, page_nums) in enumerate(grids):
            rprint(
                f"  [dim]→ grid {grid_idx+1}/{len(grids)}: "
                f"pages {page_nums[0]}–{page_nums[-1]} ({len(page_nums)} thumbnails)...[/]"
            )
            raw         = call_llm_json(CLASSIFY_GRID_PROMPT, image_parts=[grid_img])
            grid_result = _parse_grid_response(raw, page_nums)
            index.update(grid_result)
            total_classified += len(grid_result)

            for page_num in sorted(grid_result):
                ptype = grid_result[page_num].get("type", "?")
                rprint(
                    f"  [cyan][{page_num+1:>2}/{total_pages}][/] "
                    f"p{page_num+1:>2} → [green]{ptype}[/]"
                )

        _save_index(index)
        rprint(
            f"\n[bold green]Scanner:[/] classified {total_classified} pages "
            f"in {len(grids)} API call{'s' if len(grids) > 1 else ''} (composite grid)"
        )

    except Exception as e:
        rprint(f"  [yellow]Composite grid failed ({e}), falling back to per-page...[/]")
        _scan_pages_sequential(pdf_path, pages_todo, index, total_pages)

    rprint(f"  Index → {SCANNER_OUTPUT_FILE}")
    return dict(sorted(index.items()))


def get_pages_by_role(index: dict) -> dict[str, list[int]]:
    """
    Split the index into role buckets consumed by downstream agents.
    Returns: { "schedule": [...], "plan": [...], "elevation": [...], "glossary": [...] }
    """
    roles: dict[str, list[int]] = {
        "schedule": [], "plan": [], "elevation": [], "glossary": []
    }
    for page_num, info in index.items():
        ptype        = info.get("type", "")
        has_schedule = info.get("has_schedule_table") or info.get("has_member_marks")
        has_grid     = info.get("has_grid_lines")
        has_levels   = info.get("has_level_markers")
        has_legend   = info.get("has_legend_abbreviations")

        if ptype == "steel_schedule" or has_schedule:
            roles["schedule"].append(int(page_num))
        if ptype == "plan_view" or has_grid:
            roles["plan"].append(int(page_num))
        if ptype in ("elevation_view", "section_view") or has_levels:
            roles["elevation"].append(int(page_num))
        if ptype in ("general_note", "glossary") or has_legend:
            roles["glossary"].append(int(page_num))

    for role in roles:
        roles[role] = sorted(set(roles[role]))

    return roles


if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/input_pdf/structural.pdf"
    idx   = scan_pdf(pdf)
    roles = get_pages_by_role(idx)
    for role, pages in roles.items():
        rprint(f"  [yellow]{role}:[/] {pages}")
