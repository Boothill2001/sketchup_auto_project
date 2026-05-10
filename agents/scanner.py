"""
Agent 1 — Scanner
Phase 1: Classify every PDF page in PARALLEL to locate:
  - Drawing Index / Table of Contents
  - Steel Schedules (member marks + section sizes)
  - Plan views (Grid X/Y system)
  - Elevation / Section views (Z levels)
  - General Notes / Legend (abbreviation glossary)

Speed optimisations:
  - SCANNER_DPI=100 for classification (no fine detail needed)
  - ThreadPoolExecutor — all pages sent to Gemini concurrently
  - One API key per thread via round-robin in llm_wrapper

Output: data/output_json/drawing_index.json
"""

import json
import time
from pathlib import Path
from rich import print as rprint

from config import SCANNER_OUTPUT_FILE
from core.llm_wrapper import call_llm_json
from core.pdf_utils import get_page_count, render_page_fast


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

_ERROR_PAGE = {
    "type": "parse_error",
    "description": "Classification failed",
    "has_schedule_table": False, "has_member_marks": False,
    "has_grid_lines": False, "has_level_markers": False,
    "has_legend_abbreviations": False,
    "drawing_number": None, "drawing_title": None,
}


def _classify_page(pdf_path: str, page_num: int, total: int) -> tuple[int, dict]:
    """Classify a single page — called from a thread pool worker."""
    try:
        img = render_page_fast(pdf_path, page_num)
        raw = call_llm_json(CLASSIFY_PROMPT, image_parts=[img])
        parsed = json.loads(raw)
    except Exception as e:
        parsed = {**_ERROR_PAGE, "description": str(e)}
    return page_num, parsed


# Sequential exhaustion strategy: key 1 handles ALL calls until RPD-dead.
# 5 RPM limit per project → 1 call per 12s minimum.
# 13s sleep → ~4.6 calls/min → comfortably under limit → zero RPM 429s.
_SLEEP_BETWEEN_PAGES = 13

# Max pages to classify per pipeline run.
# Free tier = 20 RPD multimodal; keep 5 for scanner, rest for other agents.
# Checkpoint saves after every page → resume picks up where this stopped.
PAGES_PER_RUN = 5


def _save_index(index: dict[int, dict]) -> None:
    """Write current index to disk (called after every page for crash-safe resume)."""
    sorted_idx = dict(sorted(index.items()))
    Path(SCANNER_OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(SCANNER_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted_idx, f, indent=2, ensure_ascii=False)


def scan_pdf(pdf_path: str) -> dict:
    total_pages = get_page_count(pdf_path)

    # ── Resume from a previous partial run ──────────────────────────────────
    # Gemini free tier allows only ~20 multimodal RPD — a 27-page PDF cannot be
    # fully classified in one session. We skip pages already successfully done.
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
            pass  # corrupt file — start fresh

    pages_todo = [p for p in range(total_pages) if p not in index]
    if not pages_todo:
        rprint("[green]Scanner: all pages already classified — skipping.[/]")
        return dict(sorted(index.items()))

    # Cap this run to PAGES_PER_RUN; remaining pages stay for the next run
    batch      = pages_todo[:PAGES_PER_RUN]
    remaining  = len(pages_todo) - len(batch)
    rprint(
        f"[bold cyan]Scanner:[/] {total_pages} pages total | "
        f"{len(pages_todo)} remaining | this run: {len(batch)} pages | "
        f"{_SLEEP_BETWEEN_PAGES}s between calls"
    )

    for i, p in enumerate(batch):
        rprint(f"  [dim]-> classifying p{p+1} ({i+1}/{len(batch)})...[/]")
        page_num, result = _classify_page(pdf_path, p, total_pages)
        index[page_num] = result
        ptype = result.get("type", "?")
        desc  = result.get("description", "")[:60]
        rprint(
            f"  [cyan][{len(index):>2}/{total_pages}][/] "
            f"p{page_num+1:>2} -> [green]{ptype}[/]  {desc}"
        )
        _save_index(index)   # checkpoint — safe to stop and resume tomorrow
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

    rprint(f"  Index -> {SCANNER_OUTPUT_FILE}")
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
    idx = scan_pdf(pdf)
    roles = get_pages_by_role(idx)
    for role, pages in roles.items():
        rprint(f"  [yellow]{role}:[/] {pages}")
