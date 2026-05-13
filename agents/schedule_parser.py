"""
Agent 2 — Schedule Parser
Phase 2: Extract member marks and section sizes from Steel Schedule pages.
Uses the Glossary and PDF Analysis context to adapt to ANY convention.
Output: data/output_json/steel_schedule.json
"""

import json
from pathlib import Path
from rich import print as rprint

from config import SCHEDULE_OUTPUT_FILE, GLOSSARY_OUTPUT_FILE, USE_TEXT_EXTRACTION_FIRST
from core.llm_wrapper import call_llm_json
from core.pdf_utils import render_page_as_image_part, segment_page_regions, get_page_count, extract_tables_pdfplumber
from core.analysis_context import build_schedule_context


EXTRACT_PROMPT_TEMPLATE = """You are a Senior Structural Detailer extracting data from a structural steel schedule.

{analysis_context}

PROJECT GLOSSARY (use this to normalise abbreviations):
{glossary_json}

Extract EVERY structural member visible on this page.
For each member return:
- "mark": exact label as written (e.g. "B1", "C-1", "COL-2", "G3")
- "type": "beam" | "column" | "slab" | "brace" | "plate" | "angle" | "other"
- "section": section designation EXACTLY as written (e.g. "UB305x165x40", "200UC46", "RHS150x100x6")
- "material": grade string (e.g. "S275", "S355", "Grade 250") — use glossary default if not shown
- "length_mm": length in mm as integer — null if not on schedule
- "quantity": count as integer — null if not shown
- "grid_reference": grid/level reference if shown (e.g. "A1-B1", "Level 1") — null if absent
- "remarks": any notes (camber, splice, fire rating, etc.) — null if none

Return JSON:
{
  "page_source": <page_number_int>,
  "members": [ { ...member... }, ... ]
}

If no members found: {{"page_source": <n>, "members": []}}"""


def load_glossary() -> dict:
    if not Path(GLOSSARY_OUTPUT_FILE).exists():
        return {"abbreviations": {}, "material_grades": {"default_steel": "S275"}}
    with open(GLOSSARY_OUTPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_schedule_pages(pdf_path: str, schedule_pages: list[int]) -> list[dict]:
    if not schedule_pages:
        total = get_page_count(pdf_path)
        schedule_pages = list(range(total))
        rprint(
            f"[yellow]Schedule Parser:[/] No schedule pages classified by scanner — "
            f"falling back to all {total} pages (full scan)."
        )

    # ── Load PDF analysis context (adapts to convention) ────────────────────
    analysis_context = build_schedule_context()

    glossary = load_glossary()
    glossary_summary = json.dumps({
        "abbreviations": glossary.get("abbreviations", {}),
        "default_material": glossary.get("material_grades", {}).get("default_steel", "S275"),
    }, indent=2)

    all_members: list[dict] = []
    seen_marks: set[str] = set()

    for page_num in schedule_pages:
        rprint(f"[bold magenta]Schedule Parser:[/] Extracting page {page_num + 1}...")
        base_prompt = EXTRACT_PROMPT_TEMPLATE.replace("{analysis_context}", analysis_context)
        base_prompt = base_prompt.replace("{glossary_json}", glossary_summary)

        # ── Text-first path ──────────────────────────────────────────────────
        text_data = extract_tables_pdfplumber(pdf_path, page_num) if USE_TEXT_EXTRACTION_FIRST else None

        if text_data:
            rprint(f"  [dim]Using text extraction for page {page_num+1} (saved 1 vision call)[/]")
            text_prompt = (
                base_prompt
                + f"\n\nSTEEL SCHEDULE TABLE (CSV from PDF text layer, page {page_num+1}):\n"
                + text_data
            )
            try:
                raw = call_llm_json(text_prompt)  # no image_parts — text only
                parsed = json.loads(raw)
                members = parsed.get("members", [])
                new_members = []
                for m in members:
                    mark = m.get("mark", "").strip()
                    if mark and mark not in seen_marks:
                        seen_marks.add(mark)
                        m["page_source"] = page_num
                        new_members.append(m)
                    elif mark in seen_marks:
                        rprint(f"  [dim]Skipping duplicate mark: {mark}[/]")
                all_members.extend(new_members)
                rprint(f"  [green]+{len(new_members)} members[/] from page {page_num+1} (text-extracted)")
                continue  # skip vision extraction for this page
            except Exception as e:
                rprint(f"  [yellow]Text extraction failed ({e}), falling back to vision...[/]")

        # ── Vision fallback ──────────────────────────────────────────────────
        rprint(f"  [dim]Page {page_num+1} is scanned image, using vision extraction[/]")
        regions = segment_page_regions(pdf_path, page_num)
        for region_idx, image_part in enumerate(regions):
            label = f"page {page_num+1}" + (f" region {region_idx+1}" if len(regions) > 1 else "")
            try:
                raw = call_llm_json(base_prompt, image_parts=[image_part])
                parsed = json.loads(raw)
                members = parsed.get("members", [])
                new_members = []
                for m in members:
                    mark = m.get("mark", "").strip()
                    if mark and mark not in seen_marks:
                        seen_marks.add(mark)
                        m["page_source"] = page_num
                        new_members.append(m)
                    elif mark in seen_marks:
                        rprint(f"  [dim]Skipping duplicate mark: {mark}[/]")
                all_members.extend(new_members)
                rprint(f"  [green]+{len(new_members)} members[/] from {label}")
            except Exception as e:
                rprint(f"  [red]Error on {label}: {e}[/]")

    Path(SCHEDULE_OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    schedule = {"total_members": len(all_members), "members": all_members}
    with open(SCHEDULE_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2, ensure_ascii=False)

    rprint(f"\n[bold green]Schedule Parser complete.[/] {len(all_members)} members → {SCHEDULE_OUTPUT_FILE}")
    return all_members


if __name__ == "__main__":
    import sys
    from agents.scanner import scan_pdf, get_pages_by_role
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/input_pdf/structural.pdf"
    roles = get_pages_by_role(scan_pdf(pdf))
    parse_schedule_pages(pdf, roles["schedule"])