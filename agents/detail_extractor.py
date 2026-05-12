"""
Agent 2b — Detail Extractor (fallback). Extracts structural member
information from steelwork detail/section drawing pages when the steel schedule table
is empty or missing.
"""

import json
from pathlib import Path
from rich import print as rprint

from config import SCHEDULE_OUTPUT_FILE
from core.llm_wrapper import call_llm_json
from core.pdf_utils import render_page_fast


DETAIL_EXTRACT_PROMPT = """You are a Senior Structural Engineer reading steelwork detail/section drawings.
These pages show structural steel member cross-sections, connection details,
or framing plans with member callouts.

Extract every distinct structural steel member type you can identify.
For each member return JSON in this exact format:
{
  "mark": "<member identifier e.g. B1, C2, RB1, or section designation if no mark>",
  "type": "<one of: beam | column | brace | purlin | rafter | plate | angle | other>",
  "section": "<section designation e.g. 360UB56.7 or UB36b or 150x90x10EA or 200x10PL>",
  "material_grade": "<e.g. S275, S350, S355, 350, 300PLUS, or null if unknown>",
  "length_mm": <number or null>,
  "quantity": <number or null>,
  "grid_ref": "<grid reference if visible e.g. A-B/1-2, else null>",
  "source": "detail_drawing",
  "confidence": "<high | medium | low>"
}

Return a JSON array of all members found. If a page shows no identifiable
steel members, return an empty array for that page.

Rules:
- Include members shown as cross-section shapes with dimension annotations
- Include members called up with bubble marks on framing plans
- If section size is shown in old Australian format (e.g. UB36b, UC155d, CH35a),
  keep it as-is — do NOT convert
- Do NOT include reinforcement bars (N12, N16, R10 etc) — those are concrete rebar
- Do NOT include mesh types (SL72, SL82, SL92) — those are concrete mesh
- Deduplicate: if same mark appears on multiple pages, include only once"""

_PAGES_PER_BATCH = 4


def _normalise(member: dict, page_num: int) -> dict:
    """Map detail-extractor field names to the standard steel_schedule schema."""
    return {
        "mark":            member.get("mark", "").strip(),
        "type":            member.get("type", "other"),
        "section":         member.get("section", ""),
        "material":        member.get("material_grade"),
        "length_mm":       member.get("length_mm"),
        "quantity":        member.get("quantity"),
        "grid_reference":  member.get("grid_ref"),
        "remarks":         None,
        "page_source":     page_num,
        "source":          "detail_drawing",
        "confidence":      member.get("confidence", "low"),
    }


def extract_from_details(pdf_path: str, page_numbers: list[int]) -> list[dict]:
    """
    Extract structural members from steelwork detail drawing pages.

    Renders each page at 200 DPI, sends batches of up to 4 pages per API call,
    merges and deduplicates results, then saves to SCHEDULE_OUTPUT_FILE.

    Returns list of normalised member dicts (same schema as schedule_parser output).
    """
    if not page_numbers:
        rprint("[yellow]Detail Extractor: no pages supplied — skipping.[/]")
        return []

    all_members: list[dict] = []
    seen_marks: set[str] = set()

    # Split into batches of up to _PAGES_PER_BATCH pages
    batches = [
        page_numbers[i: i + _PAGES_PER_BATCH]
        for i in range(0, len(page_numbers), _PAGES_PER_BATCH)
    ]

    for batch_idx, batch_pages in enumerate(batches):
        rprint(
            f"[bold magenta]Detail Extractor:[/] batch {batch_idx + 1}/{len(batches)} "
            f"— pages {[p + 1 for p in batch_pages]}"
        )

        # Render all pages in this batch
        images = []
        for page_num in batch_pages:
            try:
                img = render_page_fast(pdf_path, page_num, dpi=200)
                images.append((page_num, img))
                rprint(f"  [dim]→ rendered p{page_num + 1} at 200 DPI[/]")
            except Exception as e:
                rprint(f"  [yellow]WARNING: could not render p{page_num + 1}: {e}[/]")

        if not images:
            rprint(f"  [yellow]Batch {batch_idx + 1}: no renderable pages — skipping.[/]")
            continue

        image_parts = [img for _, img in images]
        rendered_pages = [pn for pn, _ in images]

        # Call LLM with all images in one request
        try:
            raw = call_llm_json(DETAIL_EXTRACT_PROMPT, image_parts=image_parts)
            data = json.loads(raw)
        except Exception as e:
            rprint(f"  [red]WARNING: batch {batch_idx + 1} LLM call failed: {e} — skipping batch.[/]")
            continue

        # Unwrap if LLM returned {"members": [...]} instead of a bare array
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break

        if not isinstance(data, list):
            rprint(f"  [yellow]Batch {batch_idx + 1}: unexpected response type {type(data).__name__} — skipping.[/]")
            continue

        # Attribute page_source to the first page in the batch (best available without per-page response)
        batch_first_page = rendered_pages[0]
        batch_new = 0

        for item in data:
            if not isinstance(item, dict):
                continue
            mark = item.get("mark", "").strip()
            if not mark:
                continue
            if mark in seen_marks:
                rprint(f"  [dim]Skipping duplicate mark: {mark}[/]")
                continue
            seen_marks.add(mark)
            all_members.append(_normalise(item, batch_first_page))
            batch_new += 1

        rprint(
            f"  [green]+{batch_new} member(s)[/] from "
            f"page(s) {[p + 1 for p in rendered_pages]}"
        )

    # Save in steel_schedule.json format
    Path(SCHEDULE_OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    schedule = {"total_members": len(all_members), "members": all_members}
    with open(SCHEDULE_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2, ensure_ascii=False)

    rprint(
        f"\n[bold green]Detail Extractor complete.[/] "
        f"{len(all_members)} member(s) from {len(page_numbers)} detail page(s) "
        f"→ {SCHEDULE_OUTPUT_FILE}"
    )
    return all_members
