"""
Agent 2 — Parser
Phase 2: Extract technical Schedules from Source-of-Truth pages.

Extracts structured data:
  - Steel member marks (e.g. B1, C1)
  - Section sizes (e.g. 200x100x8x12, UB305x165x40)
  - Material grades (e.g. S275, S355)
  - Member lengths and quantities

Output: data/output_json/steel_schedule.json
"""

import json
from pathlib import Path
from rich import print as rprint

from config import SCANNER_OUTPUT_FILE, PARSER_OUTPUT_FILE
from core.llm_wrapper import call_llm_json
from core.pdf_utils import render_page_as_image_part


EXTRACT_PROMPT = """You are a structural engineer extracting data from a drawing sheet.
Extract ALL structural member information visible on this page.

For each member found, return an object with:
- "mark": member label (e.g. "B1", "C2", "G3")
- "type": "beam" | "column" | "slab" | "brace" | "plate" | "other"
- "section": section size string exactly as written (e.g. "UB305x165x40", "200x200x8x12", "RHS150x100x6")
- "material": material grade (e.g. "S275", "S355", "Grade 250") — null if not shown
- "length_mm": length in mm as a number — null if not shown
- "quantity": count — null if not shown
- "remarks": any notes (e.g. "splice at 3000", "camber 15mm")

Respond with JSON:
{
  "page_source": <page_number_int>,
  "members": [ { ...member object... }, ... ]
}

If no structural members are found, return {"page_source": <n>, "members": []}."""


def parse_pages(pdf_path: str, pages_to_parse: list[int]) -> list[dict]:
    """
    Run extraction on the given page numbers.
    Returns a flat list of member dicts with page_source attached.
    """
    all_members = []

    for page_num in pages_to_parse:
        rprint(f"[bold magenta]Parser:[/] Extracting page {page_num + 1}...")
        image_part = render_page_as_image_part(pdf_path, page_num)
        prompt = EXTRACT_PROMPT.replace("<page_number_int>", str(page_num))

        try:
            raw = call_llm_json(prompt, image_parts=[image_part])
            parsed = json.loads(raw)
            members = parsed.get("members", [])
            for m in members:
                m["page_source"] = page_num
            all_members.extend(members)
            rprint(f"  [green]Found {len(members)} member(s)[/]")
        except Exception as e:
            rprint(f"  [red]Error on page {page_num}: {e}[/]")

    return all_members


def save_schedule(members: list[dict]) -> None:
    schedule = {
        "total_members": len(members),
        "members": members,
    }
    Path(PARSER_OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(PARSER_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(schedule, f, indent=2, ensure_ascii=False)
    rprint(f"\n[bold green]Parser complete.[/] {len(members)} members saved → {PARSER_OUTPUT_FILE}")


def load_scanner_index() -> dict:
    with open(SCANNER_OUTPUT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import sys
    from agents.scanner import get_source_of_truth_pages

    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/input_pdf/structural.pdf"
    index = load_scanner_index()
    sot_pages = list(get_source_of_truth_pages(index).keys())
    rprint(f"[yellow]Parsing pages:[/] {sot_pages}")
    members = parse_pages(pdf, sot_pages)
    save_schedule(members)
