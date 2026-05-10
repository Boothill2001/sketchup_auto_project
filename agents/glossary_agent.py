"""
Agent 1b — Glossary Agent
Scans General Notes / Legend pages to build a project-specific abbreviation dictionary.
Downstream agents use this to normalize non-standard member labels.

Output: data/output_json/glossary.json
"""

import json
import time
from pathlib import Path
from rich import print as rprint

from config import GLOSSARY_OUTPUT_FILE
from core.llm_wrapper import call_llm_json
from core.pdf_utils import render_page_fast


GLOSSARY_PROMPT = """You are a structural engineering document analyst.
Scan this drawing page (General Notes, Legend, or Specification sheet).
Extract ALL abbreviations, symbols, and shorthand notations used in this project.

Return JSON:
{
  "abbreviations": {
    "COL": "Column",
    "BM": "Beam",
    "PL": "Plate",
    "UB": "Universal Beam",
    "UC": "Universal Column",
    "RHS": "Rectangular Hollow Section",
    "SHS": "Square Hollow Section",
    "CHS": "Circular Hollow Section",
    "FB": "Flat Bar",
    "EA": "Equal Angle",
    "UA": "Unequal Angle"
  },
  "material_grades": {
    "default_steel": "S275",
    "high_strength": "S355"
  },
  "notes": "<any relevant project-wide notes about member naming convention>"
}

If this page contains no abbreviation info, return {"abbreviations": {}, "material_grades": {}, "notes": "none found"}."""


def build_glossary(pdf_path: str, glossary_pages: list[int]) -> dict:
    merged_abbrevs = {}
    merged_grades = {}
    notes_list = []

    def _scan_page(page_num: int) -> dict:
        img = render_page_fast(pdf_path, page_num)
        raw = call_llm_json(GLOSSARY_PROMPT, image_parts=[img])
        return json.loads(raw)

    for idx, page_num in enumerate(glossary_pages):
        try:
            parsed = _scan_page(page_num)
            merged_abbrevs.update(parsed.get("abbreviations", {}))
            merged_grades.update(parsed.get("material_grades", {}))
            note = parsed.get("notes", "")
            if note and note != "none found":
                notes_list.append(note)
            rprint(f"  [yellow]Glossary p{page_num+1}:[/] {len(parsed.get('abbreviations', {}))} terms")
        except Exception as e:
            rprint(f"  [red]Glossary error page {page_num}: {e}[/]")
        if idx < len(glossary_pages) - 1:
            time.sleep(20)

    # Ensure standard defaults exist if not found in the PDF
    defaults = {
        "COL": "Column", "BM": "Beam", "PL": "Plate",
        "UB": "Universal Beam", "UC": "Universal Column",
        "RHS": "Rectangular Hollow Section", "SHS": "Square Hollow Section",
        "CHS": "Circular Hollow Section", "FB": "Flat Bar",
    }
    for k, v in defaults.items():
        merged_abbrevs.setdefault(k, v)

    glossary = {
        "abbreviations": merged_abbrevs,
        "material_grades": merged_grades or {"default_steel": "S275"},
        "notes": "; ".join(notes_list) if notes_list else "Using standard defaults",
    }

    Path(GLOSSARY_OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(GLOSSARY_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(glossary, f, indent=2, ensure_ascii=False)

    rprint(f"[bold green]Glossary complete.[/] {len(merged_abbrevs)} terms → {GLOSSARY_OUTPUT_FILE}")
    return glossary
