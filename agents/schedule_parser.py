"""
Agent 2 — Schedule Parser
Phase 2: Extract member marks and section sizes from Steel Schedule pages.
Uses the Glossary and PDF Analysis context to adapt to ANY convention.
Output: data/output_json/steel_schedule.json
"""

import json
import re
from pathlib import Path
from rich import print as rprint

from config import SCHEDULE_OUTPUT_FILE, GLOSSARY_OUTPUT_FILE, USE_TEXT_EXTRACTION_FIRST
from core.llm_wrapper import call_llm_json
from core.pdf_utils import render_page_as_image_part, segment_page_regions, get_page_count, extract_tables_pdfplumber
from core.analysis_context import build_schedule_context, load_analysis_dict


def _repair_json(raw: str) -> str:
    """Attempt to repair malformed JSON from LLM output before parsing."""
    if not raw or not raw.strip():
        return raw
    raw = raw.strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    # Fix trailing commas before closing brackets/braces
    raw = re.sub(r",(\s*[}\]])", r"\1", raw)
    # Fix missing commas between array elements (LLM sometimes omits them)
    raw = re.sub(r'"\s*\n\s*"', '",\n  "', raw)
    # Fix unquoted property names (common LLM error)
    raw = re.sub(r'(?<=[\{,])\s*(\w+)(\s*:)', r'"\1"\2', raw)
    # Fix single-quoted strings (LLM sometimes uses single quotes)
    raw = re.sub(r"(?<=:\s)'([^']*)'", r'"\1"', raw)
    # Remove trailing garbage after closing brace
    last_brace = raw.rfind("}")
    if last_brace != -1:
        # Check for balanced braces
        brace_count = 0
        cut_pos = -1
        for i, ch in enumerate(raw):
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    cut_pos = i
        if cut_pos != -1 and cut_pos < len(raw) - 1:
            raw = raw[:cut_pos + 1]
    return raw


def _safe_parse_json(raw: str, source_label: str = "") -> dict:
    """Parse JSON with repair fallback."""
    raw = _repair_json(raw)
    errors = []
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        errors.append(f"direct: {e}")
    # Try with json_repair if available
    try:
        from json_repair import repair_json
        repaired = repair_json(raw)
        return json.loads(repaired)
    except ImportError:
        pass
    except Exception as e:
        errors.append(f"repair: {e}")
    rprint(f"  [yellow]JSON parse failed for {source_label}: {errors[-1] if errors else 'unknown'}[/]")
    return {"members": []}


EXTRACT_PROMPT_TEMPLATE = """You are a Senior Structural Detailer extracting data from structural schedules and drawings.

{analysis_context}

PROJECT GLOSSARY (use this to normalise abbreviations):
{glossary_json}

# BUILDING TYPE CONTEXT (injected from Phase 0 analysis)
{building_context}

# CRITICAL — Element Extraction Guide

This drawing may contain BOTH steel AND reinforced concrete (RC) elements. Extract ALL structural members.

## IF RC BUILDING (columns, beams, walls, slabs in concrete):
- RC Columns: marks like "C1", "C2", "COL1" → type="column", material="RC"
  - Extract column dimensions from schedule: width_mm, depth_mm (or diameter_mm for circular)
  - Reinforcement notes (e.g. "12N20") → put in remarks
- RC Beams: marks like "B1", "BM1", "G1" → type="beam", material="RC"  
  - Extract beam dimensions: depth_mm, width_mm (overall cross-section)
- RC Walls: marks like "W1", "WL1" → type="wall", material="RC"
  - Extract wall thickness_mm
- RC Slabs: marks like "S1", "SL1", "PT" → type="slab", material="RC"
  - Extract slab thickness_mm

## IF STEEL ROOF/FRAME (steel members on top of RC structure):
Use these conventions for steel marks:

### Convention A — Old Australian (e.g. "30b", "35c", "CH13c")
Format: [prefix][depth_units][variant_letter]
- Prefixes: SH=SHS/RHS hollow, UB=Universal Beam, UC=Universal Column, CH=PFC Channel, 
  PF=Portal Frame, Z=Z-purlin, RB=Round Bar, LW=Lintel/Wall beam, FB=Flat Bar
- If NO prefix (just number + letter like "30b"): check SECTION column
- Depth: number × 10 = approx depth mm (e.g. "30"→~300mm). If number<15: ×25.4
- Variant letter (a,b,c,d,g): weight variant (a=lightest)

### Convention B — Metric (e.g. "360UB56.7", "200UC46")
Format: [depth_mm][type_code][mass_kg/m]

### Convention C — TCVN/Asian (e.g. "I200x100x5.5x8", "H250x250x9x14")
Format: [shape][depth]x[width]x[web]x[flange]

# TYPE INFERENCE RULES:
- Mark "C"+"number" with RC material → type="column"
- Mark "B"/"BM"/"G"+number with RC material → type="beam"
- Mark "W"/"WL" with RC material → type="wall"
- Mark "S"/"SL" with RC material → type="slab"
- Steel: UB/UC/PFC/SHS/RHS prefixes → type="beam"/"column" accordingly
- RB → type="brace", FB → type="plate", Z → type="purlin", PF → type="rafter"

# INSTRUCTIONS
Extract EVERY structural member visible. For EACH member return:
- "mark": exact label (e.g. "C1", "B3", "CH13c", "UB36b")
- "material": "RC" for concrete, "S275"/"S355" etc for steel — use glossary default for steel
- "type": "column"|"beam"|"brace"|"plate"|"purlin"|"rafter"|"wall"|"slab"|"other"
- "section": section designation AS WRITTEN (e.g. "500x500", "CH", "UB36b", "200UB25")
- "width_mm": member width/diameter in mm (for RC columns, beams, walls)
- "depth_mm": member depth in mm (for RC columns, beams; overall section depth)
- "thickness_mm": thickness in mm (for RC walls, slabs)
- "length_mm": length in mm as integer — CRITICAL: extract if ANY length shown!
- "quantity": count as integer — null if not shown
- "grid_reference": grid/level reference (e.g. "A-1", "B-2/L1") — null if absent
- "level": floor level name (e.g. "Level 1", "L1", "Roof") — null if not shown
- "remarks": any notes — null if none

Return JSON:
{
  "page_source": <page_number_int>,
  "members": [ { ...member... }, ... ]
}

If no members found: {"page_source": <n>, "members": []}"""


def _build_building_context(building_type: str) -> str:
    """Generate context block telling LLM what types of elements to look for."""
    context_map = {
        "multi_storey_rc": (
            "This is a MULTI-STOREY REINFORCED CONCRETE BUILDING.\n"
            "  PRIMARY STRUCTURE: RC columns, RC beams, RC slabs, RC walls.\n"
            "  SECONDARY: Steel roof framing / steel trusses may be present on roof level.\n"
            "  Look for: COLUMN SCHEDULES, BEAM SCHEDULES, SLAB SCHEDULES, WALL ELEVATIONS.\n"
            "  RC column marks like C1, C2, C3... should have material='RC' with dimensions in mm.\n"
            "  RC walls: INSITU WALLS, BLADEWALLS, LIFT WALLS — material='RC' with thickness_mm.\n"
            "  Steel members (if any): typically on roof/upper levels only — UB, PFC, SHS sections."
        ),
        "steel_frame": (
            "This is a STEEL FRAME BUILDING. All primary members are structural steel.\n"
            "  Look for: BEAM SCHEDULES, COLUMN SCHEDULES, BRACING SCHEDULES.\n"
            "  Extract section designations and decode per the steel conventions above."
        ),
        "composite": (
            "This is a COMPOSITE BUILDING (steel + concrete).\n"
            "  Look for BOTH steel member schedules AND RC element schedules.\n"
            "  RC: columns, walls, slabs, cores. Steel: beams, trusses, roof framing."
        ),
        "unknown": (
            "Building type unknown — extract ALL structural members visible.\n"
            "  Look for ANY schedule tables — steel, concrete, timber, masonry.\n"
            "  Determine material from context (dimensions > 300mm for RC possible)."
        ),
    }
    return context_map.get(building_type, context_map["unknown"])


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
    analysis_dict = load_analysis_dict()
    building_type = analysis_dict.get("building_type", "unknown")
    building_context = _build_building_context(building_type)

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
        base_prompt = base_prompt.replace("{building_context}", building_context)

        # ── Text-first path ──────────────────────────────────────────────────
        text_data = extract_tables_pdfplumber(pdf_path, page_num) if USE_TEXT_EXTRACTION_FIRST else None

        if text_data:
            # ── Quality check: skip if mostly commas/whitespace (scanned image) ────
            meaningful = re.sub(r'[\s,]', '', text_data)
            if len(meaningful) < 50 or not re.search(r'\d', text_data):
                rprint(f"  [yellow]Text extraction low quality ({len(meaningful)} meaningful chars), falling back to vision...[/]")
                text_data = None
            else:
                rprint(f"  [dim]Using text extraction for page {page_num+1} (saved 1 vision call)[/]")
            text_prompt = (
                base_prompt
                + f"\n\nSTEEL SCHEDULE TABLE (CSV from PDF text layer, page {page_num+1}):\n"
                + text_data
            )
            try:
                raw = call_llm_json(text_prompt)  # no image_parts — text only
                parsed = _safe_parse_json(raw, f"page {page_num+1} text")
                members = parsed.get("members", [])
                new_members = []
                for m in members:
                    mark = (m.get("mark") or "").strip()
                    if mark and mark not in seen_marks:
                        seen_marks.add(mark)
                        m["page_source"] = page_num
                        new_members.append(m)
                    elif mark in seen_marks:
                        rprint(f"  [dim]Skipping duplicate mark: {mark}[/]")
                all_members.extend(new_members)
                rprint(f"  [green]+{len(new_members)} members[/] from page {page_num+1} (text-extracted)")
                if new_members:
                    continue  # skip vision extraction for this page
                rprint(f"  [yellow]Text extraction found 0 members — falling back to vision for page {page_num+1}[/]")
            except Exception as e:
                rprint(f"  [yellow]Text extraction failed ({e}), falling back to vision...[/]")

        # ── Vision fallback ──────────────────────────────────────────────────
        rprint(f"  [dim]Page {page_num+1} is scanned image, using vision extraction[/]")
        regions = segment_page_regions(pdf_path, page_num)
        for region_idx, image_part in enumerate(regions):
            label = f"page {page_num+1}" + (f" region {region_idx+1}" if len(regions) > 1 else "")
            try:
                raw = call_llm_json(base_prompt, image_parts=[image_part])
                parsed = _safe_parse_json(raw, f"{label} vision")
                members = parsed.get("members", [])
                new_members = []
                for m in members:
                    mark = (m.get("mark") or "").strip()
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