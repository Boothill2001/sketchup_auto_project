"""
=============================================================================
Agent 3 — Spatial Parser (REWRITTEN — Zero-LLM Grid Engine)
=============================================================================
Phase 3: Extract grid system + spatial data from PDF.
NOW: Delegates to pdf_structured_parser + grid_math_engine (deterministic, $0).

Output: data/output_json/spatial_data.json
  Contains: grids_x, grids_y, levels, scale, confidence.
"""

import json
import os
from pathlib import Path
from rich import print as rprint

from config import SPATIAL_OUTPUT_FILE, INPUT_PDF_DIR
from core.pdf_structured_parser import PDFStructuredParser, extract_all_structured_data
from agents.grid_math_engine import GridMathEngine, GridSystem, MultiPageGridMerger


def parse_spatial_pages(pdf_path: str) -> dict:
    """
    NEW: Parse spatial data using deterministic zero-LLM engine.
    
    Pipeline:
      1. PDFStructuredParser → extract text, vectors, grid labels, dimension chains
      2. GridMathEngine → build GridSystem (grids_x, grids_y, levels, scale)
      3. Serialize → spatial_data.json
    
    Returns:
      dict with grids_x, grids_y, levels, scale, confidence
    """
    rprint("\n[bold cyan]══════════════ SPATIAL PARSER — Zero-LLM Grid Engine ══════════════[/bold cyan]")
    rprint(f"[dim]Input PDF: {pdf_path}[/dim]")

    # ── Step 1: Structured PDF Extraction ────────────────────────────────────
    rprint("\n[yellow]Step 1: Programmatic PDF extraction (no LLM)...[/yellow]")
    parser = PDFStructuredParser(pdf_path)
    pages = parser.parse_all()

    rprint(f"  Pages parsed: {len(pages)}")
    for p in pages:
        rprint(
            f"    Page {p.page_index}: {p.page_type} | "
            f"grid labels={len(p.grid_labels)}, "
            f"dim chains={len(p.dimension_chains)}, "
            f"levels={len(p.level_markers)}, "
            f"member marks={len(p.member_marks)}, "
            f"tables={len(p.tables)}"
        )

    plan_count = sum(1 for p in pages if p.page_type == "plan")
    elev_count = sum(1 for p in pages if p.page_type in ("elevation", "section"))
    sched_count = sum(1 for p in pages if p.page_type == "schedule")
    rprint(f"  Summary: {plan_count} plan, {elev_count} elevation/section, {sched_count} schedule pages")

    # ── Step 2: Grid Math Engine ─────────────────────────────────────────────
    rprint("\n[yellow]Step 2: Building grid system (math engine, no LLM)...[/yellow]")
    engine = GridMathEngine(pages)
    grid_system = engine.build()

    # ── Step 3: Serialize Output ─────────────────────────────────────────────
    rprint("\n[yellow]Step 3: Serializing spatial data...[/yellow]")

    output = {
        "source": "zero_llm_grid_engine",
        "pdf_path": pdf_path,
        "confidence": round(grid_system.confidence, 3),

        # Grid data — convert raw dicts to mapper-compatible list-of-dicts
        "grids_x": [
            {"name": name, "x_mm": x_mm}
            for name, x_mm in sorted(grid_system.grids_x.items())
        ],
        "grids_y": [
            {"name": name, "y_mm": y_mm}
            for name, y_mm in sorted(grid_system.grids_y.items())
        ],
        "grid_count_x": len(grid_system.grids_x),
        "grid_count_y": len(grid_system.grids_y),

        # Level data — mapper expects list of {name, z_mm}
        "levels": [
            {
                "name": lm.name,
                "z_mm": lm.z_mm,
                "confidence": lm.confidence,
                "page": lm.page,
            }
            for lm in sorted(grid_system.levels, key=lambda l: l.z_mm)
        ],
        "level_count": len(grid_system.levels),

        # Scale
        "scale_x": round(grid_system.scale_x, 4),
        "scale_y": round(grid_system.scale_y, 4),

        # Building extents
        "building_bbox_mm": {
            "x0": grid_system.building_bbox_mm[0],
            "y0": grid_system.building_bbox_mm[1],
            "x1": grid_system.building_bbox_mm[2],
            "y1": grid_system.building_bbox_mm[3],
        },

        # Structured page summary (for downstream agents)
        "page_summary": {
            "total_pages": len(pages),
            "plan_pages": [p.page_index for p in pages if p.page_type == "plan"],
            "elevation_pages": [p.page_index for p in pages if p.page_type in ("elevation", "section")],
            "schedule_pages": [p.page_index for p in pages if p.page_type == "schedule"],
            "page_types": {p.page_index: p.page_type for p in pages},
        },

        # Raw page data (compact — text counts, not full text)
        "pages": [
            {
                "index": p.page_index,
                "type": p.page_type,
                "width": p.page_width,
                "height": p.page_height,
                "grid_labels": [
                    {"label": g.label, "axis": g.axis, "px": g.px, "py": g.py}
                    for g in p.grid_labels
                ],
                "dimension_chains": [
                    {"axis": dc.axis, "values": dc.values, "confidence": dc.confidence}
                    for dc in p.dimension_chains
                ],
                "level_markers": [
                    {"name": lm.name, "z_mm": lm.z_mm, "py": lm.py}
                    for lm in p.level_markers
                ],
                "member_marks_count": len(p.member_marks),
                "table_count": len(p.tables),
            }
            for p in pages
        ],
    }

    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(SPATIAL_OUTPUT_FILE), exist_ok=True)
    with open(SPATIAL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    rprint(f"\n[green]✓ Spatial data saved to: {SPATIAL_OUTPUT_FILE}[/green]")
    rprint(f"  Grid X: {len(grid_system.grids_x)} lines ({list(grid_system.grids_x.keys())[:6]}...)")
    rprint(f"  Grid Y: {len(grid_system.grids_y)} lines ({list(grid_system.grids_y.keys())[:6]}...)")
    rprint(f"  Levels: {len(grid_system.levels)} ({[lm.name for lm in grid_system.levels[:4]]})")
    rprint(f"  Scale:  X={grid_system.scale_x:.2f} mm/px, Y={grid_system.scale_y:.2f} mm/px")
    rprint(f"  Confidence: {grid_system.confidence:.2f}")
    rprint(f"[bold green]══════════════ SPATIAL PARSER COMPLETE ══════════════[/bold green]\n")

    return output


# ── Legacy Compatibility ────────────────────────────────────────────────────

def parse_spatial_pages_legacy(pdf_path: str) -> dict:
    """
    LEGACY: Old LLM-based spatial parser (kept for reference/fallback).
    Not recommended for production.
    """
    rprint("[yellow]WARNING: Using legacy LLM-based spatial parser — high cost, low accuracy[/yellow]")
    # Redirect to new parser
    return parse_spatial_pages(pdf_path)