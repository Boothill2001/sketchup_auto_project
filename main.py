"""
Pipeline Orchestrator — 5-Agent LOD 300 Pipeline with Auto-Feedback Loop.

Flow:
  [Scanner] → [Glossary] → [Schedule Parser] → [Spatial Parser] → [Mapper]
  → [Coder] → [Auditor] → (retry Coder if issues) → Done

Usage:
  CLI:  python main.py <path_to_structural.pdf>
  API:  from main import run_pipeline; result = run_pipeline(pdf_path, log_fn=...)
"""

import sys
import io
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import rich
from rich import print as rprint

from config import MAX_AUDIT_RETRIES, CODER_OUTPUT_FILE
from core.llm_wrapper       import validate_keys
from agents.scanner         import scan_pdf, get_pages_by_role
from agents.glossary_agent  import build_glossary
from agents.schedule_parser import parse_schedule_pages
from agents.spatial_parser  import parse_spatial_pages
from agents.mapper          import run_mapper
from agents.coder           import build_ruby_script, save_ruby_script
from agents.auditor         import run_audit


# ── Rich → log_fn bridge ────────────────────────────────────────────────────

_ANSI = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_MARKUP = re.compile(r"\[/?[^\[\]]*\]")


class _QueueStream(io.TextIOBase):
    """
    TextIO sink that strips ANSI/Rich markup and forwards lines to log_fn.
    Attached to Rich's Console.file so every rprint() in every agent
    automatically reaches the Streamlit UI.
    """
    def __init__(self, fn: Callable[[str], None]) -> None:
        self._fn = fn
        self._buf = ""

    def write(self, text: str) -> int:
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            clean = _ANSI.sub("", _MARKUP.sub("", line)).strip()
            if clean:
                self._fn(clean)
        return len(text)

    def flush(self) -> None:
        if self._buf.strip():
            clean = _ANSI.sub("", _MARKUP.sub("", self._buf)).strip()
            if clean:
                self._fn(clean)
            self._buf = ""

    def readable(self) -> bool: return False
    def writable(self) -> bool: return True
    def seekable(self) -> bool: return False


@contextmanager
def _redirect_rich(log_fn: Callable[[str], None]):
    """Temporarily point Rich's global console at log_fn for this call stack."""
    console  = rich.get_console()
    old_file = console.file
    try:
        console.file = _QueueStream(log_fn)
        yield
    finally:
        console.file = old_file


# ── Pipeline ─────────────────────────────────────────────────────────────────

def _phase(label: str) -> None:
    rprint(f"\n{'='*60}")
    rprint(label)
    rprint("="*60)


_PHASE_GAP = 65  # seconds — lets all 6 RPM windows reset between phases


def _phase_gap() -> None:
    """Sleep between consecutive LLM phases so RPM windows reset before the next phase starts."""
    rprint(f"[dim]  ↳ inter-phase pause {_PHASE_GAP}s (RPM window reset)...[/]")
    time.sleep(_PHASE_GAP)


def run_pipeline(
    pdf_path: str,
    log_fn: Callable[[str], None] | None = None,
) -> dict:
    """
    Execute the full 5-agent pipeline.

    Args:
        pdf_path: Absolute or relative path to the input PDF.
        log_fn:   Optional callable(str) — receives every log line.
                  When provided, Rich's console is redirected so ALL
                  rprint() calls inside every agent flow here too.

    Returns dict with keys: ruby_path, members_total, placed, unmapped,
                            unmapped_marks, audit_passed, error.
    """
    _redirect = _redirect_rich(log_fn) if log_fn else _noop()

    with _redirect:
        return _run(pdf_path)


@contextmanager
def _noop():
    yield


def _run(pdf_path: str) -> dict:
    pdf = Path(pdf_path)
    if not pdf.exists():
        rprint(f"ERROR: PDF not found: {pdf_path}")
        return {"ruby_path": None, "error": f"PDF not found: {pdf_path}",
                "members_total": 0, "placed": 0,
                "unmapped": 0, "unmapped_marks": [], "audit_passed": False}

    rprint(f"PDF: {pdf.name}  ({pdf.stat().st_size / 1024:.1f} KB)")

    # Pre-flight: verify which API keys are alive before burning quota on failures
    _phase("PRE-FLIGHT — API KEY VALIDATION")
    key_status = validate_keys()
    alive = [k for k, v in key_status.items() if v == "ok"]
    if not alive:
        msg = "No working API keys found. Check quota/billing on all accounts."
        rprint(f"[bold red]ABORT:[/] {msg}")
        return {"ruby_path": None, "error": msg,
                "members_total": 0, "placed": 0,
                "unmapped": 0, "unmapped_marks": [], "audit_passed": False}

    # Phase 1 — Scanner
    _phase("PHASE 1 — SCANNER: Drawing Index Classification")
    index = scan_pdf(str(pdf))
    roles = get_pages_by_role(index)
    for role, pages in roles.items():
        rprint(f"  {role:12}: pages {pages}")

    # Phase 1b — Glossary
    _phase_gap()
    _phase("PHASE 1b — GLOSSARY: Abbreviation Dictionary")
    build_glossary(str(pdf), roles["glossary"])

    # Phase 2 — Schedule Parser
    _phase_gap()
    _phase("PHASE 2 — SCHEDULE PARSER: Steel Member Extraction")
    members = parse_schedule_pages(str(pdf), roles["schedule"])
    if not members:
        msg = "No structural members found. Check scanner output and PDF quality."
        rprint(f"ERROR: {msg}")
        return {"ruby_path": None, "error": msg,
                "members_total": 0, "placed": 0,
                "unmapped": 0, "unmapped_marks": [], "audit_passed": False}
    rprint(f"  {len(members)} members extracted.")

    # Phase 3 — Spatial Parser
    _phase_gap()
    _phase("PHASE 3 — SPATIAL PARSER: Grid & Level Extraction")
    parse_spatial_pages(str(pdf), roles["plan"], roles["elevation"])

    # Phase 4 — Mapper
    _phase_gap()
    _phase("PHASE 4 — MAPPER: Assigning 3D Coordinates")
    mapped_members = run_mapper(str(pdf), roles["plan"], roles["elevation"])

    # Phase 5 — Coder
    _phase_gap()
    _phase("PHASE 5 — CODER: Ruby LOD 300 Script Generation")
    script = build_ruby_script(mapped_members)
    save_ruby_script(script)

    # Phase 6 — Auditor + auto-feedback loop
    _phase("PHASE 6 — AUDITOR: Cross-Validation")
    audit = run_audit()

    for retry_num in range(1, MAX_AUDIT_RETRIES + 1):
        if audit["final_passed"]:
            break
        error_map = audit.get("error_feedback_map", {})
        if not error_map:
            break
        _phase(f"RETRY {retry_num}/{MAX_AUDIT_RETRIES} — CODER AUTO-CORRECTION")
        rprint(f"  Fixing {len(error_map)} member(s)...")
        script = build_ruby_script(mapped_members, error_feedback_map=error_map)
        save_ruby_script(script)
        audit = run_audit()

    # Summary
    unmapped       = audit.get("unmapped_count", 0)
    unmapped_marks = audit.get("unmapped_marks", [])
    ruby_path      = str(CODER_OUTPUT_FILE) if Path(CODER_OUTPUT_FILE).exists() else None

    _phase("PIPELINE COMPLETE")
    rprint(f"  Total members : {len(members)}")
    rprint(f"  3D-placed     : {len(mapped_members) - unmapped}")
    rprint(f"  Unmapped      : {unmapped}  {unmapped_marks or ''}")
    rprint(f"  Audit passed  : {audit['final_passed']}")
    rprint(f"  Ruby output   : {ruby_path}")

    return {
        "ruby_path":      ruby_path,
        "members_total":  len(members),
        "placed":         len(mapped_members) - unmapped,
        "unmapped":       unmapped,
        "unmapped_marks": unmapped_marks,
        "audit_passed":   audit["final_passed"],
        "error":          None,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        rprint("[yellow]Usage:[/] python main.py <path_to_structural.pdf>")
        sys.exit(1)
    run_pipeline(sys.argv[1])
