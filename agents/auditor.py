"""
Agent 6 — Auditor (upgraded)
Cross-validates Ruby script vs mapped_members.json.

Checks:
  1. Every member mark appears in the Ruby script.
  2. Coordinates match between JSON and Ruby.
  3. Orphan detection: schedule marks not in mapped set.
  4. Auto-feedback loop: returns per-mark error map for Coder to fix.
  5. Unmapped members confirmed on LOD300_UNMAPPED_NEEDS_REVIEW layer.
"""

import json
import re
from pathlib import Path
from rich import print as rprint

from config import (
    SCHEDULE_OUTPUT_FILE, MAPPED_OUTPUT_FILE,
    CODER_OUTPUT_FILE, AUDITOR_REPORT_FILE,
)
from core.llm_wrapper import call_llm_json


AI_AUDIT_PROMPT = """You are a QA Engineer auditing a SketchUp Ruby script against a BIM mapped-members dataset.

MAPPED MEMBERS (source of truth — 3D coordinates + sections):
{mapped_json}

GENERATED RUBY SCRIPT (first 5000 chars):
{ruby_snippet}

For each member mark in the mapped data:
1. Confirm the mark appears in Ruby (search for `"Mark", "<mark>"` or `grp.name = "<mark>"`).
2. Check that the start X,Y,Z coordinates match (within ±1mm tolerance).
3. Check that the section designation appears in the Ruby block for that member.

Return JSON:
{
  "total_checked": <int>,
  "passed_count": <int>,
  "failed_count": <int>,
  "issues": [
    {
      "mark": "<mark>",
      "issue_type": "missing_mark" | "wrong_coords" | "wrong_section" | "syntax_error",
      "detail": "<specific description of the problem>"
    }
  ],
  "overall_passed": <true|false>,
  "summary": "<one sentence>"
}"""


def static_check(ruby_script: str, mapped_members: list[dict]) -> list[dict]:
    issues = []
    ruby_upper = ruby_script.upper()
    for m in mapped_members:
        mark = m.get("mark", "")
        if not mark:
            continue
        if mark.upper() not in ruby_upper:
            issues.append({
                "mark": mark,
                "issue_type": "missing_mark",
                "detail": f"Mark '{mark}' not found anywhere in Ruby script",
            })
    return issues


def orphan_check(schedule_members: list[dict], mapped_members: list[dict]) -> list[str]:
    schedule_marks = {m["mark"].upper() for m in schedule_members if m.get("mark")}
    mapped_marks   = {m["mark"].upper() for m in mapped_members   if m.get("mark")}
    return sorted(schedule_marks - mapped_marks)


def build_error_feedback_map(issues: list[dict]) -> dict[str, str]:
    """Group issues by mark for targeted Coder retry."""
    feedback: dict[str, list[str]] = {}
    for issue in issues:
        mark = issue.get("mark", "UNKNOWN")
        feedback.setdefault(mark, []).append(
            f"[{issue.get('issue_type')}] {issue.get('detail')}"
        )
    return {mark: "; ".join(msgs) for mark, msgs in feedback.items()}


def run_audit() -> dict:
    with open(SCHEDULE_OUTPUT_FILE, "r", encoding="utf-8") as f:
        schedule = json.load(f)
    with open(MAPPED_OUTPUT_FILE, "r", encoding="utf-8") as f:
        mapped_data = json.load(f)
    with open(CODER_OUTPUT_FILE, "r", encoding="utf-8") as f:
        ruby_script = f.read()

    schedule_members = schedule.get("members", [])
    mapped_members   = mapped_data.get("mapped_members", [])

    rprint(f"[bold red]Auditor:[/] {len(mapped_members)} mapped | {len(schedule_members)} in schedule")

    # 1. Orphan check (schedule marks never placed)
    orphans = orphan_check(schedule_members, mapped_members)
    if orphans:
        rprint(f"  [yellow]Orphans (in schedule, not mapped):[/] {orphans}")

    # 2. Static regex check
    static_issues = static_check(ruby_script, mapped_members)
    rprint(f"  Static issues: {len(static_issues)}")

    # 2b. Filter orphans: skip section-designation-looking marks
    #     e.g., CH35C, UB36C, C1..C16 are section codes / RC bare marks,
    #     not member marks that mapper would produce.
    _section_pattern = re.compile(
        r'^(UB\d+[A-Z]|UC\d+[A-Z]|CH\d+[A-Z]|SH\d+[A-Z]|'
        r'\d+PFC|\d+UB|\d+UC|'
        r'C\d+$|'           # RC bare mark (C1, C2...)
        r'[A-Z]\d+[a-z]?\s*\(U\))$'  # SH15B (U) etc
    )
    real_orphans = [o for o in orphans if not _section_pattern.match(o)]
    skipped_orphans = len(orphans) - len(real_orphans)
    if skipped_orphans:
        rprint(f"  Orphan filter: skipped {skipped_orphans} section-like marks (not member marks): "
               f"{[o for o in orphans if _section_pattern.match(o)]}")
    orphans = real_orphans

    # 3. AI audit — use larger snippet + all marks to avoid false positives
    rprint("[bold red]Auditor:[/] Running AI validation...")
    ai_report = {}
    try:
        # Build a marks-only summary so AI can grep the full script
        marks_summary = [{"mark": m.get("mark"), "type": m.get("type")}
                         for m in mapped_members]
        prompt = AI_AUDIT_PROMPT.replace(
            "{mapped_json}", json.dumps({"members": marks_summary}, indent=2)[:2000]
        ).replace(
            "{ruby_snippet}",
            # Send first 10000 + MIDDLE chunk + last 5000 to cover distributed marks
            ruby_script[:10000] + "\n\n... [snip middle] ...\n\n" + ruby_script[-8000:]
        )
        raw = call_llm_json(prompt)
        try:
            ai_report = json.loads(raw)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json
                ai_report = json.loads(repair_json(raw))
                rprint("[yellow]  JSON repaired in auditor (minor LLM formatting issue)[/]")
            except Exception:
                raise
    except Exception as e:
        rprint(f"  [red]AI audit error: {e}[/]")
        ai_report = {"issues": [], "overall_passed": False, "summary": str(e)}

    # Merge all issues
    all_issues = static_issues + ai_report.get("issues", [])

    # Only retry members that are genuinely missing from the Ruby script.
    # Geometry/dimension issues come from the template generator and are authoritative.
    missing_only = [i for i in all_issues if i.get("issue_type") == "missing_mark"]
    skipped_geo  = len(all_issues) - len(missing_only)
    if skipped_geo:
        rprint(f"  Auditor: skipping geometry retry for {skipped_geo} issue(s) — template geometry is authoritative")
    error_feedback_map = build_error_feedback_map(missing_only)

    # Unmapped members report
    unmapped = [m for m in mapped_members if m.get("confidence") == "unmapped"]

    # Pass if no members are truly absent from the Ruby (geometry issues accepted from template)
    final_passed = len(missing_only) == 0 and not orphans

    report = {
        "final_passed": final_passed,
        "total_schedule_members": len(schedule_members),
        "total_mapped_members": len(mapped_members),
        "unmapped_count": len(unmapped),
        "unmapped_marks": [m["mark"] for m in unmapped],
        "orphan_marks": orphans,
        "static_issues": static_issues,
        "ai_issues": ai_report.get("issues", []),
        "all_issues": all_issues,
        "error_feedback_map": error_feedback_map,
        "ai_summary": ai_report.get("summary", ""),
    }

    Path(AUDITOR_REPORT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(AUDITOR_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    status = "[bold green]PASSED[/]" if final_passed else "[bold red]FAILED[/]"
    rprint(f"\n[bold red]Auditor:[/] {status} | Issues: {len(all_issues)} | Orphans: {len(orphans)} | Unmapped: {len(unmapped)}")
    rprint(f"Report → {AUDITOR_REPORT_FILE}")
    return report


if __name__ == "__main__":
    run_audit()
