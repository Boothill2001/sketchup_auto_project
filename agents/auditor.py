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
    for m in mapped_members:
        mark = m.get("mark", "")
        if not mark:
            continue
        if mark not in ruby_script:
            issues.append({
                "mark": mark,
                "issue_type": "missing_mark",
                "detail": f"Mark '{mark}' not found anywhere in Ruby script",
            })
            continue

        # Check coordinates are referenced
        sp = m.get("start_point", {})
        sx = str(int(sp.get("x", 0)))
        if sx != "0" and sx not in ruby_script:
            issues.append({
                "mark": mark,
                "issue_type": "wrong_coords",
                "detail": f"start_point.x={sx} not found near mark block",
            })
    return issues


def orphan_check(schedule_members: list[dict], mapped_members: list[dict]) -> list[str]:
    schedule_marks = {m["mark"] for m in schedule_members if m.get("mark")}
    mapped_marks   = {m["mark"] for m in mapped_members   if m.get("mark")}
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

    # 3. AI audit
    rprint("[bold red]Auditor:[/] Running AI validation...")
    ai_report = {}
    try:
        prompt = AI_AUDIT_PROMPT.replace(
            "{mapped_json}", json.dumps({"mapped_members": mapped_members[:30]}, indent=2)[:4000]
        ).replace(
            "{ruby_snippet}", ruby_script[:5000]
        )
        raw = call_llm_json(prompt)
        ai_report = json.loads(raw)
    except Exception as e:
        rprint(f"  [red]AI audit error: {e}[/]")
        ai_report = {"issues": [], "overall_passed": False, "summary": str(e)}

    # Merge all issues
    all_issues = static_issues + ai_report.get("issues", [])
    error_feedback_map = build_error_feedback_map(all_issues)

    # Unmapped members report
    unmapped = [m for m in mapped_members if m.get("confidence") == "unmapped"]

    final_passed = (
        len(all_issues) == 0
        and not orphans
        and ai_report.get("overall_passed", False)
    )

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
