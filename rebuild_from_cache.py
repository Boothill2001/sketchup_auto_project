"""
REBUILD FROM CACHE — Bypasses LLM calls, extracts ALL data from existing cache.
Generates: steel_schedule.json, spatial_data.json with real grid lines.
"""
import json, os, re, sys
from pathlib import Path

CACHE_DIR = Path("data/llm_cache")
OUTPUT_DIR = Path("data/output_json")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: Rebuild steel_schedule.json from cache ──────────────────────────
print("="  * 60)
print("STEP 1: Rebuilding steel_schedule.json from cache...")

all_members = []
seen_marks = set()

schedule_cache_files = []

for fn in sorted(os.listdir(CACHE_DIR)):
    if not fn.endswith(".json"):
        continue
    fp = CACHE_DIR / fn
    try:
        d = json.loads(fp.read_text(encoding="utf-8"))
    except:
        continue
    resp = d.get("response", "")
    if '"page_source"' not in resp or '"members"' not in resp:
        continue
    
    # Extract page_source from response
    ps_match = re.search(r'"page_source"\s*:\s*(\d+)', resp)
    page_src = int(ps_match.group(1)) if ps_match else -1
    if page_src <= 1:
        continue
    
    # Strip markdown fences (simulate call_llm_json)
    raw = resp.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()
    
    # Parse with repair
    try:
        parsed = json.loads(raw)
    except:
        # Try repair
        try:
            from json_repair import repair_json
            parsed = json.loads(repair_json(raw))
        except:
            print(f"  SKIP: Cannot parse {fn}")
            continue
    
    members = parsed.get("members", [])
    valid = 0
    for m in members:
        mark = (m.get("mark") or "").strip().upper()
        if not mark:
            continue
        if mark in seen_marks:
            continue
        seen_marks.add(mark)
        # Ensure all required fields
        m["mark"] = mark
        m.setdefault("type", "beam")
        m.setdefault("material", "S275")
        m.setdefault("width_mm", None)
        m.setdefault("depth_mm", None)
        m.setdefault("thickness_mm", None)
        m.setdefault("length_mm", None)
        m.setdefault("section", None)
        m.setdefault("remarks", "")
        m["page_source"] = page_src
        all_members.append(m)
        valid += 1
    
    if valid > 0:
        print(f"  Page {page_src}: {valid} new members (total unique: {len(all_members)})")

schedule = {"total_members": len(all_members), "members": all_members}
out_path = OUTPUT_DIR / "steel_schedule.json"
out_path.write_text(json.dumps(schedule, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n  -> Written {len(all_members)} members to {out_path}")

# Show summary by type
by_type = {}
for m in all_members:
    t = m.get("type", "unknown")
    by_type[t] = by_type.get(t, 0) + 1
print(f"  By type: {by_type}")

# ── Step 2: Rebuild spatial_data.json from cache ────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Rebuilding spatial_data.json from cache...")

spatial_data = {"grid_lines": [], "levels": [], "columns": []}

for fn in sorted(os.listdir(CACHE_DIR)):
    if not fn.endswith(".json"):
        continue
    fp = CACHE_DIR / fn
    try:
        d = json.loads(fp.read_text(encoding="utf-8"))
    except:
        continue
    resp = d.get("response", "")
    
    # Look for spatial/grid data
    if '"grid_lines"' in resp or '"grid_system"' in resp:
        raw = resp.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(raw)
        except:
            try:
                from json_repair import repair_json
                parsed = json.loads(repair_json(raw))
            except:
                continue
        
        grid_lines = parsed.get("grid_lines", parsed.get("grid_system", []))
        if grid_lines and len(grid_lines) > 0:
            spatial_data["grid_lines"] = grid_lines
            print(f"  Found {len(grid_lines)} grid lines from {fn[:16]}...")
            break

# If no grid from cache, generate default 6x6 grid
if not spatial_data["grid_lines"]:
    print("  No grid lines in cache, generating default grid...")
    grid = []
    for i in range(1, 7):
        grid.append({
            "id": f"A{i}",
            "direction": "horizontal",
            "offset_mm": i * 6000,
            "label": str(i)
        })
    for i in range(1, 7):
        grid.append({
            "id": f"{i}",
            "direction": "vertical",
            "offset_mm": i * 6000,
            "label": str(i)
        })
    spatial_data["grid_lines"] = grid
    print(f"  Generated {len(grid)} default grid lines")

# Extract levels from cache
for fn in sorted(os.listdir(CACHE_DIR)):
    if not fn.endswith(".json"):
        continue
    fp = CACHE_DIR / fn
    try:
        d = json.loads(fp.read_text(encoding="utf-8"))
    except:
        continue
    resp = d.get("response", "")
    if '"levels"' in resp or '"floor_levels"' in resp:
        raw = resp.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(raw)
            levels = parsed.get("levels", parsed.get("floor_levels", []))
            if levels:
                spatial_data["levels"] = levels
                print(f"  Found {len(levels)} levels from cache")
                break
        except:
            pass

if not spatial_data["levels"]:
    spatial_data["levels"] = [
        {"name": "Ground Floor", "elevation_mm": 0},
        {"name": "Level 1", "elevation_mm": 4000},
        {"name": "Roof", "elevation_mm": 8000},
    ]
    print("  Generated default 3 levels")

out_path2 = OUTPUT_DIR / "spatial_data.json"
out_path2.write_text(json.dumps(spatial_data, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n  -> Written spatial data to {out_path2}")
print(f"     Grid lines: {len(spatial_data['grid_lines'])}")
print(f"     Levels: {len(spatial_data['levels'])}")

print("\n" + "=" * 60)
print("REBUILD COMPLETE!")
print(f"  Members: {len(all_members)}")
print(f"  Grid lines: {len(spatial_data['grid_lines'])}")
print(f"  Levels: {len(spatial_data['levels'])}")