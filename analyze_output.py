import json, sys

with open(r'c:\Users\minht\Feeldx\sketchup_auto_project\data\output_json\mapped_members.json','r',encoding='utf-8') as f:
    raw = json.load(f)

members = raw.get('mapped_members', [])
if isinstance(members, dict):
    members = list(members.values())
elif isinstance(members, str):
    print("ERROR: mapped_members is a string")
    sys.exit(1)

print(f"Total members: {len(members)}")

# By source
sources = {}
for m in members:
    src = m.get('source', 'unknown') if isinstance(m, dict) else 'non-dict'
    sources[src] = sources.get(src, 0) + 1
print(f"By source: {sources}")

# By type
types = {}
for m in members:
    t = m.get('type', 'unknown') if isinstance(m, dict) else 'non-dict'
    types[t] = types.get(t, 0) + 1
print(f"By type: {types}")

# With valid Z (Z is nested in start_point / end_point)
def _get_z(m):
    if not isinstance(m, dict):
        return None
    sp = m.get('start_point', {})
    return sp.get('z', 0) if isinstance(sp, dict) else 0

has_z = [m for m in members if _get_z(m) > 0]
print(f"With Z>0: {len(has_z)}/{len(members)}")

# Members at ground Z=0 (may indicate unplaced)
at_zero_z = [m.get('mark','?') for m in members if _get_z(m) == 0]
print(f"At Z=0 ({len(at_zero_z)}): {at_zero_z[:10]}{'...' if len(at_zero_z) > 10 else ''}")

# With section
has_section = [m for m in members if isinstance(m, dict) and m.get('section') and str(m.get('section','')).strip()]
print(f"With section: {len(has_section)}/{len(members)}")

# Missing sections
missing_sec = [m.get('mark','?') for m in members if isinstance(m, dict) and (not m.get('section') or not str(m.get('section','')).strip())]
print(f"Missing sections ({len(missing_sec)}): {missing_sec[:15]}{'...' if len(missing_sec) > 15 else ''}")

# Grid-coerced / fallback
coerced = [m.get('mark','?') for m in members if isinstance(m, dict) and m.get('source') in ('grid_coerce', 'fallback')]
print(f"Grid-coerced/fallback ({len(coerced)}): {coerced[:10]}{'...' if len(coerced) > 10 else ''}")

# Column segments (split by levels)
col_segments = [m for m in members if isinstance(m, dict) and '_' in str(m.get('mark','')) and m.get('type') == 'column']
print(f"Column segments (split by level): {len(col_segments)}")

# RC members
rc_members = [m for m in members if isinstance(m, dict) and str(m.get('material','')).upper() == 'RC']
print(f"RC members: {len(rc_members)}")

# Sample a few entries
print("\n--- Sample members ---")
for m in members[:5]:
    if isinstance(m, dict):
        sp = m.get('start_point', {})
        ep = m.get('end_point', {})
        print(json.dumps({
            'mark': m.get('mark'),
            'x': sp.get('x') if isinstance(sp, dict) else None,
            'y': sp.get('y') if isinstance(sp, dict) else None,
            'z': sp.get('z') if isinstance(sp, dict) else None,
            'type': m.get('type'),
            'section': m.get('section'),
            'source': m.get('source'),
            'material': m.get('material'),
        }, indent=2))
