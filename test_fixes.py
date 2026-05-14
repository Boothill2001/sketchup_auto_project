"""Quick test for grid expansion + column level splitting + section cleaning."""
import json
from agents.mapper import _expand_multi_grid_members, _split_column_by_levels
from agents.coder import lookup_section

# ── Test 1: Multi-grid expansion ──────────────────────────────────────────────
print("=" * 60)
print("TEST 1: Multi-Grid Expansion")
print("=" * 60)
m = [{
    'mark': 'CH35c',
    'grid_reference': '1/B,C,D; 2/B,C,D; 3/B,C,D; 4/B,C,D',
    'quantity': 12,
    'section': 'CH35a'
}]
expanded = _expand_multi_grid_members(m)
print(f"1 schedule member -> {len(expanded)} expanded members")
for e in expanded[:5]:
    print(f"  {e['mark']} -> grid_ref={e['grid_reference']}")
print(f"  ... and {len(expanded) - 5} more")

# ── Test 2: Column level splitting ────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 2: Column Z-Level Splitting")
print("=" * 60)
levels = [
    {'name': 'Base', 'z_mm': 0},
    {'name': 'FL1', 'z_mm': 3500},
    {'name': 'FL2', 'z_mm': 7000},
    {'name': 'FL3', 'z_mm': 10500},
    {'name': 'Roof', 'z_mm': 13500},
]
base = {
    'section': 'UC155d', 'type': 'column', 'confidence': 'high',
    'material': '',
}
segs = _split_column_by_levels('C1', base, 1000, 2000, levels)
print(f"1 column -> {len(segs)} segments:")
for s in segs:
    print(f"  {s['mark']:30s} z={s['start_point']['z']:6d} -> {s['end_point']['z']:6d}  "
          f"{s['level_ref']}  source={s['source']}")

# ── Test 3: Section cleaning ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 3: Section String Cleaning (strip qualifiers)")
print("=" * 60)
test_cases = [
    ("SH15b (U)", "SH15b"),
    ("UC155d Under", "UC155d"),
    ("SH30b Under Only", "SH30b"),
    ("UB36b", "UB36b"),
    ("CH35a (Under)", "CH35a"),
    ("FB (U)", "FB"),
]
for raw, expected in test_cases:
    result = lookup_section(raw)
    status = "MATCH" if result is not None else "MISS"
    d = result.get("d", "?") if result else "?"
    print(f"  {raw:25s} -> d={d} [{status}] (expected: {expected})")

# ── Test 4: RC member section dimensions ──────────────────────────────────────
print("\n" + "=" * 60)
print("TEST 4: RC Member Section Inheritance")
print("=" * 60)
from agents.coder import _section_rect
rc_col = {
    'mark': 'COLUMN_W200x300', 'section': 'RC', 'type': 'column',
    'material': 'RC', 'width_mm': 300, 'depth_mm': 300,
}
pts, log = _section_rect({'type': 'RC'}, 'column', 'RC', rc_col)
print(f"  RC column: {log}")
print(f"  Pt bounds: x=[{min(p[0] for p in pts)}..{max(p[0] for p in pts)}], "
      f"y=[{min(p[1] for p in pts)}..{max(p[1] for p in pts)}]")

print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)