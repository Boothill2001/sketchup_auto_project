"""Scan LLM cache for schedule extraction results and diagnose failure."""
import json, os

cache_dir = "data/llm_cache"
results = []

for fn in os.listdir(cache_dir):
    if not fn.endswith(".json"):
        continue
    fp = os.path.join(cache_dir, fn)
    with open(fp, "r", encoding="utf-8") as fh:
        try:
            d = json.load(fh)
        except Exception:
            continue
    resp = str(d.get("response", ""))
    call_type = d.get("call_type", "")
    
    # Detect schedule extraction calls
    is_schedule = False
    if '"page_source"' in resp and '"members"' in resp and '"mark"' in resp:
        is_schedule = True
    elif "STEEL SCHEDULE" in resp or "COLUMN SCHEDULE" in resp or "BEAM SCHEDULE" in resp:
        is_schedule = True
    
    if not is_schedule:
        continue
    
    # Extract page_source and member count
    import re
    ps_match = re.search(r'"page_source"\s*:\s*(\d+)', resp)
    page_src = int(ps_match.group(1)) if ps_match else -1
    
    member_count = resp.count('"mark"')
    # Check if members have real dimensions
    has_actual_data = '"width_mm": null' not in resp[:1000] if member_count > 0 else False
    
    # Get first 800 chars of response for preview
    preview = resp[:800]
    
    results.append((page_src, member_count, has_actual_data, fn, preview))

results.sort()
print(f"Found {len(results)} schedule extraction cache entries:")
print("=" * 80)
for pg, cnt, has_data, fn, preview in results:
    print(f"\n--- Page {pg}: {cnt} members | has_real_dims={has_data} | {fn}")
    print(f"  Response preview: {preview}")
    print()