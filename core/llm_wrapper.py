"""
Centralized LLM wrapper — per-key cooldown tracking.
All agents call this module — model version is enforced here.
DO NOT DOWNGRADE OR CHANGE MODEL VERSION (config.py).
"""

import re
import io
import time
import json
import hashlib
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

from google import genai
from rich import print as rprint

from config import (
    GEMINI_MODEL, GEMINI_API_KEYS, GENERATION_CONFIG,
    LLM_CACHE_ENABLED, LLM_CACHE_DIR,
    RPD_LIMIT_PER_KEY, RPD_STATE_FILE,
)

# ── Cache stats ───────────────────────────────────────────────────────────────
_cache_hits   = 0
_cache_misses = 0


def get_cache_stats() -> dict:
    return {"hits": _cache_hits, "misses": _cache_misses}


def _img_bytes(part) -> bytes | None:
    """Extract hashable bytes from an image part (PIL Image, bytes, or genai Part)."""
    if isinstance(part, bytes):
        return part
    if hasattr(part, "tobytes") and hasattr(part, "mode"):  # PIL Image
        try:
            buf = io.BytesIO()
            part.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return part.tobytes()
    if hasattr(part, "inline_data"):  # genai Part
        try:
            return part.inline_data.data
        except Exception:
            pass
    if isinstance(part, dict) and "data" in part:
        return part["data"]
    return None


def _compute_cache_key(prompt: str, image_parts: list | None) -> str:
    h = hashlib.sha256()
    h.update(prompt.encode("utf-8"))
    for part in (image_parts or []):
        b = _img_bytes(part)
        if b:
            h.update(b)
    return h.hexdigest()


def _cache_path(key: str) -> Path:
    return Path(LLM_CACHE_DIR) / f"{key}.json"


def _cache_load(key: str) -> str | None:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        response = data.get("response", "")
        if not response or response == "null":
            return None
        return response
    except Exception:
        return None


def _cache_save(key: str, model: str, response: str, call_type: str) -> None:
    try:
        Path(LLM_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        data = {
            "prompt_hash": key,
            "model": model,
            "response": response,
            "cached_at": datetime.utcnow().isoformat(),
            "call_type": call_type,
        }
        _cache_path(key).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def clear_cache() -> int:
    """Delete all cache files and reset in-memory counters. Returns deleted file count."""
    global _cache_hits, _cache_misses
    _cache_hits = 0
    _cache_misses = 0
    cache_dir = Path(LLM_CACHE_DIR)
    if not cache_dir.exists():
        return 0
    deleted = 0
    for f in cache_dir.glob("*.json"):
        try:
            f.unlink()
            deleted += 1
        except Exception:
            pass
    return deleted


# ── Persistent RPD quota state ────────────────────────────────────────────────
# Key identifier = first 8 hex chars of sha256(api_key) — stable when keys are
# added/removed since it's content-based, not positional.

def _key_id(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()[:8]


def _next_midnight_utc() -> str:
    now = datetime.now(timezone.utc)
    midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.isoformat()


def _default_key_state() -> dict:
    return {
        "rpd_used": 0,
        "rpd_reset_at": _next_midnight_utc(),
        "rpm_blocked_until": None,
    }


def _load_quota_state() -> dict:
    path = Path(RPD_STATE_FILE)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        now = datetime.now(timezone.utc)
        for kid, st in data.items():
            try:
                reset_at = datetime.fromisoformat(st.get("rpd_reset_at", ""))
                if now >= reset_at:
                    st["rpd_used"] = 0
                    st["rpd_reset_at"] = _next_midnight_utc()
            except Exception:
                st["rpd_used"] = 0
                st["rpd_reset_at"] = _next_midnight_utc()
        return data
    except Exception:
        return {}


# Module-level state — loaded once at import, persisted after every successful call.
# Protected by _state_lock (defined below) for all in-memory mutations.
_quota_state: dict[str, dict] = {}   # populated after _state_lock is created (see below)


def _save_quota_state() -> None:
    """Snapshot quota state under lock, then write to disk outside the lock."""
    try:
        path = Path(RPD_STATE_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _state_lock:
            snapshot = json.dumps(_quota_state, indent=2)
        path.write_text(snapshot, encoding="utf-8")
    except Exception:
        pass


# ── Public: quota status ───────────────────────────────────────────────────────

def get_key_quota_status() -> list[dict]:
    """Return live RPD quota info for every configured key (used by app.py sidebar)."""
    now_utc = datetime.now(timezone.utc)
    result = []
    with _state_lock:
        for i, key in enumerate(GEMINI_API_KEYS):
            kid = _key_id(key)
            if kid not in _quota_state:
                _quota_state[kid] = _default_key_state()
            st = _quota_state[kid]
            try:
                reset_at = datetime.fromisoformat(st["rpd_reset_at"])
                if now_utc >= reset_at:
                    st["rpd_used"] = 0
                    st["rpd_reset_at"] = _next_midnight_utc()
            except Exception:
                st["rpd_used"] = 0
                st["rpd_reset_at"] = _next_midnight_utc()
            result.append({
                "index":       i + 1,
                "key_id":      kid,
                "rpd_used":    st["rpd_used"],
                "rpd_limit":   RPD_LIMIT_PER_KEY,
                "exhausted":   st["rpd_used"] >= RPD_LIMIT_PER_KEY,
                "rpd_reset_at": st.get("rpd_reset_at", ""),
            })
    return result


def any_key_available() -> bool:
    """True if at least one key has remaining RPD quota today."""
    return any(not s["exhausted"] for s in get_key_quota_status())


# ── Global rate limiter ───────────────────────────────────────────────────────
# Enforces max 5 calls/min (1 call per 12s) proactively — prevents 429 entirely.
# Applies to ALL call_llm() invocations regardless of which key is selected.
_MIN_CALL_GAP_S = 12.0
_last_call_at   = 0.0

# ── Per-key state ─────────────────────────────────────────────────────────────
# _key_available_at[k]:
#   0.0          → available now
#   future ts    → in RPM cooldown until that monotonic time
#   float("inf") → dead for today (RPD exhausted or billing error)
_key_available_at: dict[str, float] = {k: 0.0 for k in GEMINI_API_KEYS}
_key_use_count:    dict[str, int]   = {k: 0   for k in GEMINI_API_KEYS}
_state_lock = threading.Lock()

# Populate quota state now that _state_lock exists
_quota_state.update(_load_quota_state())

_RPM_COOLDOWN_S = 65.0          # one 60s window + 5s safety buffer
_RPD_COOLDOWN_S = float("inf")  # daily / billing limit → dead for this process run

_RETRY_AFTER_RE = re.compile(r"retry[_ ]in[:\s]+(\d+\.?\d*)\s*s", re.IGNORECASE)

# Phrases that indicate a hard daily/billing quota — not recoverable by waiting 60s.
# "free_tier_requests" is Google's metric name for the 20 RPD multimodal free-tier limit;
# it appears even when the API also returns a short retry hint (misleading for daily caps).
_HARD_QUOTA_PHRASES = (
    "free_tier_requests",            # Google's daily free-tier metric (20/day multimodal)
    "billing", "check your plan", "upgrade", "per_day", "daily", "exhausted",
    "exceeded your current quota",   # Google's generic hard-limit message
)

# Phrases that narrow a hard-quota error back down to RPM (short per-minute window).
# Only applies if NONE of the absolute-dead indicators are present.
_RPM_WITHIN_HARD = (
    "per_minute", "requests_per_minute", "rpm",
)


def _parse_retry_after(err_str: str) -> float | None:
    m = _RETRY_AFTER_RE.search(err_str)
    return float(m.group(1)) + 3.0 if m else None


def _classify_error(err_str: str) -> str:
    """
    'rpm'   — per-minute rate limit; recoverable after ~60s
    'rpd'   — daily/billing quota; dead for this process run
    'other' — non-quota error
    """
    lower = err_str.lower()
    is_quota = "429" in err_str or "quota" in lower or "resource_exhausted" in lower
    if not is_quota:
        return "other"

    # "free_tier_requests" = Google's daily RPD metric → always dead, ignore retry hint
    if "free_tier_requests" in lower:
        return "rpd"

    # Other hard-limit phrases: only rescue as RPM if the error explicitly names a
    # per-minute metric (not just a short retry hint, which Google gives for daily caps too)
    if any(p in lower for p in _HARD_QUOTA_PHRASES):
        if any(p in lower for p in _RPM_WITHIN_HARD):
            return "rpm"
        return "rpd"

    if _parse_retry_after(err_str) is not None:
        return "rpm"

    return "rpm"   # ambiguous → treat as recoverable


def _mark_key(key: str, err_type: str, suggested_wait: float | None) -> None:
    save_needed = False
    with _state_lock:
        if err_type == "rpd":
            _key_available_at[key] = _RPD_COOLDOWN_S   # dead for this session
            kid = _key_id(key)
            if kid not in _quota_state:
                _quota_state[kid] = _default_key_state()
            _quota_state[kid]["rpd_used"] = RPD_LIMIT_PER_KEY  # saturate counter
            save_needed = True
        else:
            cooldown = suggested_wait if suggested_wait else _RPM_COOLDOWN_S
            _key_available_at[key] = time.monotonic() + cooldown
    if save_needed:
        _save_quota_state()


def _pick_key() -> tuple[str, float]:
    """
    Returns (key, wait_seconds).

    Strategy — sequential exhaustion:
      - Always pick the lowest-index key that is neither RPD-dead (API error this
        session) nor at its soft RPD limit (persistent counter >= RPD_LIMIT_PER_KEY).
      - If that key is ready now  → return (key, 0.0) and increment use_count.
      - If that key is RPM-cooling → return (key, remaining_wait); caller sleeps
        and must call _pick_key() again (same key will be returned once ready).
      - Only advances to the next key when the current one is RPD-exhausted.
      - Auto-resets per-key quota in memory if the reset timestamp has passed
        (handles processes that run across midnight without restart).
      - Raises RuntimeError with earliest reset time when every key is exhausted.
    """
    now     = time.monotonic()
    now_utc = datetime.now(timezone.utc)

    with _state_lock:
        for key in GEMINI_API_KEYS:
            # Skip if API reported hard RPD/billing error this session
            if _key_available_at[key] >= _RPD_COOLDOWN_S:
                continue

            # Ensure quota entry exists; auto-reset counter if a new day has started
            kid = _key_id(key)
            if kid not in _quota_state:
                _quota_state[kid] = _default_key_state()
            else:
                qs = _quota_state[kid]
                try:
                    reset_at = datetime.fromisoformat(qs["rpd_reset_at"])
                    if now_utc >= reset_at:
                        qs["rpd_used"] = 0
                        qs["rpd_reset_at"] = _next_midnight_utc()
                except Exception:
                    qs["rpd_used"] = 0
                    qs["rpd_reset_at"] = _next_midnight_utc()

            # Skip if soft daily limit reached
            if _quota_state[kid]["rpd_used"] >= RPD_LIMIT_PER_KEY:
                continue

            # Check RPM cooldown window
            wait = _key_available_at[key] - now
            if wait <= 0:
                _key_use_count[key] += 1
                return key, 0.0
            return key, wait  # RPM-cooling; do NOT increment use_count yet

        # ── All keys exhausted — report earliest reset time ───────────────────
        earliest_reset: datetime | None = None
        for key in GEMINI_API_KEYS:
            kid = _key_id(key)
            qs  = _quota_state.get(kid, {})
            try:
                reset_dt = datetime.fromisoformat(qs.get("rpd_reset_at", ""))
                if earliest_reset is None or reset_dt < earliest_reset:
                    earliest_reset = reset_dt
            except Exception:
                pass

        reset_msg = (
            earliest_reset.strftime("%Y-%m-%d %H:%M UTC")
            if earliest_reset else "midnight UTC"
        )
        raise RuntimeError(
            f"ALL API KEYS EXHAUSTED for today. Keys reset at {reset_msg}. "
            "Come back tomorrow or add more keys."
        )


def _build_contents(prompt: str, image_parts: list | None) -> list:
    """Build contents list in google-genai Part format."""
    from PIL import Image as _PILImage
    parts = [genai.types.Part.from_text(text=prompt)]
    for img in (image_parts or []):
        if isinstance(img, _PILImage.Image):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            parts.append(genai.types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
        elif isinstance(img, bytes):
            parts.append(genai.types.Part.from_bytes(data=img, mime_type="image/png"))
    return parts


# ── Public: key health-check ──────────────────────────────────────────────────

def validate_keys() -> dict[str, str]:
    """
    Test every API key with a minimal prompt.
    Returns {key_index_label: "ok" | "rpm" | "rpd" | "error:<msg>"}.
    Call this at pipeline startup to surface bad/exhausted keys immediately.
    """
    probe = "Reply with the single word: ok"
    results: dict[str, str] = {}
    n = len(GEMINI_API_KEYS)

    for i, key in enumerate(GEMINI_API_KEYS, 1):
        label = f"key[{i}/{n}]"
        try:
            client = genai.Client(api_key=key)
            resp   = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=probe,
                config=genai.types.GenerateContentConfig(max_output_tokens=5),
            )
            _ = resp.text
            results[label] = "ok"
            rprint(f"  [green]OK[/] {label} — alive")
        except Exception as e:
            err = str(e)
            etype = _classify_error(err)
            results[label] = etype if etype != "other" else f"error:{err[:80]}"
            colour = "red" if etype == "rpd" else "yellow"
            rprint(f"  [{colour}]XX[/] {label} — {etype} | {err[:120]}")
        time.sleep(13)   # stay under 5 RPM per key during validation

    alive = sum(1 for v in results.values() if v == "ok")
    rprint(f"\n  Key validation: {alive}/{n} keys alive")
    return results


# ── Public: main LLM call ─────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    image_parts: list | None = None,
    retries: int = 12,
    retry_delay: float = 65.0,
    _call_type: str = "text",
) -> str:
    """
    Strategy — sequential exhaustion (mirrors _pick_key):
    - Always use the lowest-index non-RPD-dead key.
    - On RPM 429: mark key with cooldown, sleep until IT recovers, retry same key.
    - On RPD 429 (billing/daily): mark key dead, next iteration advances to next key.
    - On non-quota error: sleep retry_delay, then retry same key.
    """
    global _last_call_at, _cache_hits, _cache_misses

    # ── Cache check ───────────────────────────────────────────────────────────
    cache_key = None
    if LLM_CACHE_ENABLED:
        cache_key = _compute_cache_key(prompt, image_parts)
        cached = _cache_load(cache_key)
        if cached is not None:
            _cache_hits += 1
            rprint(f"  [green]Cache HIT[/] [{cache_key[:8]}]")
            return cached

    last_error = None
    n_keys     = len(GEMINI_API_KEYS)

    # Global 5 RPM guard — sleep if less than 12s since last successful call
    with _state_lock:
        gap = _MIN_CALL_GAP_S - (time.monotonic() - _last_call_at)
    if gap > 0:
        rprint(f"  [dim]Rate limit guard — sleeping {gap:.1f}s (5 RPM)[/]")
        time.sleep(gap)

    for attempt in range(retries):
        key, wait = _pick_key()
        key_label = f"key[{GEMINI_API_KEYS.index(key) + 1}/{n_keys}]"

        if wait > 0:
            rprint(
                f"  [yellow]RPM cooldown — sleeping {wait + 2:.0f}s "
                f"({key_label} recovers, attempt {attempt+1}/{retries})[/]"
            )
            time.sleep(wait + 2.0)   # +2s buffer so window is truly clear
            # Re-pick: same key will now be returned ready; use_count incremented inside
            key, wait2 = _pick_key()
            if wait2 > 0:
                time.sleep(wait2 + 1.0)
                key, _ = _pick_key()
            key_label = f"key[{GEMINI_API_KEYS.index(key) + 1}/{n_keys}]"

        try:
            client   = genai.Client(api_key=key)
            rprint(f"  [dim]LLM call → {key_label} (use #{_key_use_count[key]})[/]")
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=_build_contents(prompt, image_parts),
                config=genai.types.GenerateContentConfig(
                    temperature=GENERATION_CONFIG["temperature"],
                    top_p=GENERATION_CONFIG["top_p"],
                    top_k=GENERATION_CONFIG["top_k"],
                    max_output_tokens=GENERATION_CONFIG["max_output_tokens"],
                ),
            )
            text = response.text
            with _state_lock:
                _last_call_at = time.monotonic()
                kid = _key_id(key)
                if kid not in _quota_state:
                    _quota_state[kid] = _default_key_state()
                _quota_state[kid]["rpd_used"] += 1
            _save_quota_state()
            if LLM_CACHE_ENABLED and cache_key:
                _cache_misses += 1
                _cache_save(cache_key, GEMINI_MODEL, text, _call_type)
                rprint(f"  [dim]Cache MISS → saved [{cache_key[:8]}][/]")
            return text

        except Exception as e:
            last_error = e
            err_str    = str(e)
            err_type   = _classify_error(err_str)
            suggested  = _parse_retry_after(err_str)

            if err_type in ("rpm", "rpd"):
                _mark_key(key, err_type, suggested)
                if err_type == "rpd":
                    tag = "RPD/BILLING — key marked dead"
                elif suggested:
                    tag = f"RPM — cooldown {suggested:.0f}s"
                else:
                    tag = f"RPM — cooldown {_RPM_COOLDOWN_S:.0f}s"
                rprint(f"  [red]{tag}[/] {key_label} | {err_str[:150]}")
            else:
                rprint(
                    f"  [red]LLM error[/] {key_label} "
                    f"(attempt {attempt+1}/{retries}): {err_str[:300]}"
                )
                if attempt < retries - 1:
                    time.sleep(retry_delay)

    raise RuntimeError(f"All {retries} LLM attempts failed. Last: {last_error}")


def call_llm_json(
    prompt: str,
    image_parts: list | None = None,
    retries: int = 12,
) -> str:
    """Like call_llm but enforces JSON-only output and strips markdown fences."""
    json_prompt = (
        prompt
        + "\n\nCRITICAL: Respond ONLY with valid JSON. No markdown fences, no explanation, no trailing text."
    )
    raw = call_llm(json_prompt, image_parts, retries=retries, _call_type="json")

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    return raw


def call_llm_with_feedback(
    original_prompt: str,
    error_feedback: str,
    previous_output: str,
    image_parts: list | None = None,
) -> str:
    """Re-send to Gemini with Auditor error feedback attached. Used by Coder retry loop."""
    retry_prompt = f"""{original_prompt}

--- PREVIOUS OUTPUT (contained errors) ---
{previous_output[:3000]}

--- ERROR FEEDBACK FROM AUDITOR ---
{error_feedback}

Fix ALL issues listed in the error feedback. Respond with corrected output only."""
    return call_llm(retry_prompt, image_parts)
