"""
Centralized LLM wrapper — per-key cooldown tracking.
All agents call this module — model version is enforced here.
DO NOT DOWNGRADE OR CHANGE MODEL VERSION (config.py).
"""

import re
import time
import threading
import google.generativeai as genai
from rich import print as rprint

from config import GEMINI_MODEL, GEMINI_API_KEYS, GENERATION_CONFIG

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
    with _state_lock:
        if err_type == "rpd":
            _key_available_at[key] = _RPD_COOLDOWN_S   # dead
        else:
            cooldown = suggested_wait if suggested_wait else _RPM_COOLDOWN_S
            _key_available_at[key] = time.monotonic() + cooldown


def _pick_key() -> tuple[str, float]:
    """
    Returns (key, wait_seconds).

    Strategy — sequential exhaustion:
      - Always pick the lowest-index key that is not RPD-dead.
      - If that key is ready now  → return (key, 0.0) and increment use_count.
      - If that key is RPM-cooling → return (key, remaining_wait); caller sleeps
        and must call _pick_key() again (same key will be returned once ready).
      - Only advance to the next key when the current one is RPD-dead.
      - Raises RuntimeError only when every key is RPD-dead.
    """
    now = time.monotonic()
    with _state_lock:
        for key in GEMINI_API_KEYS:
            if _key_available_at[key] >= _RPD_COOLDOWN_S:
                continue  # RPD-dead — skip permanently
            wait = _key_available_at[key] - now
            if wait <= 0:
                _key_use_count[key] += 1
                return key, 0.0
            return key, wait  # RPM-cooling; do NOT increment use_count yet

        raise RuntimeError(
            f"All {len(GEMINI_API_KEYS)} API keys have hit their daily/billing quota limit. "
            "Check each account's Google AI Studio dashboard. "
            "Reset happens at midnight Pacific time."
        )


def _get_model(api_key: str) -> genai.GenerativeModel:
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config=GENERATION_CONFIG,
    )


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
            genai.configure(api_key=key)
            model = genai.GenerativeModel(GEMINI_MODEL, generation_config={"max_output_tokens": 5})
            resp  = model.generate_content(probe)
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
) -> str:
    """
    Strategy — sequential exhaustion (mirrors _pick_key):
    - Always use the lowest-index non-RPD-dead key.
    - On RPM 429: mark key with cooldown, sleep until IT recovers, retry same key.
    - On RPD 429 (billing/daily): mark key dead, next iteration advances to next key.
    - On non-quota error: sleep retry_delay, then retry same key.
    """
    global _last_call_at
    content    = [prompt] + (image_parts or [])
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
            model    = _get_model(key)
            rprint(f"  [dim]LLM call → {key_label} (use #{_key_use_count[key]})[/]")
            response = model.generate_content(content)
            with _state_lock:
                _last_call_at = time.monotonic()
            return response.text

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
    raw = call_llm(json_prompt, image_parts, retries=retries)

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
