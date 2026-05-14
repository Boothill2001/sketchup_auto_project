"""
Centralized LLM wrapper — Vertex AI Service Account authentication.
All agents call this module — model version is enforced here.
DO NOT DOWNGRADE OR CHANGE MODEL VERSION (config.py).
"""

import io
import time
import json
import hashlib
import threading
from datetime import datetime
from pathlib import Path

from google import genai
from rich import print as rprint

from config import (
    GEMINI_MODEL, VERTEX_PROJECT, VERTEX_LOCATION,
    GENERATION_CONFIG,
    LLM_CACHE_ENABLED, LLM_CACHE_DIR,
)

# ── Vertex AI client — single instance, auth via GOOGLE_APPLICATION_CREDENTIALS
_client = genai.Client(
    vertexai=True,
    project=VERTEX_PROJECT,
    location=VERTEX_LOCATION,
)

# ── Cache stats ───────────────────────────────────────────────────────────────
_cache_hits   = 0
_cache_misses = 0
_state_lock   = threading.Lock()


def get_cache_stats() -> dict:
    return {"hits": _cache_hits, "misses": _cache_misses}


def _img_bytes(part) -> bytes | None:
    if isinstance(part, bytes):
        return part
    if hasattr(part, "tobytes") and hasattr(part, "mode"):  # PIL Image
        try:
            buf = io.BytesIO()
            part.save(buf, format="PNG")
            return buf.getvalue()
        except Exception:
            return part.tobytes()
    if hasattr(part, "inline_data"):
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
        _raw = path.read_text(encoding="utf-8")
        try:
            data = json.loads(_raw)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json
                data = json.loads(repair_json(_raw))
            except Exception:
                raise
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


# ── Rate limiter — light 1 s gap (Vertex AI allows 60+ RPM) ─────────────────
_MIN_CALL_GAP_S = 1.0
_last_call_at   = 0.0


def _build_contents(prompt: str, image_parts: list | None) -> list:
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


# ── Backward-compat stubs (used by app.py sidebar + main.py budget) ──────────

def get_key_quota_status() -> list[dict]:
    return [{
        "index":      1,
        "key_id":     "vertex-ai",
        "rpd_used":   0,
        "rpd_limit":  999999,
        "exhausted":  False,
        "rpd_reset_at": "",
    }]


def any_key_available() -> bool:
    return True


# ── Public: health-check ──────────────────────────────────────────────────────

def validate_keys() -> dict[str, str]:
    probe = "Reply with the single word: ok"
    try:
        resp = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=probe,
            config=genai.types.GenerateContentConfig(max_output_tokens=5),
        )
        _ = resp.text
        rprint(f"  [green]OK[/] Vertex AI — alive")
        rprint(f"  [dim]Project: {VERTEX_PROJECT} | Location: {VERTEX_LOCATION}[/]")
        rprint(f"\n  Vertex AI: 1/1 connection alive")
        return {"vertex_ai": "ok"}
    except Exception as e:
        err = str(e)
        rprint(f"  [red]XX[/] Vertex AI — error | {err[:200]}")
        rprint(f"\n  Vertex AI: 0/1 connection alive")
        return {"vertex_ai": f"error:{err[:80]}"}


# ── Public: main LLM call ─────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    image_parts: list | None = None,
    retries: int = 6,
    retry_delay: float = 30.0,
    _call_type: str = "text",
) -> str:
    global _last_call_at, _cache_hits, _cache_misses

    if LLM_CACHE_ENABLED:
        cache_key = _compute_cache_key(prompt, image_parts)
        cached = _cache_load(cache_key)
        if cached is not None:
            _cache_hits += 1
            rprint(f"  [green]Cache HIT[/] [{cache_key[:8]}]")
            return cached
    else:
        cache_key = None

    with _state_lock:
        gap = _MIN_CALL_GAP_S - (time.monotonic() - _last_call_at)
    if gap > 0:
        time.sleep(gap)

    last_error = None
    for attempt in range(retries):
        try:
            rprint(f"  [dim]LLM call -> Vertex AI (attempt {attempt+1}/{retries})[/]")
            response = _client.models.generate_content(
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
            if LLM_CACHE_ENABLED and cache_key:
                _cache_misses += 1
                _cache_save(cache_key, GEMINI_MODEL, text, _call_type)
                rprint(f"  [dim]Cache MISS -> saved [{cache_key[:8]}][/]")
            return text

        except Exception as e:
            last_error = e
            err_str = str(e)
            rprint(f"  [red]LLM error[/] (attempt {attempt+1}/{retries}): {err_str[:300]}")
            if attempt < retries - 1:
                wait = 65.0 if "429" in err_str else retry_delay
                rprint(f"  [dim]Retrying in {wait:.0f}s...[/]")
                time.sleep(wait)

    raise RuntimeError(f"All {retries} LLM attempts failed. Last: {last_error}")


def call_llm_json(
    prompt: str,
    image_parts: list | None = None,
    retries: int = 6,
) -> str:
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
    retry_prompt = f"""{original_prompt}

--- PREVIOUS OUTPUT (contained errors) ---
{previous_output[:3000]}

--- ERROR FEEDBACK FROM AUDITOR ---
{error_feedback}

Fix ALL issues listed in the error feedback. Respond with corrected output only."""
    return call_llm(retry_prompt, image_parts)
