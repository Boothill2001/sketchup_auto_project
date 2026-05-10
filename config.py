# =============================================================================
# SYSTEM CONFIGURATION — PDF to SketchUp LOD 300 Multi-Agent Pipeline
# =============================================================================
# DO NOT DOWNGRADE OR CHANGE MODEL VERSION
# Model is locked to gemini-2.5-flash for multimodal context window + OCR depth.
# Any deviation will corrupt structured output contracts between agents.
# =============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

_base = Path(__file__).parent
# Load .env first; fall back to .env.example if .env doesn't exist
_env_file = _base / ".env" if (_base / ".env").exists() else _base / ".env.example"
load_dotenv(_env_file)

# ---- MODEL LOCK ----
GEMINI_MODEL = "gemini-2.5-flash"  # DO NOT DOWNGRADE OR CHANGE MODEL VERSION

# ---- API KEY POOL (Round-Robin to bypass rate limits) ----
# Keys sourced from .env — add more GEMINI_KEY_N entries to expand the pool.
_RAW_KEYS = [
    os.environ.get("GEMINI_KEY",   ""),
    os.environ.get("GEMINI_KEY_2", ""),
    os.environ.get("GEMINI_KEY_3", ""),
    os.environ.get("GEMINI_KEY_4", ""),
    os.environ.get("GEMINI_KEY_5", ""),
    os.environ.get("GEMINI_KEY_6", ""),
]
GEMINI_API_KEYS: list[str] = [k.strip() for k in _RAW_KEYS if k.strip()]

if not GEMINI_API_KEYS:
    raise EnvironmentError("No GEMINI_KEY_* found in .env — add at least one API key.")

# ---- PATHS ----
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
INPUT_PDF_DIR   = os.path.join(BASE_DIR, "data", "input_pdf")
OUTPUT_JSON_DIR = os.path.join(BASE_DIR, "data", "output_json")
RUBY_OUTPUT_DIR = os.path.join(BASE_DIR, "output", "final_ruby_scripts")

# ---- AGENT OUTPUT FILES ----
SCANNER_OUTPUT_FILE        = os.path.join(OUTPUT_JSON_DIR, "drawing_index.json")
GLOSSARY_OUTPUT_FILE       = os.path.join(OUTPUT_JSON_DIR, "glossary.json")
SCHEDULE_OUTPUT_FILE       = os.path.join(OUTPUT_JSON_DIR, "steel_schedule.json")
SPATIAL_OUTPUT_FILE        = os.path.join(OUTPUT_JSON_DIR, "spatial_data.json")
MAPPED_OUTPUT_FILE         = os.path.join(OUTPUT_JSON_DIR, "mapped_members.json")
CODER_OUTPUT_FILE          = os.path.join(RUBY_OUTPUT_DIR, "lod300_model.rb")
AUDITOR_REPORT_FILE        = os.path.join(OUTPUT_JSON_DIR, "audit_report.json")

# ---- LLM GENERATION SETTINGS ----
GENERATION_CONFIG = {
    "temperature": 0.1,      # Low temp → deterministic, data-accurate output
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

# ---- PDF PROCESSING ----
SCANNER_DPI = 100      # Low-res for page classification — fast, no detail needed
PDF_DPI     = 200      # Mid-res for schedule/spatial extraction (was 300, too slow)
OPENCV_ENABLED = True  # Auto-segment multi-drawing pages with contour detection

# ---- PARALLELISM ----
# Gemini 2.5 Flash free tier (multimodal): 5 RPM / 20 RPD *per project*, not per key.
# Round-robin across keys only helps if keys are from DIFFERENT Google projects.
# If all keys are in the same project, use sequential (max_workers=1) + 20s sleep.
SCANNER_MAX_WORKERS = 1

# ---- LOD TARGET ----
LOD_LEVEL = 300

# ---- AUDITOR RETRY SETTINGS ----
MAX_AUDIT_RETRIES = 2  # How many times Coder re-runs if Auditor finds issues
