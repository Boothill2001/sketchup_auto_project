"""
Streamlit Web UI — PDF to SketchUp LOD 300 Converter
Run: streamlit run app.py
"""

import sys
import threading
import queue
from pathlib import Path
from datetime import datetime

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from config import INPUT_PDF_DIR, CODER_OUTPUT_FILE, SKP_OUTPUT_FILE, RPD_LIMIT_PER_KEY
from main import run_pipeline
from core.llm_wrapper import (
    get_cache_stats, clear_cache,
    get_key_quota_status, any_key_available,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF → SketchUp LOD 300",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size: 2.1rem; font-weight: 800; color: #1E3A5F; line-height:1.1; }
    .sub-caption { color: #6B7280; font-size: 0.9rem; margin-top: -4px; margin-bottom: 12px; }
    div[data-testid="metric-container"] {
        background: #F0F9FF; border: 1px solid #BAE6FD;
        border-radius: 10px; padding: 12px 8px; text-align: center;
    }
    .warn-box {
        background: #FFFBEB; border: 1px solid #FCD34D;
        border-radius: 8px; padding: 12px; font-size: 0.88rem;
    }
    .stDownloadButton > button {
        background: #2563EB !important; color: white !important;
        font-size: 1.05rem !important; padding: 12px 24px !important;
        border-radius: 8px !important; font-weight: 700 !important;
        width: 100%;
    }
    .stDownloadButton > button:hover { background: #1D4ED8 !important; }
    /* Subtle tab styling */
    button[data-baseweb="tab"] { font-size: 0.97rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
h1, h2 = st.columns([1, 14])
with h1:
    st.markdown("## 🏗️")
with h2:
    st.markdown('<p class="main-title">PDF to SketchUp LOD 300 Converter</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-caption">Multi-Agent AI &nbsp;·&nbsp; Gemini 2.5 Flash &nbsp;·&nbsp; '
        'Structural Steel → Ruby Script</p>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Layout: sidebar-style left + main right ───────────────────────────────────
left, right = st.columns([1.1, 1.9], gap="large")

# ═══════════════════════════════════════════
# LEFT — Upload + controls
# ═══════════════════════════════════════════
with left:
    st.subheader("📂 Upload PDF")
    uploaded_file = st.file_uploader(
        "Drag & drop or browse", type=["pdf"], label_visibility="collapsed"
    )
    if uploaded_file:
        kb = len(uploaded_file.getvalue()) / 1024
        st.success(f"✅ **{uploaded_file.name}**\n\n{kb:.1f} KB")

    st.divider()
    st.subheader("🌐 Vertex AI Status")
    st.success("✅ Service Account connected\n\nProject: `river-bedrock-496101-a7`")

    st.divider()
    st.subheader("⚙️ Options")
    auto_scroll_log = st.toggle("Auto-scroll terminal log", value=True)
    st.caption("Keep last 60 lines visible while pipeline runs.")
    st.divider()
    st.subheader("🗑️ Cache")
    _stats_sidebar = get_cache_stats()
    st.caption(f"Hits: {_stats_sidebar['hits']}  ·  Misses: {_stats_sidebar['misses']}")
    if st.button("Clear LLM Cache", use_container_width=True):
        n = clear_cache()
        st.success(f"Cleared {n} cache file(s).")
    st.divider()

    run_disabled = uploaded_file is None
    run_btn = st.button(
        "▶  Start Conversion",
        type="primary",
        disabled=run_disabled,
        use_container_width=True,
    )
    if run_disabled:
        st.caption("⬆️ Upload a PDF to enable.")

# ═══════════════════════════════════════════
# RIGHT — Tabs
# ═══════════════════════════════════════════
with right:
    tab_dash, tab_term = st.tabs(["📊  Dashboard", "👨‍💻  Terminal & Debug Logs"])

    # ── Tab 1: Dashboard (clean, user-facing) ──────────────────────────────
    with tab_dash:
        progress_slot  = st.empty()   # progress bar during run
        status_slot    = st.empty()   # status message
        metrics_slot   = st.empty()   # 4 metrics after run
        warning_slot   = st.empty()   # unmapped warning
        download_slot  = st.empty()   # download button

        if not run_btn:
            status_slot.info(
                "Pipeline idle. Upload a structural PDF and press **Start Conversion**."
            )

    # ── Tab 2: Terminal (technical, admin-facing) ───────────────────────────
    with tab_term:
        term_slot      = st.empty()   # live scrolling log box
        log_dl_slot    = st.empty()   # download-log button (appears after run)

        if not run_btn:
            term_slot.code("Waiting for pipeline to start...", language=None)


# ═══════════════════════════════════════════
# PIPELINE EXECUTION
# ═══════════════════════════════════════════
PHASE_ICONS = {
    "SCANNER": "🔍", "GLOSSARY": "📖", "SCHEDULE": "📋",
    "SPATIAL": "📐", "MAPPER": "🗺️",  "CODER": "💻",
    "AUDITOR": "🔎", "RETRY": "🔄",   "COMPLETE": "✅", "ERROR": "❌",
}
PHASE_WEIGHTS = {
    "SCANNER": 10, "GLOSSARY": 5, "SCHEDULE": 20,
    "SPATIAL": 15, "MAPPER": 25, "CODER": 15, "AUDITOR": 10,
}

def _icon(line: str) -> str:
    upper = line.upper()
    for k, v in PHASE_ICONS.items():
        if k in upper:
            return v
    return " ·"


if run_btn and uploaded_file:

    # Pre-flight: verify Vertex AI connection
    if not any_key_available():
        with tab_dash:
            status_slot.error(
                "❌ **Vertex AI connection unavailable.**  \n"
                "Check that the service account JSON key file exists in the project root "
                "and the project `river-bedrock-496101-a7` has the Vertex AI API enabled."
            )
        st.stop()

    # Estimate API calls needed for user awareness
    _pdf_bytes  = uploaded_file.getvalue()
    _n_pages    = max(1, round(len(_pdf_bytes) / (80 * 1024)))  # ~80 KB per page heuristic
    _est_calls  = _n_pages * 5
    with tab_dash:
        status_slot.info(
            f"📄 This PDF (~{_n_pages} pages) will use ~{_est_calls} API calls. Starting pipeline…"
        )

    # 1. Save PDF to disk
    Path(INPUT_PDF_DIR).mkdir(parents=True, exist_ok=True)
    save_path = Path(INPUT_PDF_DIR) / uploaded_file.name
    save_path.write_bytes(uploaded_file.getvalue())

    # 2. Shared state
    log_q: queue.Queue[str] = queue.Queue()
    log_lines: list[str] = []
    result_holder: dict = {}

    def _log_fn(msg: str) -> None:
        log_q.put(msg)

    def _run():
        try:
            res = run_pipeline(str(save_path), log_fn=_log_fn)
        except Exception as exc:
            res = {
                "ruby_path": None, "error": str(exc),
                "members_total": 0, "placed": 0,
                "unmapped": 0, "unmapped_marks": [], "audit_passed": False,
            }
        result_holder.update(res)
        log_q.put("__DONE__")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # 3. Live streaming into BOTH tabs simultaneously
    current_pct = 0

    with tab_dash:
        progress_slot.progress(0, text="Initialising pipeline...")
        status_slot.empty()

    while True:
        try:
            msg = log_q.get(timeout=0.4)
        except queue.Empty:
            continue

        if msg == "__DONE__":
            break

        log_lines.append(msg)

        # ── Advance progress bar (Dashboard) ──
        upper = msg.upper()
        for phase_key, weight in PHASE_WEIGHTS.items():
            if phase_key in upper and "PHASE" in upper:
                current_pct = min(current_pct + int(weight / sum(PHASE_WEIGHTS.values()) * 100), 95)
                with tab_dash:
                    progress_slot.progress(current_pct, text=f"Running: {msg[:55]}…")
                break

        # ── Update terminal (Tab 2) ──
        display_lines = log_lines[-60:] if auto_scroll_log else log_lines
        display = "\n".join(f"{_icon(ln)}  {ln}" for ln in display_lines)
        with tab_term:
            term_slot.code(display, language=None)

    # 4. Final progress tick
    with tab_dash:
        progress_slot.progress(100, text="Done.")

    # ── Populate Dashboard (Tab 1) ─────────────────────────────────────────
    error     = result_holder.get("error")
    ruby_path = result_holder.get("ruby_path")

    with tab_dash:
        if error:
            status_slot.error(f"❌ **Pipeline failed:** {error}")
            progress_slot.empty()
        else:
            status_slot.success("✅ Pipeline complete!")

            # Metrics row
            with metrics_slot.container():
                st.markdown("#### 📈 Extraction Summary")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total Members",  result_holder.get("members_total", 0))
                c2.metric("3D Placed",       result_holder.get("placed", 0))
                c3.metric("⚠️ Unmapped",     result_holder.get("unmapped", 0))
                c4.metric("Audit",           "✅ Pass" if result_holder.get("audit_passed") else "⚠️ Warn")
                c5.metric("API Calls Saved", get_cache_stats()["hits"])

            unmapped_marks = result_holder.get("unmapped_marks", [])
            if unmapped_marks:
                warning_slot.markdown(
                    f'<div class="warn-box">⚠️ <b>{len(unmapped_marks)} member(s)</b> could not be '
                    f'auto-placed: <code>{", ".join(unmapped_marks)}</code><br>'
                    f'In SketchUp → filter layer <b>LOD300_UNMAPPED_NEEDS_REVIEW</b></div>',
                    unsafe_allow_html=True,
                )

        # Download buttons (always show if .rb exists, even on audit warn)
        if ruby_path and Path(ruby_path).exists():
            rb_bytes = Path(ruby_path).read_bytes()
            rb_name  = f"lod300_{uploaded_file.name.replace('.pdf', '')}.rb"
            skp_path = Path(SKP_OUTPUT_FILE)
            with download_slot.container():
                st.divider()
                st.markdown("#### 📥 Download Output Files")
                st.download_button(
                    label=f"⬇️  Download  {rb_name}",
                    data=rb_bytes,
                    file_name=rb_name,
                    mime="text/plain",
                    use_container_width=True,
                )
                st.caption(
                    f"SketchUp: **Extensions → Ruby Console** → `load 'path/to/{rb_name}'`  "
                    f"&nbsp;·&nbsp; {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
                if skp_path.exists():
                    skp_bytes = skp_path.read_bytes()
                    skp_mb    = len(skp_bytes) / (1024 * 1024)
                    st.download_button(
                        label="⬇️  Download  lod300_model.skp",
                        data=skp_bytes,
                        file_name="lod300_model.skp",
                        mime="application/octet-stream",
                        use_container_width=True,
                    )
                    st.caption(f"SketchUp model · {skp_mb:.1f} MB · open directly in SketchUp Pro")
                else:
                    st.caption(
                        "💡 **lod300_model.skp** will appear here after you run the .rb script in SketchUp "
                        "(it auto-saves to the same folder)."
                    )

    # ── Populate Terminal (Tab 2) ──────────────────────────────────────────
    full_log = "\n".join(log_lines)
    with tab_term:
        # Show full log (not truncated)
        term_slot.code(full_log, language=None)

        # Download log button
        log_dl_slot.download_button(
            label="📄  Download pipeline_debug.log",
            data=full_log.encode("utf-8"),
            file_name=f"pipeline_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            mime="text/plain",
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>PDF to SketchUp LOD 300 &nbsp;·&nbsp; Gemini 2.5 Flash "
    "&nbsp;·&nbsp; Multi-Agent AI Pipeline</small></center>",
    unsafe_allow_html=True,
)
