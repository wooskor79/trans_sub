import streamlit as st
import os
import io
import zipfile
import asyncio
from dotenv import load_dotenv

import utils
import trans_nllb
import trans_gemini
import trans_deepl
import trans_claude

# ======================
# SETUP & STYLE
# ======================
utils.setup_logging()
load_dotenv()

st.set_page_config(page_title="Ultra Subtitle Translator", layout="wide", page_icon="🎬")

# Custom CSS for Premium Look
st.markdown("""
<style>
    /* Global Background & Font */
    .stApp {
        background-color: #0e1117;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #e6edf3 !important;
        font-weight: 600;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #21262d;
        border-radius: 8px;
        color: #8b949e;
        border: 1px solid #30363d;
        padding: 0 16px;
        font-size: 14px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #30363d;
        color: #c9d1d9;
    }
    .stTabs [aria-selected="true"] {
        background-color: #238636 !important;
        color: white !important;
        border: none !important;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        color: #58a6ff;
        font-size: 1.8rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #8b949e;
    }
    
    /* File Uploader */
    section[data-testid="stFileUploader"] {
        background-color: #161b22;
        border: 1px dashed #30363d;
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: bold;
        background-color: #238636;
        color: white;
        border: none;
        height: 48px;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background-color: #2ea043;
        box-shadow: 0 4px 12px rgba(35, 134, 54, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ======================
# SIDEBAR CONTROLS
# ======================
with st.sidebar:
    st.title("🎬 Ultra Transl")
    st.caption("v2.0 | RTX 5080 Optimized")
    
    st.markdown("### 🔑 API Keys")
    GEMINI_API_KEY = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    DEEPL_API_KEY = st.text_input("DeepL API Key", value=os.getenv("DEEPL_API_KEY", ""), type="password")
    CLAUDE_API_KEY = st.text_input("Claude API Key", value=os.getenv("CLAUDE_API_KEY", ""), type="password")
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    
    # Refresh VRAM button (hidden logic, just auto updates on interaction)
    u_vram, t_vram = utils.get_vram_status()
    st.metric("GPU VRAM", f"{u_vram:.1f} GB", f"Total {t_vram:.1f} GB")
    
    used_d, limit_d = trans_deepl.get_usage(DEEPL_API_KEY)
    if used_d is not None:
        safe_limit = limit_d if limit_d and limit_d > 0 else 500_000
        pct = (used_d / safe_limit)
        st.metric("DeepL Usage", f"{int(pct*100)}%", f"{used_d:,} / {safe_limit:,} chars")
        st.progress(min(pct, 1.0))
    else:
        st.metric("DeepL Usage", "Offline", "Check API Key")

# ======================
# MAIN CONTENT
# ======================
st.subheader("Select Translation Engine")

tab_titles = [
    "🚀 NLLB (Local GPU)", 
    "✨ Gemini Flash (Ultra)", 
    "🌐 DeepL Pro", 
    "🤖 Claude Sonnet"
]
tabs = st.tabs(tab_titles)

# [TAB 1] NLLB
with tabs[0]:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.info("💡 **Local GPU Powerhouse**: Uses RTX 5080 optimized FP16/CUDA inference. Best for privacy and unlimited usage.")
    files = st.file_uploader("Upload SRT Files", type=["srt"], accept_multiple_files=True, key="nllb_up")
    
    if st.button("Start NLLB Translation", type="primary") and files:
        start_dt = utils.get_now()
        status_area = st.empty()
        zip_buf = io.BytesIO()
        
        # Load Model
        with st.spinner("Loading NLLB-3.3B Model to VRAM..."):
            tok, mdl = trans_nllb.load_model("facebook/nllb-200-3.3B")
            
        with zipfile.ZipFile(zip_buf, "w") as z:
            for idx, f in enumerate(files, 1):
                raw_text = f.read().decode("utf-8", "ignore")
                rows = utils.parse_srt(raw_text)
                
                out = trans_nllb.translate(rows, tok, mdl, status_area, f.name, idx, len(files))
                
                built_srt = utils.build_srt([[r[0], r[1], t] for r, t in zip(rows, out)])
                z.writestr(f"KR_{f.name}", built_srt)
                
        end_dt = utils.get_now()
        status_area.empty()
        st.success(f"🎉 All Completed in {utils.format_duration(start_dt, end_dt)}")
        st.download_button("📥 Download Result ZIP", zip_buf.getvalue(), "NLLB_Translated.zip")

# [TAB 2] Gemini
with tabs[1]:
    st.info("💡 **Gemini 2.0 Flash**: Context-aware translation (±3 lines). Tuned for natural Korean subtitles.")
    polish_mode = st.toggle("🛠️ Polishing Mode (Input is already Korean)", value=False)
    files = st.file_uploader("Upload SRT Files", type=["srt"], accept_multiple_files=True, key="gemini_up")
    
    if st.button("Start Gemini Translation", type="primary") and files:
        if not GEMINI_API_KEY:
            st.error("⚠️ Please enter Gemini API Key in the sidebar.")
        else:
            start_dt = utils.get_now()
            status_area = st.empty()
            zip_buf = io.BytesIO()
            model_name = "gemini-2.0-flash"
            
            with zipfile.ZipFile(zip_buf, "w") as z:
                for idx, f in enumerate(files, 1):
                    raw_text = f.read().decode("utf-8", "ignore")
                    rows = utils.parse_srt(raw_text)
                    
                    out = asyncio.run(trans_gemini.translate_async(
                        rows, GEMINI_API_KEY, model_name, status_area, f.name, polish_mode, idx, len(files)
                    ))
                    
                    built_srt = utils.build_srt([[r[0], r[1], t] for r, t in zip(rows, out)])
                    z.writestr(f"KR_{f.name}", built_srt)

            end_dt = utils.get_now()
            status_area.empty()
            st.success(f"🎉 All Completed in {utils.format_duration(start_dt, end_dt)}")
            st.download_button("📥 Download Result ZIP", zip_buf.getvalue(), "Gemini_Translated.zip")

# [TAB 3] DeepL
with tabs[2]:
    st.info("💡 **DeepL Pro**: Industry standard accuracy. Sequential processing to avoid rate limits.")
    files = st.file_uploader("Upload SRT Files", type=["srt"], accept_multiple_files=True, key="deepl_up")
    
    if st.button("Start DeepL Translation", type="primary") and files:
        if not DEEPL_API_KEY:
            st.error("⚠️ Please enter DeepL API Key in the sidebar.")
        else:
            start_dt = utils.get_now()
            status_area = st.empty()
            zip_buf = io.BytesIO()
            
            with zipfile.ZipFile(zip_buf, "w") as z:
                for idx, f in enumerate(files, 1):
                    raw_text = f.read().decode("utf-8", "ignore")
                    rows = utils.parse_srt(raw_text)
                    
                    out = asyncio.run(trans_deepl.translate_async(
                        rows, DEEPL_API_KEY, status_area, f.name, idx, len(files)
                    ))
                    
                    built_srt = utils.build_srt([[r[0], r[1], t] for r, t in zip(rows, out)])
                    z.writestr(f"KR_{f.name}", built_srt)
            
            end_dt = utils.get_now()
            status_area.empty()
            st.success(f"🎉 All Completed in {utils.format_duration(start_dt, end_dt)}")
            st.download_button("📥 Download Result ZIP", zip_buf.getvalue(), "DeepL_Translated.zip")

# [TAB 4] Claude
with tabs[3]:
    st.info("💡 **Claude 3.5 Sonnet**: High nuance understanding.")
    polish_mode_c = st.toggle("🛠️ Polishing Mode", value=False, key="c_polish")
    files = st.file_uploader("Upload SRT Files", type=["srt"], accept_multiple_files=True, key="claude_up")
    
    if st.button("Start Claude Translation", type="primary") and files:
        if not CLAUDE_API_KEY:
            st.error("⚠️ Please enter Claude API Key in the sidebar.")
        else:
            start_dt = utils.get_now()
            status_area = st.empty()
            zip_buf = io.BytesIO()
            
            with zipfile.ZipFile(zip_buf, "w") as z:
                for idx, f in enumerate(files, 1):
                    raw_text = f.read().decode("utf-8", "ignore")
                    rows = utils.parse_srt(raw_text)
                    
                    out = asyncio.run(trans_claude.translate_async(
                        rows, CLAUDE_API_KEY, status_area, f.name, polish_mode_c, idx, len(files)
                    ))
                    
                    built_srt = utils.build_srt([[r[0], r[1], t] for r, t in zip(rows, out)])
                    z.writestr(f"KR_{f.name}", built_srt)
            
            end_dt = utils.get_now()
            status_area.empty()
            st.success(f"🎉 All Completed in {utils.format_duration(start_dt, end_dt)}")
            st.download_button("📥 Download Result ZIP", zip_buf.getvalue(), "Claude_Translated.zip")