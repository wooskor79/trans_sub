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
# SETUP
# ======================
utils.setup_logging()
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

st.set_page_config(page_title="Subtitle Translator", layout="wide")

# ======================
# MAIN UI
# ======================
_, center_col, _ = st.columns([1, 2.5, 1])

with center_col:
    st.title("🚀 Subtitle Translator")

    # ---- Status Bar ----
    u_vram, t_vram = utils.get_vram_status()
    col1, col2 = st.columns(2)

    with col1:
        st.metric("VRAM Usage", f"{u_vram:.1f} / {t_vram:.1f} GB")

    with col2:
        used_d, limit_d = trans_deepl.get_usage(DEEPL_API_KEY)
        if used_d is not None:
            safe_limit = limit_d if limit_d and limit_d > 0 else 500_000
            usage_pct = (used_d / safe_limit) * 100
            st.metric(
                "DeepL Usage",
                f"{used_d:,} / {safe_limit:,}",
                delta=f"{usage_pct:.1f}% Used",
                delta_color="inverse"
            )
            st.progress(min(used_d / safe_limit, 1.0))
        else:
            st.metric("DeepL Usage", "Offline")

    st.markdown("---")

    # ======================
    # TABS (🔥 핵심: 이 블록 절대 밖으로 나가면 안 됨)
    # ======================
    tabs = st.tabs([
        "[GPU] NLLB",
        "[API] Gemini (최상급)",
        "[API] DeepL",
        "[API] Claude"
    ])

    # ======================
    # [GPU] NLLB
    # ======================
    with tabs[0]:
        st.subheader("🧠 GPU 기반 NLLB 번역")

        files = st.file_uploader(
            "SRT 업로드 (NLLB)",
            type=["srt"],
            accept_multiple_files=True,
            key="nllb_uploader"
        )

        if st.button("▶ NLLB 번역 시작") and files:
            status = st.empty()
            zip_buf = io.BytesIO()

            tok, mdl = trans_nllb.load_model("facebook/nllb-200-distilled-600M")

            with zipfile.ZipFile(zip_buf, "w") as z:
                for idx, f in enumerate(files, 1):
                    rows = utils.parse_srt(f.read().decode("utf-8", "ignore"))
                    out = trans_nllb.translate(rows, tok, mdl, status, f.name, idx, len(files))
                    z.writestr(f.name, utils.build_srt([[r[0], r[1], t] for r, t in zip(rows, out)]))

            st.success("✅ NLLB 번역 완료")
            st.download_button("⬇ 결과 ZIP 다운로드", zip_buf.getvalue(), "nllb.zip")

    # ======================
    # [API] Gemini
    # ======================
    with tabs[1]:
        model_choice = st.radio(
            "🧠 Gemini 모델 선택",
            ["Flash (빠름)", "Pro (최상급 품질)"],
            horizontal=True
        )

        polish = st.toggle("🛠 한국어 자막 윤문 모드", value=False)

        files = st.file_uploader(
            "SRT 업로드 (Gemini)",
            type=["srt"],
            accept_multiple_files=True,
            key="gemini_uploader"
        )

        if st.button("▶ Gemini 번역 시작") and files:
            if not GEMINI_API_KEY:
                st.error("Gemini API Key가 없습니다.")
            else:
                model_name = "gemini-2.0-pro" if "Pro" in model_choice else "gemini-2.0-flash"
                status = st.empty()
                zip_buf = io.BytesIO()

                with zipfile.ZipFile(zip_buf, "w") as z:
                    for idx, f in enumerate(files, 1):
                        rows = utils.parse_srt(f.read().decode("utf-8", "ignore"))
                        out = asyncio.run(
                            trans_gemini.translate_async(
                                rows, GEMINI_API_KEY, model_name,
                                status, f.name, polish, idx, len(files
                            ))
                        )
                        z.writestr(f.name, utils.build_srt([[r[0], r[1], t] for r, t in zip(rows, out)]))

                st.success("✅ Gemini 번역 완료")
                st.download_button("⬇ 결과 ZIP 다운로드", zip_buf.getvalue(), "gemini.zip")

    # ======================
    # [API] DeepL
    # ======================
    with tabs[2]:
        st.subheader("🌐 DeepL API 번역")

        files = st.file_uploader(
            "SRT 업로드 (DeepL)",
            type=["srt"],
            accept_multiple_files=True,
            key="deepl_uploader"
        )

        if st.button("▶ DeepL 번역 시작") and files:
            if not DEEPL_API_KEY:
                st.error("DeepL API Key가 없습니다.")
            else:
                status = st.empty()
                zip_buf = io.BytesIO()

                with zipfile.ZipFile(zip_buf, "w") as z:
                    for idx, f in enumerate(files, 1):
                        rows = utils.parse_srt(f.read().decode("utf-8", "ignore"))
                        out = asyncio.run(
                            trans_deepl.translate_async(
                                rows, DEEPL_API_KEY, status, f.name, idx, len(files)
                            )
                        )
                        z.writestr(f.name, utils.build_srt([[r[0], r[1], t] for r, t in zip(rows, out)]))

                st.success("✅ DeepL 번역 완료")
                st.download_button("⬇ 결과 ZIP 다운로드", zip_buf.getvalue(), "deepl.zip")

    # ======================
    # [API] Claude
    # ======================
    with tabs[3]:
        st.subheader("🧠 Claude 번역")

        polish = st.toggle(
            "🛠 한국어 자막 윤문 모드",
            value=False,
            key="claude_polish"
        )

        files = st.file_uploader(
            "SRT 업로드 (Claude)",
            type=["srt"],
            accept_multiple_files=True,
            key="claude_uploader"
        )

        if st.button("▶ Claude 번역 시작") and files:
            if not CLAUDE_API_KEY:
                st.error("Claude API Key가 없습니다.")
            else:
                status = st.empty()
                zip_buf = io.BytesIO()

                with zipfile.ZipFile(zip_buf, "w") as z:
                    for idx, f in enumerate(files, 1):
                        rows = utils.parse_srt(f.read().decode("utf-8", "ignore"))
                        out = asyncio.run(
                            trans_claude.translate_async(
                                rows, CLAUDE_API_KEY, status,
                                f.name, polish, idx, len(files)
                            )
                        )
                        z.writestr(f.name, utils.build_srt([[r[0], r[1], t] for r, t in zip(rows, out)]))

                st.success("✅ Claude 번역 완료")
                st.download_button("⬇ 결과 ZIP 다운로드", zip_buf.getvalue(), "claude.zip")
