import re
import os
import gc
import torch
import logging
import warnings
import langid

# ======================
# CONFIG & LOGGING
# ======================
def setup_logging():
    # 1. Python 경고 차단
    warnings.filterwarnings("ignore")
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TORCH_LOGS"] = "-all"

    # 2. 로깅 레벨 강제 조정
    loggers_to_silence = [
        "streamlit", "streamlit.runtime.scriptrunner_utils.script_run_context",
        "streamlit.runtime.state.session_state_proxy", "git", "filelock", "fsspec", "urllib3",
    ]
    for logger_name in loggers_to_silence:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
        logger.propagate = False

# ======================
# TEXT & SRT UTILS
# ======================
LANG_MAP = {
    "en": "eng_Latn", "ja": "jpn_Jpan", "de": "deu_Latn", "fr": "fra_Latn",
    "es": "spa_Latn", "pt": "por_Latn", "zh": "zho_Hans", "ru": "rus_Cyrl",
    "it": "ita_Latn", "nl": "nld_Latn", "pl": "pol_Latn", "tr": "tur_Latn",
    "vi": "vie_Latn", "id": "ind_Latn", "th": "tha_Thai",
}

def detect_language(text):
    lang, _ = langid.classify(text)
    return LANG_MAP.get(lang, "eng_Latn"), lang

def clean_text(t):
    if not t: return ""
    return re.sub(r"[\x00-\x1f]", "", t).strip()

def parse_srt(txt):
    blocks = re.split(r"\n\s*\n", txt.strip())
    rows = []
    for b in blocks:
        lines = b.splitlines()
        if len(lines) >= 2:
            idx = lines[0] if lines[0].isdigit() else ""
            tc = lines[1] if "-->" in lines[1] else ""
            text = " ".join(lines[2:]) if tc else " ".join(lines[1:])
            rows.append([idx, tc, text])
    return rows

def build_srt(rows):
    out = []
    for i, t, x in rows:
        if i: out.append(str(i))
        if t: out.append(t)
        out.append(x)
        out.append("")
    return "\n".join(out)

# ======================
# SYSTEM UTILS
# ======================
def clear_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_vram_status():
    if not torch.cuda.is_available():
        return 0, 0
    used = torch.cuda.memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory
    return used / 1024**3, total / 1024**3