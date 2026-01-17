import re
import os
import gc
import torch
import logging
import warnings
import langid
from datetime import datetime

# ======================
# CONFIG & LOGGING
# ======================
def setup_logging():
    warnings.filterwarnings("ignore")
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["TORCH_LOGS"] = "-all"
    
    # Streamlit 및 기타 라이브러리 로그 강제 차단
    loggers_to_silence = [
        "streamlit", "streamlit.runtime.scriptrunner_utils.script_run_context",
        "streamlit.runtime.state.session_state_proxy", "git", "filelock", "fsspec", "urllib3",
    ]
    for logger_name in loggers_to_silence:
        l = logging.getLogger(logger_name)
        l.setLevel(logging.ERROR)
        l.propagate = False

def get_now():
    """현재 시간을 datetime 객체로 반환"""
    return datetime.now()

def format_duration(start_time, end_time):
    """소요 시간을 분/초 형식으로 변환"""
    duration = end_time - start_time
    seconds = int(duration.total_seconds())
    mins, secs = divmod(seconds, 60)
    return f"{mins}분 {secs}초"

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