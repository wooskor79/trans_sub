import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 256

@st.cache_resource
def load_model(model_id):
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    mdl.eval()
    return tok, mdl

def translate(rows, tok, mdl, status, file_info, file_idx, total_files):
    texts = [r[2] for r in rows]
    out = texts[:]
    todo_map = {}
    translation_cache = {} # 로컬 캐시

    # 중복 제거 및 작업 목록 생성
    for i, t in enumerate(texts):
        cleaned = utils.clean_text(t)
        if not cleaned: continue
        if cleaned in translation_cache:
            out[i] = translation_cache[cleaned]
        else:
            todo_map.setdefault(cleaned, []).append(i)

    unique_texts = list(todo_map.keys())
    batch_size = 16 
    
    for p in range(0, len(unique_texts), batch_size):
        batch_src = unique_texts[p : p + batch_size]
        src_lang, _ = utils.detect_language(batch_src[0])
        tok.src_lang = src_lang

        with torch.no_grad():
            inputs = tok(batch_src, return_tensors="pt", padding=True).to(DEVICE)
            gen = mdl.generate(**inputs, forced_bos_token_id=tok.convert_tokens_to_ids("kor_Hang"), max_new_tokens=MAX_NEW_TOKENS)
            results = tok.batch_decode(gen, skip_special_tokens=True)

        for src, res in zip(batch_src, results):
            translation_cache[src] = res
            for idx in todo_map[src]:
                out[idx] = res

        # UI 업데이트
        u_vram, t_vram = utils.get_vram_status()
        status.markdown(f"""
        <div style="background:#1e1e1e;padding:20px;border-radius:10px;border:1px solid #00ffcc;">
        <h3 style="color:#00ffcc;">📊 NLLB 번역 중 (Batch 가속)</h3>
        <p><b>파일:</b> {file_info} ({file_idx}/{total_files}) | <b>진행:</b> {p+len(batch_src)}/{len(unique_texts)}</p>
        <p><b>VRAM:</b> {u_vram:.1f}/{t_vram:.1f}GB</p>
        <hr>
        <p style="color:#888;"><b>원문:</b> {utils.clean_text(batch_src[-1])}</p>
        <p style="color:#00ffcc;"><b>번역:</b> {utils.clean_text(results[-1])}</p>
        </div>
        """, unsafe_allow_html=True)
        
    return out