import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 256

@st.cache_resource
def load_model(model_id):
    # 품질 최우선: 압축 없이 FP16 로드
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",  # 강제 CUDA 할당
        low_cpu_mem_usage=True
    )
    mdl.eval()
    return tok, mdl

def translate(rows, tok, mdl, status, file_info, file_idx, total_files):
    texts = [r[2] for r in rows]
    out = texts[:]
    todo_map = {}
    translation_cache = {}

    for i, t in enumerate(texts):
        cleaned = utils.clean_text(t)
        if not cleaned: continue
        if cleaned in translation_cache:
            out[i] = translation_cache[cleaned]
        else:
            todo_map.setdefault(cleaned, []).append(i)

    unique_texts = list(todo_map.keys())
    unique_texts = list(todo_map.keys())
    batch_size = 64  # RTX 5080: 고성능 배칭
    
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

        u_vram, t_vram = utils.get_vram_status()
        status.markdown(f"""
        <div style="background:#1e1e1e;padding:15px;border-radius:12px;border:1px solid #7df9ff; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
            <h4 style="margin:0;color:#7df9ff;">🚀 NLLB 3.3B (RTX 5080 Extreme)</h4>
            <span style="background:#333;padding:4px 8px;border-radius:4px;font-size:0.8em;color:#eee;">{min(p+len(batch_src), len(unique_texts))}/{len(unique_texts)}</span>
            <span style="background:#333;padding:4px 8px;border-radius:4px;font-size:0.8em;color:#eee;">File {file_idx}/{total_files}</span>
            <span style="background:#333;padding:4px 8px;border-radius:4px;font-size:0.8em;color:#888;">VRAM: {u_vram:.1f}GB</span>
        </div>
        <div style="font-size:0.9em;color:#aaa;margin-bottom:5px;">📂 {file_info}</div>
        <div style="background:#2d2d2d;padding:10px;border-radius:8px;margin-bottom:8px;">
            <span style="color:#888;font-size:0.85em;">Original</span><br>
            <span style="color:#eee;">{utils.clean_text(batch_src[-1])}</span>
        </div>
        <div style="background:#263238;padding:10px;border-radius:8px;border-left:4px solid #7df9ff;">
            <span style="color:#7df9ff;font-size:0.85em;">Translated</span><br>
            <span style="color:#fff;font-weight:bold;">{utils.clean_text(results[-1])}</span>
        </div>
        </div>
        """, unsafe_allow_html=True)
        
    return out