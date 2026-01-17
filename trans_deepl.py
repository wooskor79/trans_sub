import aiohttp
import asyncio
import requests
import utils

DEEPL_FREE_LIMIT = 500000

def get_usage(api_key):
    if not api_key: return None, None
    try:
        r = requests.get("https://api-free.deepl.com/v2/usage", headers={"Authorization": f"DeepL-Auth-Key {api_key}"}, timeout=5)
        if r.status_code == 200:
            data = r.json()
            count = data.get("character_count", 0)
            limit = data.get("character_limit", DEEPL_FREE_LIMIT)
            return count, limit
    except: pass
    return None, None

async def fetch_deepl_batch(session, api_key, text_chunk, start_idx, out_list, cache):
    url = "https://api-free.deepl.com/v2/translate"
    # 이미 번역된 문장이 캐시에 있다면? (배치 단위라 좀 복잡하지만, 여기선 단순화)
    # DeepL은 줄바꿈을 기준으로 문장을 인식하므로 그대로 join해서 보냄.
    
    joined_text = "\n".join(text_chunk)
    
    for attempt in range(3):
        try:
            async with session.post(
                url, 
                headers={"Authorization": f"DeepL-Auth-Key {api_key}"}, 
                data={"text": text_chunk, "target_lang": "KO"}, # DeepL은 리스트로 보내면 리스트로 줌 (공식 지원)
                timeout=30
            ) as r:
                if r.status == 200:
                    data = await r.json()
                    translations = data["translations"]
                    
                    for i, t in enumerate(translations):
                        out_list[start_idx + i] = t["text"]
                    return len(translations)
                elif r.status == 429:
                    await asyncio.sleep(2 ** (attempt + 1))
        except: 
            await asyncio.sleep(1)
            pass
    return 0

async def translate_async(rows, api_key, status, file_info, file_idx, total_files):
    texts = [r[2] for r in rows]
    out = texts[:]
    
    # 번역할 대상 인덱스 추출
    targets = [i for i, t in enumerate(texts) if utils.clean_text(t)]
    if not targets: return out
    
    connector = aiohttp.TCPConnector(limit_per_host=2) # 배치니까 동시성 낮아도 됨
    async with aiohttp.ClientSession(connector=connector) as session:
        # 20개씩 묶어서 배치 처리
        batch_size = 20
        for i in range(0, len(targets), batch_size):
            chunk_indices = targets[i : i + batch_size]
            chunk_texts = [texts[idx] for idx in chunk_indices]
            
            await fetch_deepl_batch(session, api_key, chunk_texts, 0, [], {}) # Dummy call definition above needs adjustment
            
            # 실제 구현: fetch_deepl_batch 내장 로직을 여기서 풀어씀 (리스트 지원 활용)
            url = "https://api-free.deepl.com/v2/translate"
            success = False
            
            for attempt in range(3):
                try:
                    # DeepL API는 'text' 파라미터를 여러 개 보낼 수 있음 (Multi-param)
                    # aiohttp에서 data에 리스트를 주면 같은 키로 여러 개 날라감
                    # text=A&text=B&text=C ... <-- 이게 정석
                    payload = {"target_lang": "KO"}
                    current_payload = [("text", t) for t in chunk_texts]
                    current_payload.append(("target_lang", "KO"))
                    
                    async with session.post(
                        url, 
                        headers={"Authorization": f"DeepL-Auth-Key {api_key}"}, 
                        data=current_payload,
                        timeout=30
                    ) as r:
                        if r.status == 200:
                            data = await r.json()
                            res_list = data["translations"]
                            for k, item in enumerate(res_list):
                                real_idx = chunk_indices[k]
                                out[real_idx] = item["text"]
                            success = True
                            break
                        elif r.status == 429:
                            await asyncio.sleep(2 ** (attempt + 1))
                except: 
                    await asyncio.sleep(1)
            
            if not success:
                pass # 실패 시 원문 유지

            # Progress UI
            last_idx = chunk_indices[-1]
            status.markdown(f"""
            <div style="background:#1e1e1e;padding:15px;border-radius:12px;border:1px solid #ff9a9e; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <h4 style="margin:0;color:#ff9a9e;">🌐 DeepL Pro (Context Batch)</h4>
                <span style="background:#333;padding:4px 8px;border-radius:4px;font-size:0.8em;color:#eee;">{min(i+len(chunk_indices), len(targets))}/{len(targets)}</span>
                <span style="background:#333;padding:4px 8px;border-radius:4px;font-size:0.8em;color:#eee;">File {file_idx}/{total_files}</span>
            </div>
            <div style="font-size:0.9em;color:#aaa;margin-bottom:5px;">📂 {file_info}</div>
             <div style="background:#2d2d2d;padding:10px;border-radius:8px;margin-bottom:8px;">
                <span style="color:#888;font-size:0.85em;">Original</span><br>
                <span style="color:#eee;">{utils.clean_text(chunk_texts[-1])}</span>
            </div>
            <div style="background:#263238;padding:10px;border-radius:8px;border-left:4px solid #ff9a9e;">
                <span style="color:#ff9a9e;font-size:0.85em;">Translated</span><br>
                <span style="color:#fff;font-weight:bold;">{utils.clean_text(out[last_idx])}</span>
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            await asyncio.sleep(0.5) # 안전 딜레이
            
    return out