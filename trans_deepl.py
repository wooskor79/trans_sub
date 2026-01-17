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

async def fetch_deepl(session, api_key, text, idx, out_list, cache):
    url = "https://api-free.deepl.com/v2/translate"
    cleaned = utils.clean_text(text)
    if cleaned in cache:
        out_list[idx] = cache[cleaned]
        return idx

    for attempt in range(3):
        try:
            async with session.post(url, headers={"Authorization": f"DeepL-Auth-Key {api_key}"}, data={"text": text, "target_lang": "KO"}, timeout=15) as r:
                if r.status == 200:
                    data = await r.json()
                    res = data["translations"][0]["text"]
                    out_list[idx] = res
                    cache[cleaned] = res
                    return idx
                elif r.status == 429:
                    await asyncio.sleep((2 ** attempt) + 1)
        except: await asyncio.sleep(0.5)
    return None

async def translate_async(rows, api_key, status, file_info, file_idx, total_files):
    texts = [r[2] for r in rows]
    out = texts[:]
    targets = [i for i, t in enumerate(texts) if utils.clean_text(t)]
    translation_cache = {}
    
    connector = aiohttp.TCPConnector(limit_per_host=1)
    async with aiohttp.ClientSession(connector=connector) as session:
        for idx, i in enumerate(targets):
            await fetch_deepl(session, api_key, texts[i], i, out, translation_cache)
            status.markdown(f"""
            <div style="background:#1e1e1e;padding:15px;border-radius:12px;border:1px solid #ff9a9e; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <h4 style="margin:0;color:#ff9a9e;">🌐 DeepL Pro Translation</h4>
                <span style="background:#333;padding:4px 8px;border-radius:4px;font-size:0.8em;color:#eee;">{idx+1}/{len(targets)}</span>
                <span style="background:#333;padding:4px 8px;border-radius:4px;font-size:0.8em;color:#eee;">File {file_idx}/{total_files}</span>
            </div>
            <div style="font-size:0.9em;color:#aaa;margin-bottom:5px;">📂 {file_info}</div>
             <div style="background:#2d2d2d;padding:10px;border-radius:8px;margin-bottom:8px;">
                <span style="color:#888;font-size:0.85em;">Original</span><br>
                <span style="color:#eee;">{utils.clean_text(texts[i])}</span>
            </div>
            <div style="background:#263238;padding:10px;border-radius:8px;border-left:4px solid #ff9a9e;">
                <span style="color:#ff9a9e;font-size:0.85em;">Translated</span><br>
                <span style="color:#fff;font-weight:bold;">{utils.clean_text(out[i])}</span>
            </div>
            </div>
            """, unsafe_allow_html=True)
            await asyncio.sleep(0.1)  # 429 방지 안정화 딜레이 약간 증가
    return out