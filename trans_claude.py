import aiohttp
import asyncio
import re
import utils

CLAUDE_CONTEXT = 4
CLAUDE_MODEL = "claude-sonnet-4-20250514"

def is_korean(text):
    return bool(re.search(r"[가-힣]", text))

async def fetch_claude_retry(session, api_key, payload, idx, out_list):
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    
    for attempt in range(3):
        try:
            async with session.post(url, headers=headers, json=payload, timeout=60) as r:
                if r.status == 200:
                    data = await r.json()
                    text = data["content"][0]["text"].strip()
                    # 앵무새 방지: 혹시라도 원문이 그대로 나오면(간단한 체크) 재시도할 수도 있음. 
                    # 여기선 일단 결과 저장.
                    out_list[idx] = text
                    return idx
                elif r.status == 429:
                    await asyncio.sleep(2 ** (attempt + 1))
        except: 
            await asyncio.sleep(1)
    return None

async def translate_async(rows, api_key, status, file_info, polish_ko, file_idx, total_files):
    texts = [r[2] for r in rows]
    out = texts[:]
    targets = []

    for i, t in enumerate(texts):
        cleaned = utils.clean_text(t)
        if not cleaned: continue
        if polish_ko:
            if is_korean(cleaned): targets.append(i)
        else:
            if not is_korean(cleaned): targets.append(i)

    connector = aiohttp.TCPConnector(limit_per_host=5)
    async with aiohttp.ClientSession(connector=connector) as session:
        for j in range(0, len(targets), 5):
            chunk = targets[j:j + 5]
            tasks = []
            for i in chunk:
                prev_ctx = "\n".join(texts[max(0, i - CLAUDE_CONTEXT):i])
                next_ctx = "\n".join(texts[i + 1:i + 1 + CLAUDE_CONTEXT])
                
                if polish_ko:
                    # 교정 모드: 이미 한국어이므로 자연스럽게 다듬기
                    user_prompt = f"""[System]
You are a professional Korean subtitle editor. The following text is already in Korean (or broken Korean).
Polishing it into natural, high-quality Korean movie subtitles.
Maintain the original meaning but improve fluency, tone, and spacing.

[Input Text]
{texts[i]}

[Output]
Provide ONLY the polished Korean text. Do not add explanations."""
                    # Pre-fill (다듬은 결과:)
                    prefill = "다듬은 결과:"
                
                else:
                    # 번역 모드: 앵무새 방지 강화
                    user_prompt = f"""[Role]
You are a professional subtitle translator. Translate the target text into 'Korean'.

[Context Info]
{prev_ctx}

[Target Text to Translate]
{texts[i]}

[Context Info]
{next_ctx}

[Constraints]
1. Translate exactly into Korean.
2. DO NOT repeat the original text.
3. DO NOT add notes or explanations.
4. Use natural spoken Korean (subtitles)."""
                    # Pre-fill (한국어 자막:) -> 강제로 한국어를 뱉게 유도
                    prefill = "한국어 자막:"

                payload = {
                    "model": CLAUDE_MODEL,
                    "max_tokens": 1024,
                    "messages": [
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": prefill} # Prefill Added
                    ],
                    "temperature": 0.1
                }
                
                # Retry Logic
                tasks.append(fetch_claude_retry(session, api_key, payload, i, out))

            await asyncio.gather(*tasks)
            if chunk:
                last = chunk[-1]
                status.markdown(f"""
                <div style="background:#1e1e1e;padding:15px;border-radius:12px;border:1px solid #ffb703; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                    <h4 style="margin:0;color:#ffb703;">✨ Claude Sonnet 3.5</h4>
                    <span style="background:#333;padding:4px 8px;border-radius:4px;font-size:0.8em;color:#eee;">{min(j+len(chunk), len(targets))}/{len(targets)}</span>
                    <span style="background:#333;padding:4px 8px;border-radius:4px;font-size:0.8em;color:#eee;">File {file_idx}/{total_files}</span>
                </div>
                <div style="font-size:0.9em;color:#aaa;margin-bottom:5px;">📂 {file_info}</div>
                <div style="background:#2d2d2d;padding:10px;border-radius:8px;margin-bottom:8px;">
                    <span style="color:#888;font-size:0.85em;">Original</span><br>
                    <span style="color:#eee;">{utils.clean_text(texts[last])}</span>
                </div>
                <div style="background:#263238;padding:10px;border-radius:8px;border-left:4px solid #ffb703;">
                    <span style="color:#ffb703;font-size:0.85em;">Translated</span><br>
                    <span style="color:#fff;font-weight:bold;">{utils.clean_text(out[last])}</span>
                </div>
                </div>
                """, unsafe_allow_html=True)
    return out