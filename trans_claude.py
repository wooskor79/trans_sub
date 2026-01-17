import aiohttp
import asyncio
import re
import utils

CLAUDE_CONTEXT = 4
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"

def is_korean(text):
    return bool(re.search(r"[가-힣]", text))

async def fetch_claude(session, api_key, prompt, idx, out_list):
    url = "https://api.anthropic.com/v1/messages"
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 512,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        async with session.post(url, headers=headers, json=payload, timeout=60) as r:
            if r.status == 200:
                data = await r.json()
                text = data["content"][0]["text"].strip()
                out_list[idx] = text
                return idx
    except: return None

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
                    instruction = "문맥에 맞게 자연스러운 한국어 자막으로 윤문하세요."
                else:
                    instruction = "문맥을 반영하여 자연스러운 한국어로 번역하세요."

                prompt = f"""당신은 전문 자막 번역가입니다.
[규칙]
- 한국어만 출력하세요.
- 앞뒤 4줄의 문맥을 참고하세요.

[문맥]
{prev_ctx}
>>> 번역대상: {texts[i]}
{next_ctx}

[지시] {instruction}
한국어 결과만:""".strip()
                tasks.append(fetch_claude(session, api_key, prompt, i, out))

            await asyncio.gather(*tasks)
            if chunk:
                last = chunk[-1]
                status.markdown(f"""
                <div style="background:#1e1e1e;padding:20px;border-radius:10px;border:1px solid #ffb703;">
                <h3 style="color:#ffb703;">✨ Claude 번역 (문맥 4줄)</h3>
                <p><b>파일:</b> {file_info} ({file_idx}/{total_files}) | <b>진행:</b> {min(j+len(chunk), len(targets))}/{len(targets)}</p>
                <hr>
                <p style="color:#888;"><b>원문:</b> {utils.clean_text(texts[last])}</p>
                <p style="color:#ffb703;"><b>결과:</b> {utils.clean_text(out[last])}</p>
                </div>
                """, unsafe_allow_html=True)
    return out