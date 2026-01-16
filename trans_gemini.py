import aiohttp
import asyncio
import re
import json
import utils

GEMINI_CONTEXT = 4

def is_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))

async def fetch_gemini(session, api_key, model_name, prompt, idx, out_list):
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={api_key}"
    )

    # 1. 안전 필터 완전 해제 (BLOCK_NONE)
    # 2. JSON 모드 강제 (application/json)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
        "generationConfig": {
            "response_mime_type": "application/json"
        }
    }

    try:
        async with session.post(url, json=payload, timeout=60) as r:
            if r.status == 200:
                data = await r.json()
                # JSON 파싱 시도
                try:
                    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                    parsed = json.loads(raw_text)
                    # "translation" 키가 있으면 그것을, 없으면 전체 텍스트 사용
                    if isinstance(parsed, dict) and "translation" in parsed:
                        text = parsed["translation"]
                    else:
                        text = str(parsed)
                    
                    out_list[idx] = text.strip()
                    return idx
                except (KeyError, json.JSONDecodeError):
                    # JSON 파싱 실패 시 원문 유지하지 않고 빈 문자열이라도 넣어서 확인 가능하게 함
                    return None
            else:
                # 에러(필터 걸림 등) 발생 시
                # print(await r.text()) # 디버깅 필요시 주석 해제
                return None
    except Exception:
        return None

async def translate_async(
    rows,
    api_key,
    model_name,
    status,
    file_info,
    polish_ko,
    file_idx,
    total_files
):
    texts = [r[2] for r in rows]
    out = texts[:]
    targets = []

    for i, t in enumerate(texts):
        cleaned = utils.clean_text(t)
        if not cleaned:
            continue
        if polish_ko:
            if is_korean(cleaned): targets.append(i)
        else:
            if not is_korean(cleaned): targets.append(i)

    status.write(f"Gemini targets: {len(targets)}")

    if not targets:
        return out

    connector = aiohttp.TCPConnector(limit_per_host=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        for j in range(0, len(targets), 10):
            chunk = targets[j:j + 10]
            tasks = []

            for i in chunk:
                prev_ctx = "\n".join(texts[max(0, i - GEMINI_CONTEXT):i])
                next_ctx = "\n".join(texts[i + 1:i + 1 + GEMINI_CONTEXT])

                # JSON 출력을 강제하는 강력한 프롬프트
                prompt = f"""
You are a professional subtitle translator.

Task: Translate the [TARGET] line into natural spoken Korean.
Format: Return a JSON object with a single key "translation".

[RULES]
1. value of "translation" MUST be Korean.
2. Translate names in parentheses phonetically (e.g., (Miyabi) -> (미야비)).
3. Do NOT output the original text.

[EXAMPLE]
Input: (ミヤビ) おはよう
Output: {{"translation": "(미야비) 안녕"}}

[CONTEXT BEFORE]
{prev_ctx}

[TARGET]
{texts[i]}

[CONTEXT AFTER]
{next_ctx}
""".strip()

                tasks.append(
                    fetch_gemini(
                        session,
                        api_key,
                        model_name,
                        prompt,
                        i,
                        out
                    )
                )

            await asyncio.gather(*tasks)

            if chunk:
                last = chunk[-1]
                # UI 업데이트
                status.markdown(
                    f"""
<div style="background:#1e1e1e;padding:20px;border-radius:10px;border:1px solid #4facfe;">
<h3 style="color:#4facfe;">✨ Gemini 최상급 번역 (JSON Mode)</h3>
<p><b>파일:</b> {file_info} ({file_idx}/{total_files})</p>
<p><b>진행:</b> {min(j + 10, len(targets))}/{len(targets)}</p>
<hr>
<p style="color:#888;"><b>원문:</b> {utils.clean_text(texts[last])}</p>
<p style="color:#4facfe;"><b>결과:</b> {utils.clean_text(out[last])}</p>
</div>
""",
                    unsafe_allow_html=True
                )

    return out