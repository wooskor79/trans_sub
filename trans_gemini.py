import aiohttp
import asyncio
import re
import utils

GEMINI_CONTEXT = 4


def is_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))


async def fetch_gemini(session, api_key, model_name, prompt, idx, out_list):
    url = (
        f"https://generativelanguage.googleapis.com/v1/models/"
        f"{model_name}:generateContent?key={api_key}"
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        async with session.post(url, json=payload, timeout=60) as r:
            if r.status == 200:
                data = await r.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                # 혹시라도 모델이 또 원문을 뱉을 경우를 대비한 2차 방어선 (선택 사항)
                out_list[idx] = text
                return idx
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

    # ======================
    # 대상 줄 선택
    # ======================
    for i, t in enumerate(texts):
        cleaned = utils.clean_text(t)
        if not cleaned:
            continue

        if polish_ko:
            if is_korean(cleaned):
                targets.append(i)
        else:
            if not is_korean(cleaned):
                targets.append(i)

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

                if polish_ko:
                    prompt = f"""
You are a professional subtitle editor.
Refine the [TARGET] Korean line to be more natural.

[CONTEXT BEFORE]
{prev_ctx}

[TARGET]
{texts[i]}

[CONTEXT AFTER]
{next_ctx}

Refined Korean:
""".strip()
                else:
                    # [수정됨] 강력한 예시(Few-Shot) 기반 프롬프트
                    prompt = f"""
You are a professional subtitle translator.
Translate the [TARGET] line into natural spoken Korean.

[RULES]
1. Output ONLY the Korean translation.
2. NEVER output the original text.
3. Translate names in parentheses phonetically (e.g., (Miyabi) -> (미야비)).

[EXAMPLE]
Input: (ミヤビ) おはよう
Output: (미야비) 안녕
Input: 毎日 少しずつ
Output: 매일 조금씩

[CONTEXT BEFORE]
{prev_ctx}

[TARGET]
{texts[i]}

[CONTEXT AFTER]
{next_ctx}

Korean Translation:
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
                status.markdown(
                    f"""
<div style="background:#1e1e1e;padding:20px;border-radius:10px;border:1px solid #4facfe;">
<h3 style="color:#4facfe;">✨ Gemini 최상급 번역</h3>
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