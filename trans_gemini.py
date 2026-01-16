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
                out_list[idx] = text
                return idx
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
            # 윤문 모드: 한글만 다듬기
            if is_korean(cleaned):
                targets.append(i)
        else:
            # 번역 모드: 한글이 아니면 전부 번역 (언어 자동 감지)
            if not is_korean(cleaned):
                targets.append(i)

    # 디버그용 (문제 생기면 바로 확인 가능)
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
                    instruction = (
                        "The current line is already Korean.\n"
                        "Polish and refine it naturally for subtitles.\n"
                        "Do NOT translate or change the meaning."
                    )
                else:
                    instruction = (
                        "Detect the language automatically.\n"
                        "If the current line is NOT Korean, you MUST translate it into natural Korean.\n"
                        "If it IS Korean, keep it unchanged."
                    )

                prompt = f"""
You are a professional subtitle translator.

[ABSOLUTE RULES]
- Output Korean only.
- Subtitle style: short, natural spoken Korean.
- Do NOT add explanations.
- Do NOT keep foreign language if translation is required.

[INSTRUCTION]
{instruction}

[PREVIOUS SUBTITLES]
{prev_ctx}

[CURRENT LINE]
{texts[i]}

[NEXT SUBTITLES]
{next_ctx}

Korean subtitle only:
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
