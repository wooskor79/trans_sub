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
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.45,
            "topP": 0.9,
            "maxOutputTokens": 256
        }
    }

    try:
        async with session.post(url, json=payload, timeout=90) as r:
            if r.status == 200:
                data = await r.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                out_list[idx] = text
    except Exception:
        pass


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
    # 번역 대상 선택
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

    connector = aiohttp.TCPConnector(limit_per_host=6)
    async with aiohttp.ClientSession(connector=connector) as session:
        for j in range(0, len(targets), 6):
            chunk = targets[j:j + 6]
            tasks = []

            for i in chunk:
                prev_ctx = "\n".join(texts[max(0, i - GEMINI_CONTEXT):i])
                next_ctx = "\n".join(texts[i + 1:i + 1 + GEMINI_CONTEXT])

                if polish_ko:
                    instruction = (
                        "The CURRENT LINE is already Korean.\n"
                        "Polish it to sound natural as a subtitle.\n"
                        "Do NOT translate.\n"
                        "Do NOT change meaning."
                    )
                else:
                    instruction = (
                        "This is a TRANSLATION TASK.\n"
                        "If the CURRENT LINE is NOT Korean, you MUST translate it into Korean.\n"
                        "If any non-Korean characters remain, the task has FAILED.\n"
                        "Never repeat the original text."
                    )

                prompt = f"""
You are a professional subtitle translator.

ABSOLUTE RULES:
- Output Korean only.
- Subtitle style: short, natural spoken Korean.
- Do NOT add explanations.
- Do NOT keep the original language.

TASK:
{instruction}

PREVIOUS SUBTITLES:
{prev_ctx}

CURRENT LINE:
{texts[i]}

NEXT SUBTITLES:
{next_ctx}

Korean subtitle:
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

            last = chunk[-1]
            status.markdown(
                f"""
<div style="background:#1e1e1e;padding:18px;border-radius:10px;border:1px solid #4facfe;">
<h4 style="color:#4facfe;">✨ Gemini 번역 진행</h4>
<p><b>파일:</b> {file_info} ({file_idx}/{total_files})</p>
<p><b>진행:</b> {min(j + 6, len(targets))}/{len(targets)}</p>
<hr>
<p style="color:#888;"><b>원문:</b> {utils.clean_text(texts[last])}</p>
<p style="color:#4facfe;"><b>결과:</b> {utils.clean_text(out[last])}</p>
</div>
""",
                unsafe_allow_html=True
            )

    return out
