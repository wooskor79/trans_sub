import aiohttp
import asyncio
import re
import utils

CLAUDE_CONTEXT = 4
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"


def is_korean(text):
    return bool(re.search(r"[가-힣]", text))


def is_english(text):
    return bool(re.search(r"[A-Za-z]", text)) and not is_korean(text)


async def fetch_claude(session, api_key, prompt, idx, out_list):
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 512,
        "temperature": 0.2,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    try:
        async with session.post(url, headers=headers, json=payload, timeout=60) as r:
            if r.status == 200:
                data = await r.json()
                text = data["content"][0]["text"].strip()
                out_list[idx] = text
                return idx
    except Exception:
        return None


async def translate_async(
    rows,
    api_key,
    status,
    file_info,
    polish_ko,
    file_idx,
    total_files
):
    texts = [r[2] for r in rows]
    out = texts[:]
    targets = []

    # ---- 대상 필터링 ----
    for i, t in enumerate(texts):
        cleaned = utils.clean_text(t)
        if not cleaned:
            continue

        prev_ctx = texts[max(0, i - CLAUDE_CONTEXT):i]
        next_ctx = texts[i + 1:i + 1 + CLAUDE_CONTEXT]
        ctx_block = prev_ctx + next_ctx

        korean_ratio = (
            sum(1 for x in ctx_block if is_korean(x)) /
            max(len(ctx_block), 1)
        )

        if polish_ko:
            if is_korean(cleaned):
                targets.append(i)
        else:
            if is_english(cleaned) and korean_ratio >= 0.6:
                targets.append(i)

    connector = aiohttp.TCPConnector(limit_per_host=5)

    async with aiohttp.ClientSession(connector=connector) as session:
        for j in range(0, len(targets), 10):
            chunk = targets[j:j + 10]
            tasks = []

            for i in chunk:
                prev_ctx = "\n".join(texts[max(0, i - CLAUDE_CONTEXT):i])
                next_ctx = "\n".join(texts[i + 1:i + 1 + CLAUDE_CONTEXT])

                prompt = f"""
You are a professional subtitle translator.

[STRICT RULES]
- Output Korean only
- Subtitle style (short, natural spoken Korean)
- Do NOT expand sentence length
- Do NOT add explanations
- Keep tone consistent with surrounding lines

[PREVIOUS SUBTITLES]
{prev_ctx}

[CURRENT LINE]
{texts[i]}

[NEXT SUBTITLES]
{next_ctx}

{"Polish the Korean subtitle naturally." if polish_ko else "Translate into Korean to match the context."}

Korean subtitle only:
""".strip()

                tasks.append(
                    fetch_claude(
                        session,
                        api_key,
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
<div style="background:#1e1e1e;padding:20px;border-radius:10px;border:1px solid #ffb703;">
<h3 style="color:#ffb703;">✨ Claude 번역</h3>
<p><b>파일:</b> {file_info} ({file_idx}/{total_files})</p>
<p><b>진행:</b> {min(j + 10, len(targets))}/{len(targets)}</p>
<hr>
<p style="color:#888;"><b>원문:</b> {utils.clean_text(texts[last])}</p>
<p style="color:#ffb703;"><b>결과:</b> {utils.clean_text(out[last])}</p>
</div>
""",
                    unsafe_allow_html=True
                )

    return out
