import aiohttp
import asyncio
import re
import utils

GEMINI_CONTEXT = 3

def is_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))

async def fetch_gemini(session, api_key, model_name, prompt, idx, out_list):
    url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,  # 정밀도 최우선
            "topP": 0.9,
            "maxOutputTokens": 1024
        }
    }
    for attempt in range(3):
        try:
            async with session.post(url, json=payload, timeout=90) as r:
                if r.status == 200:
                    data = await r.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    # 불필요한 마크다운 및 따옴표 제거
                    text = re.sub(r"```[a-z]*\n?|\n?```", "", text).strip()
                    text = re.sub(r'^["\']|["\']$', '', text)
                    out_list[idx] = text
                    return
                elif r.status == 429:
                    # Rate Limit 걸리면 지수 백오프: 2초, 4초, 8초 대기
                    await asyncio.sleep(2 ** (attempt + 1))
        except:
            await asyncio.sleep(1)
            pass

async def translate_async(rows, api_key, model_name, status, file_info, polish_ko, file_idx, total_files):
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

    if not targets: return out

    # Gemini 2.0 Flash는 빠르므로 동시성 10까지 허용 (Rate Limit 주의)
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
                        "이 문장은 이미 한국어입니다. 오타나 어색한 표현을 수정하여 완벽한 자막체로 다듬으십시오.\n"
                        "의미를 왜곡하지 말고, 자연스러운 구어체로 만드세요."
                    )
                else:
                    instruction = (
                        "이것은 영상 자막 번역 작업입니다. 주어진 문장을 '완벽한 한국어'로 번역하세요.\n"
                        "- 직역투를 피하고, 상황에 맞는 자연스러운 구어체/대화체를 사용하세요.\n"
                        "- 인물 호칭, 고유명사는 한국어 표준 발음 표기를 따르십시오.\n"
                        "- 원문(영어/일본어 등)을 절대 포함하지 마십시오."
                    )

                prompt = f"""[Role]
You are Korea's top-tier subtitle translator. Translate the following text into natural, high-quality Korean subtitles.

[Context Info]
User settings: Context window ±{GEMINI_CONTEXT} lines.
Use the context below to infer tone, gender, and situation.

Previous:
{prev_ctx if prev_ctx else "(Start)"}

Target Sentence:
{texts[i]}

Next:
{next_ctx if next_ctx else "(End)"}

[Command]
{instruction}

[Output]
Provide ONLY the Korean translation."""

                tasks.append(fetch_gemini(session, api_key, model_name, prompt, i, out))

            await asyncio.gather(*tasks)
            last = chunk[-1]
            status.markdown(f"""
            <div style="background:#1e1e1e;padding:15px;border-radius:12px;border:1px solid #4facfe; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <h4 style="margin:0;color:#4facfe;">✨ Gemini Flash Ultra</h4>
                <span style="background:#333;padding:4px 8px;border-radius:4px;font-size:0.8em;color:#eee;">{min(j+len(chunk), len(targets))}/{len(targets)}</span>
                <span style="background:#333;padding:4px 8px;border-radius:4px;font-size:0.8em;color:#eee;">File {file_idx}/{total_files}</span>
            </div>
            <div style="font-size:0.9em;color:#aaa;margin-bottom:5px;">📂 {file_info}</div>
            <div style="background:#2d2d2d;padding:10px;border-radius:8px;margin-bottom:8px;">
                <span style="color:#888;font-size:0.85em;">Original</span><br>
                <span style="color:#eee;">{utils.clean_text(texts[last])}</span>
            </div>
            <div style="background:#263238;padding:10px;border-radius:8px;border-left:4px solid #4facfe;">
                <span style="color:#4facfe;font-size:0.85em;">Translated</span><br>
                <span style="color:#fff;font-weight:bold;">{utils.clean_text(out[last])}</span>
            </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Rate Limit 방지를 위한 안전 딜레이 (0.5초)
            await asyncio.sleep(0.5)
    return out