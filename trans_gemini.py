import aiohttp
import asyncio
import re
import utils

GEMINI_CONTEXT = 4

def is_korean(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))

async def fetch_gemini(session, api_key, model_name, prompt, idx, out_list):
    url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.1,  # 창의성보다 정확도와 일관성에 집중
            "topP": 0.8,
            "maxOutputTokens": 512
        }
    }
    try:
        async with session.post(url, json=payload, timeout=90) as r:
            if r.status == 200:
                data = await r.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                # 마크다운 서식 및 따옴표 제거
                text = re.sub(r"```[a-z]*\n?|\n?```", "", text).strip()
                text = re.sub(r'^["\']|["\']$', '', text)
                out_list[idx] = text
    except: pass

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

    connector = aiohttp.TCPConnector(limit_per_host=6)
    async with aiohttp.ClientSession(connector=connector) as session:
        for j in range(0, len(targets), 6):
            chunk = targets[j:j + 6]
            tasks = []
            for i in chunk:
                prev_ctx = "\n".join(texts[max(0, i - GEMINI_CONTEXT):i])
                next_ctx = "\n".join(texts[i + 1:i + 1 + GEMINI_CONTEXT])
                
                if polish_ko:
                    instruction = "이 문장은 이미 한국어입니다. 문맥에 맞춰 아주 매끄럽고 자연스러운 자막 말투로 교정하세요. 설명 없이 결과만 출력하세요."
                else:
                    instruction = (
                        "이것은 번역 작업입니다. 현재 문장을 반드시 한국어로 번역하세요.\n"
                        "절대로 원문(일본어, 영어 등)을 그대로 출력하지 마세요.\n"
                        "괄호 안의 이름이나 호칭(예: (成増), (成増))도 무조건 한국어 발음(예: (나리마스))으로 번역해야 합니다.\n"
                        "한자나 가나가 결과물에 남아있다면 당신은 임무에 실패한 것입니다."
                    )

                prompt = f"""당신은 최고의 영상 자막 번역 전문가입니다.

[강제 지침]
- 오직 한국어(한글)로만 답변하세요.
- 원문을 복사해서 내뱉는 행위(앵무새)를 절대 금지합니다.
- 자막 스타일: 짧고, 자연스러운 구어체.
- 앞뒤 문맥 4줄을 참고하여 대화의 흐름을 유지하세요.

[문맥 정보]
이전 상황:
{prev_ctx if prev_ctx else "(시작)"}

현재 번역할 문장:
{texts[i]}

이후 상황:
{next_ctx if next_ctx else "(끝)"}

[작업 명령]
{instruction}

결과물(한국어만):""".strip()
                tasks.append(fetch_gemini(session, api_key, model_name, prompt, i, out))

            await asyncio.gather(*tasks)
            last = chunk[-1]
            status.markdown(f"""
            <div style="background:#1e1e1e;padding:18px;border-radius:10px;border:1px solid #4facfe;">
            <h4 style="color:#4facfe;">✨ Gemini Flash 고성능 번역 중</h4>
            <p><b>파일:</b> {file_info} ({file_idx}/{total_files}) | <b>진행:</b> {min(j+len(chunk), len(targets))}/{len(targets)}</p>
            <hr>
            <p style="color:#888;"><b>원문:</b> {utils.clean_text(texts[last])}</p>
            <p style="color:#4facfe;"><b>결과:</b> {utils.clean_text(out[last])}</p>
            </div>
            """, unsafe_allow_html=True)
    return out