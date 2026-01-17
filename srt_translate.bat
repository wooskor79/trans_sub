@echo off
chcp 65001 > nul
title Subtitle Translator (NLLB 3.3B / Gemini / DeepL / Claude)

:: 작업 폴더로 이동
cd /d "F:\우성 개인자료\sub_tran"

:: Python UTF-8 인코딩 강제 설정 (한글 처리 및 로그 출력 최적화)
set PYTHONUTF8=1

:: 업데이트된 메인 파일(main.py) 실행
:: --browser.gatherUsageStats false: 불필요한 통계 수집을 비활성화하여 리소스 절약
python -m streamlit run main.py --browser.gatherUsageStats false

pause
``` 

1. RTX 5080에서 NLLB 3.3B 모델 로딩 시 VRAM 점유율이나 실행 속도에 문제는 없으신가요?
2. 현재 설정된 작업 폴더(`F:\우성 개인자료\sub_tran`) 외에 다른 경로에서도 실행이 필요한 환경인가요?