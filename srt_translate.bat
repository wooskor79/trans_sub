@echo off
chcp 65001 > nul

:: 작업 폴더로 이동
cd /d "F:\우성 개인자료\sub_tran"

:: 실행 파일 변경 (index.py -> main.py)
python -m streamlit run main.py

pause