공통으로 필요한 파이선 3.10 버전
https://www.python.org/downloads/release/python-31019/?utm_source=chatgpt.com
설치할때 path 등록 안하면 나중에 환경변수에서 등록해줘야함
환경변수 등록은
C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\
C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python310\Scripts\
이 경로 등록인데 
C:\Users\woo\AppData\Local\Programs\Python\Python310\
C:\Users\woo\AppData\Local\Programs\Python\Python310\Scripts\

이런식으로  c:에서 사용자 폴더 가보면 로그인아이디 보일꺼임 그거 넣음됨



50시리즈 준비
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install streamlit transformers langid python-dotenv requests

40시리즈 준비
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install streamlit transformers langid python-dotenv requests

30시리즈준비
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install streamlit transformers langid python-dotenv requests

