# 1. 베이스가 될 파이썬 버전을 선택합니다.
FROM python:3.10-slim

# 2. ★★★ OpenCV에 필요한 시스템 라이브러리 설치 (핵심 수정!) ★★★
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. 서버 내부에서 작업할 폴더를 만듭니다.
WORKDIR /app

# 4. 필요한 부품 목록(requirements.txt)을 먼저 복사해서 설치합니다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 우리 프로젝트 코드 전체(app.py, templates/ 등)를 복사합니다.
COPY . .

# 6. 8080 포트를 사용하도록 고정
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]