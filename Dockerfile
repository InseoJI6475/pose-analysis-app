# 1. 베이스가 될 파이썬 버전을 선택합니다. (Render와 동일하게 3.10)
FROM python:3.10-slim

# 2. 서버 내부에서 작업할 폴더를 만듭니다.
WORKDIR /app

# 3. 필요한 부품 목록(requirements.txt)을 먼저 복사해서 설치합니다.
# (이렇게 하면 코드가 바뀔 때마다 매번 라이브러리를 새로 설치하지 않아 효율적입니다.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 우리 프로젝트 코드 전체(app.py, templates/ 등)를 복사합니다.
COPY . .

# 5. Gunicorn이 사용할 포트를 환경 변수로 받도록 설정합니다.
# Cloud Run이 이 $PORT 변수에 자동으로 포트 번호를 넣어줍니다.
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]