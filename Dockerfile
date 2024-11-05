# 베이스 이미지 설정
FROM python:3.12.7

# 작업 디렉토리 설정
WORKDIR /flaskfolder

# 필요한 파일 복사
COPY requirements.txt ./
COPY app.py ./
COPY model/ ./model/

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# Flask 실행을 위한 환경 변수 설정
ENV FLASK_APP=app.py
ENV FLASK_RUN_PORT=5000

# 외부 포트 노출
EXPOSE 5000

# Flask 서버 시작 명령어
CMD ["flask", "run", "--host=0.0.0.0"]
