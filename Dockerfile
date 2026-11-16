FROM mcr.microsoft.com/playwright/python:v1.55.0-noble

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py agent_protocol.py ./

ENV HEADLESS=true \
    PROFILE_DIR=/data/profile \
    PORT=8080 \
    COOKIES_FILE=/secrets/cookies.json

RUN mkdir -p /data/profile && chown -R pwuser:pwuser /data
USER pwuser
EXPOSE 8080
CMD ["python", "main.py", "serve"]
