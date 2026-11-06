FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    QSCAN_HOST=0.0.0.0 \
    QSCAN_CACHE=/tmp/qscan_cache \
    QSCAN_WEB=/app/docs \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY api /app/api
COPY docs /app/docs

RUN mkdir -p /tmp/qscan_cache && chmod 777 /tmp/qscan_cache

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/ping',timeout=4)" || exit 1

CMD ["python", "/app/api/混合识别服务.py"]
