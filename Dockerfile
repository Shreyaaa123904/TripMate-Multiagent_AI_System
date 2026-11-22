FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Application code stays root-owned and read-only for the runtime user.
COPY . .

RUN useradd --create-home --uid 1000 appuser \
    && chmod -R a+rX,go-w /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)" || exit 1

# Forwarded headers are honoured only from FORWARDED_ALLOW_IPS (default:
# 127.0.0.1, i.e. no external proxy is trusted). See README "Deployment".
ENV FORWARDED_ALLOW_IPS=127.0.0.1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--timeout-graceful-shutdown", "20"]
