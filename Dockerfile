# Playwright needs a real display: Cloudflare Turnstile does not solve in
# headless Chromium, so the facade runs inside Xvfb.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

COPY pyproject.toml README.md ./
COPY theoldllm ./theoldllm

RUN pip install ".[proxy]" \
    && python -m playwright install --with-deps chromium \
    && apt-get update \
    && apt-get install -y --no-install-recommends xvfb \
    && rm -rf /var/lib/apt/lists/*

ENV PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/health').read()"

CMD xvfb-run -a python -m theoldllm serve --host 0.0.0.0 --port "${PORT:-8080}"
