# Playwright needs a real display: Cloudflare Turnstile does not solve in
# headless Chromium, so the facade runs inside Xvfb.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

COPY pyproject.toml README.md ./
COPY theoldllm ./theoldllm

# xvfb-run needs xauth; real Chrome is much less likely to be challenged than
# Playwright's bundled Chromium, so install it and select it with a channel.
RUN pip install ".[proxy]" \
    && python -m playwright install --with-deps chromium \
    && apt-get update \
    && apt-get install -y --no-install-recommends xvfb xauth wget ca-certificates \
    && wget -qO /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && apt-get install -y --no-install-recommends /tmp/chrome.deb \
    && rm -f /tmp/chrome.deb \
    && rm -rf /var/lib/apt/lists/*

ENV THEOLDLLM_CHANNEL=chrome

ENV PORT=8080 \
    THEOLDLLM_STORAGE=/app/data/session.json
EXPOSE 8080

# Railway mounts a volume here if you add one; otherwise it is just a cache.
RUN mkdir -p /app/data

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8080')+'/health').read()"

CMD xvfb-run -a python -m theoldllm serve --host 0.0.0.0 --port "${PORT:-8080}"
