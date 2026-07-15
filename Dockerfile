FROM python:3.11-slim

WORKDIR /app

# Copy package
COPY theoldllm/ /app/theoldllm/
COPY setup.py /app/
COPY README.md /app/
COPY railway/server.py /app/server.py

# Install deps (no Chromium/Playwright needed)
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir aiohttp curl_cffi

ENV HOST=0.0.0.0
ENV PORT=8080

EXPOSE 8080

CMD ["python", "/app/server.py"]
