FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HOME=/tmp HF_HOME=/tmp/huggingface
WORKDIR /app
COPY requirements.txt .
# CPU wheels avoid shipping CUDA packages on a small evaluation server.
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check
RUN groupadd --gid 10001 evalora && useradd --uid 10001 --gid 10001 --no-create-home evalora \
    && mkdir -p /var/lib/evalora/reports /var/lib/evalora/state \
    && chown -R evalora:evalora /var/lib/evalora
COPY --chown=evalora:evalora app app
COPY --chown=evalora:evalora dashboard dashboard
COPY --chown=evalora:evalora experiments experiments
COPY --chown=evalora:evalora deploy deploy
USER 10001:10001
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-proxy-headers", "--no-access-log", "--limit-concurrency", "20", "--timeout-keep-alive", "5"]
