# The champion behind a public URL, in one process.
#
# Hugging Face Spaces is the target because it is free, gives a real URL, holds
# the provider key as a secret rather than in the image, and runs a long-lived
# container -- a multi-hop agent question takes ~40s, which rules out most
# serverless hosts.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    AGENT_TOOL_PATH=/app \
    ESP_STATE=/app/state \
    PORT=7860

WORKDIR /app
COPY pyproject.toml README.md ./
COPY esp ./esp
RUN pip install --no-cache-dir -e "."

COPY apps ./apps
COPY registries ./registries
COPY results ./results
COPY tests/fixtures ./tests/fixtures

# Spaces runs as uid 1000 and the app writes rendered networks under /app.
RUN useradd -m -u 1000 user && mkdir -p /app/state /app/.esp-networks \
    && chown -R user:user /app
USER user

EXPOSE 7860
HEALTHCHECK --interval=60s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:7860/health')"

CMD ["python", "apps/web/serve.py"]
