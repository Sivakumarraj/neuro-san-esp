# Two things ship in this image, and they have different needs.
#
# The offline half of ESP -- Phase B and C -- runs with no API key, no budget
# and no network. That half is deterministic, so the image either reproduces
# the published numbers or it does not.
#
# The optimiser service needs a provider key, which is passed in at run time and
# never baked in. It also needs apps/ and registries/, which an earlier version
# of this file did not copy: the image could run the offline search but could
# not run the service it was documented as running.
FROM python:3.12-slim

# Never write bytecode or buffer stdout: a container's logs are its only
# window, and a buffered crash shows nothing.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    AGENT_TOOL_PATH=/app \
    AGENT_MANIFEST_FILE=/app/registries/manifest.hocon \
    ESP_STATE=/app/state

WORKDIR /app

# Dependencies first, so a source change does not re-resolve the whole tree.
COPY pyproject.toml README.md ./
COPY esp ./esp
RUN pip install --no-cache-dir -e ".[dev]" pyhocon

COPY tests ./tests
COPY scripts ./scripts
COPY apps ./apps
COPY registries ./registries
COPY tests/fixtures ./tests/fixtures
COPY Makefile ./

# The state directory is a mount point in production -- see compose.yaml. It is
# created here so the image also works without one, rather than failing its
# first wake on a directory that does not exist.
RUN mkdir -p /app/state /app/results

# Run as a normal user. Nothing here needs root, and an evaluation harness
# executes model-chosen tool calls -- a small blast radius is cheap to have.
RUN useradd --create-home --uid 10001 esp \
    && chown -R esp:esp /app
USER esp

# A container that cannot reach its own state or tools should fail the health
# check rather than quietly evaluate nothing. --check runs the preflight and
# exits.
HEALTHCHECK --interval=5m --timeout=30s --start-period=10s --retries=2 \
    CMD python apps/optimizer/run_optimizer.py --check || exit 1

# The offline search is the default because it is the only thing that runs with
# no credentials at all. The service is an explicit command -- see compose.yaml.
CMD ["python", "scripts/offline_search.py", "--pool", "2000"]
