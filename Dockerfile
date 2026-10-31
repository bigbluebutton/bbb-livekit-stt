FROM python:3.11-slim

# Use uv instead of pip, see https://github.com/astral-sh/uv
RUN pip install --no-cache-dir uv==0.10.4

ENV VIRTUAL_ENV=/opt/venv

# Linux names a process after the file it was started from. Through this link,
# the agent's processes get a name process-exporter can match (see README).
RUN uv venv $VIRTUAL_ENV && ln -s python3 "$VIRTUAL_ENV/bin/bbb-livekit-stt"

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./

# Without precompiled bytecode, each new container compiles every module it
# imports before the worker can register.
RUN uv sync --no-cache --active --compile-bytecode

COPY . .

# Log level for the agent: TRACE, DEBUG, INFO, WARN, ERROR or CRITICAL.
# The LiveKit CLI only takes it as a flag, so the CMD below forwards it.
ENV LOG_LEVEL=INFO

# With BBB_STT_PROMETHEUS_PORT set, jobs need PROMETHEUS_MULTIPROC_DIR for their
# metrics to reach /metrics. prometheus_client reads it on first import, so the
# shell exports it before Python starts. Without a port, nothing serves /metrics.
CMD ["sh", "-c", "if [ -n \"$BBB_STT_PROMETHEUS_PORT\" ]; then export PROMETHEUS_MULTIPROC_DIR=\"${PROMETHEUS_MULTIPROC_DIR:-/tmp/prometheus-multiproc}\"; mkdir -p \"$PROMETHEUS_MULTIPROC_DIR\"; fi; exec bbb-livekit-stt main.py start --log-level \"$LOG_LEVEL\""]
