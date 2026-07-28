FROM python:3.10-slim

# Use uv instead of pip, see https://github.com/astral-sh/uv
RUN pip install --no-cache-dir uv==0.10.4

ENV VIRTUAL_ENV=/opt/venv

RUN uv venv $VIRTUAL_ENV

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --no-cache --active

COPY . .

# Log level for the agent: TRACE, DEBUG, INFO, WARN, ERROR or CRITICAL.
# The LiveKit CLI only takes it as a flag, so the CMD below forwards it.
ENV LOG_LEVEL=INFO

CMD ["sh", "-c", "exec python3 main.py start --log-level \"$LOG_LEVEL\""]
