FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked

COPY . .

RUN chmod +x entrypoint.sh

ENTRYPOINT ./entrypoint.sh