FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --group langchain --no-install-project

COPY . .

ENV PATH="/app/.venv/bin:$PATH"
ENV RAG_BACKEND=native
EXPOSE 8000

CMD ["python", "-m", "api"]
