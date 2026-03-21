FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for psycopg2 and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-editable

COPY src/ src/
COPY config/ config/
COPY alembic/ alembic/
COPY alembic.ini .

COPY scripts/ scripts/
RUN chmod +x scripts/entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH"
CMD ["scripts/entrypoint.sh"]
