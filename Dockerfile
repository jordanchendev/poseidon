FROM python:3.13-slim

WORKDIR /app

# Install system dependencies for psycopg2, uv, and TA-Lib
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libc6-dev libpq-dev wget make && \
    wget -q https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz && \
    tar xzf ta-lib-0.6.4-src.tar.gz && \
    cd ta-lib-0.6.4 && ./configure --prefix=/usr && make -j$(nproc) && make install && \
    cd .. && rm -rf ta-lib-0.6.4 ta-lib-0.6.4-src.tar.gz && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-dev --no-editable
COPY config/ config/
COPY alembic/ alembic/
COPY alembic.ini .

COPY scripts/ scripts/
RUN chmod +x scripts/entrypoint.sh

ENV PATH="/app/.venv/bin:$PATH"
CMD ["scripts/entrypoint.sh"]
