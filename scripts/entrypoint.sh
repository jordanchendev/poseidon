#!/bin/bash
set -e

echo "Running Alembic migrations..."
alembic upgrade head

echo "Starting Poseidon API..."
exec uvicorn poseidon.main:app --host 0.0.0.0 --port 8000
