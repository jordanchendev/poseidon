"""Queue namespace tests for Poseidon Celery.

Poseidon shares the Redis broker host with other apps. Queue names must be
namespaced so CPU/GPU tasks cannot be consumed by unrelated workers.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CELERY_APP_PATH = ROOT / "src" / "poseidon" / "workers" / "celery_app.py"
CPU_TASKS_PATH = ROOT / "src" / "poseidon" / "workers" / "cpu_tasks.py"
QLIB_TASKS_PATH = ROOT / "src" / "poseidon" / "workers" / "qlib_tasks.py"
COMPOSE_PATH = ROOT / "docker-compose.yml"
FACTOR_ANALYSIS_API_PATH = ROOT / "src" / "poseidon" / "api" / "factor_analysis.py"
RESEARCH_API_PATH = ROOT / "src" / "poseidon" / "api" / "research_api.py"


def test_celery_routes_use_poseidon_queue_namespaces():
    content = CELERY_APP_PATH.read_text()

    assert '"poseidon.workers.gpu_tasks.*": {"queue": "poseidon_gpu"}' in content
    assert '"poseidon.workers.cpu_tasks.*": {"queue": "poseidon_cpu"}' in content
    assert '"poseidon.workers.qlib_tasks.*": {"queue": "poseidon_qlib"}' in content


def test_explicit_task_decorators_use_namespaced_queues():
    cpu_content = CPU_TASKS_PATH.read_text()
    qlib_content = QLIB_TASKS_PATH.read_text()

    assert 'queue="poseidon_cpu"' in cpu_content
    assert 'queue="poseidon_qlib"' in qlib_content


def test_docker_compose_workers_subscribe_to_namespaced_queues():
    content = COMPOSE_PATH.read_text()

    assert "worker -Q poseidon_gpu -c 1 --loglevel=info" in content
    assert "worker -Q poseidon_cpu -c 6 --loglevel=info" in content
    assert "worker -Q poseidon_qlib -c 1 --pool=solo --loglevel=info" in content


def test_api_dispatchers_send_to_namespaced_queues():
    factor_content = FACTOR_ANALYSIS_API_PATH.read_text()
    research_content = RESEARCH_API_PATH.read_text()

    assert "queue=POSEIDON_CPU_QUEUE" in factor_content
    assert "queue=POSEIDON_QLIB_QUEUE" in factor_content
    assert "queue=POSEIDON_QLIB_QUEUE" in research_content
