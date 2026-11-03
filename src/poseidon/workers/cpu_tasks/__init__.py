"""CPU worker Celery tasks. Task names keep the ``poseidon.workers.cpu_tasks.<task>`` prefix
that the queue routes and the beat schedule use; each submodule registers its tasks on import."""
