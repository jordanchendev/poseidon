"""Strategy pipeline: idea -> spec -> backtest -> gates -> shadow paper -> release.

Importing this package must stay side-effect free: no Celery app, no beat
schedule, no database connection. See ``celery_app`` / ``app`` for the runner
entry points.
"""
