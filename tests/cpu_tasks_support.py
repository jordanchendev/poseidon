"""Test helpers for the CPU task modules under ``poseidon.workers.cpu_tasks``."""

import importlib
import pkgutil

import poseidon.workers.cpu_tasks as _package

MODULES = tuple(
    importlib.import_module(f"{_package.__name__}.{info.name}") for info in pkgutil.iter_modules(_package.__path__)
)


def cpu_task(name):
    """The task or helper bound to ``name`` in the cpu_tasks modules (one object, wherever it is imported)."""
    values = [vars(module)[name] for module in MODULES if name in vars(module)]
    if not values:
        raise LookupError(f"no cpu_tasks module binds {name}")
    if any(value is not values[0] for value in values):
        raise LookupError(f"cpu_tasks modules bind different objects to {name}")
    return values[0]


def patch_cpu_tasks(monkeypatch, name, value, *, raising=True):
    """Replace ``name`` in every cpu_tasks module that binds it, so every task sees the replacement."""
    bound = [module for module in MODULES if name in vars(module)]
    if not bound and raising:
        raise AttributeError(f"no cpu_tasks module binds {name}")
    for module in bound:
        monkeypatch.setattr(module, name, value)
