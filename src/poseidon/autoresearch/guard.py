"""Immutability guard for autoresearch runs.

Uses a ContextVar flag to prevent post-init attribute mutation on protected
classes (FeatureEngine, BacktestRunner, RiskEngine) while allowing normal
construction and strategy config mutation.

Exports:
    _AUTORESEARCH_ACTIVE: ContextVar flag (default False)
    ImmutabilityViolationError: Raised on forbidden mutation
    autoresearch_guard: Class decorator for protected classes
    autoresearch_context: Context manager to activate the guard
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager

_AUTORESEARCH_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_AUTORESEARCH_ACTIVE", default=False
)


class ImmutabilityViolationError(RuntimeError):
    """Raised when autoresearch code attempts to mutate a protected object."""


def autoresearch_guard(cls=None, *, mutable_attrs: frozenset[str] = frozenset()):  # noqa: ANN001, ANN201
    """Class decorator that prevents __setattr__ while autoresearch is active.

    Allows initial construction (__init__) by tracking an ``_ar_initialized``
    flag.  After ``__init__`` completes, mutation is blocked when
    ``_AUTORESEARCH_ACTIVE`` is True.

    Args:
        mutable_attrs: Attribute names that are allowed to mutate even during
            autoresearch (e.g. per-run runtime state that gets reset each trial).
    """
    def _apply(cls):  # noqa: ANN001, ANN201
        original_init = cls.__init__

        def guarded_init(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            object.__setattr__(self, "_ar_initialized", False)
            original_init(self, *args, **kwargs)
            object.__setattr__(self, "_ar_initialized", True)

        def guarded_setattr(self, name, value):  # noqa: ANN001, ANN201
            if (
                getattr(self, "_ar_initialized", False)
                and _AUTORESEARCH_ACTIVE.get(False)
                and not name.startswith("_ar_")
                and name not in mutable_attrs
            ):
                raise ImmutabilityViolationError(
                    f"Cannot modify {type(self).__name__}.{name} during autoresearch run. "
                    f"Only strategy JSON config is mutable."
                )
            object.__setattr__(self, name, value)

        # Create a NEW subclass instead of mutating the original.
        # This ensures `autoresearch_guard(Cls) is not Cls`.
        guarded_cls = type(cls.__name__, (cls,), {
            "__init__": guarded_init,
            "__setattr__": guarded_setattr,
            "__module__": cls.__module__,
            "__qualname__": cls.__qualname__,
        })
        return guarded_cls

    if cls is not None:
        # Called as @autoresearch_guard without arguments
        return _apply(cls)
    # Called as @autoresearch_guard(mutable_attrs=...)
    return _apply


@contextmanager
def autoresearch_context():
    """Context manager that activates the autoresearch immutability guard."""
    token = _AUTORESEARCH_ACTIVE.set(True)
    try:
        yield
    finally:
        _AUTORESEARCH_ACTIVE.reset(token)
