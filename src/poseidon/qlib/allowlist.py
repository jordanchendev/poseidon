"""Handler/model class allowlist for Research API (RCE prevention).

Per D-08/D-09: Static Python dicts -- NOT a database table. Changes require
code deployment, not API calls. API callers cannot expand the execution surface.

The ``resolve_handler`` / ``resolve_model`` functions validate user input and
return the fully-qualified import path for the allowed class. Unknown names
raise ``ValueError`` with a descriptive message listing all allowed options.
"""

ALLOWED_HANDLER_CLASSES: dict[str, str] = {
    "Alpha158Handler": "poseidon.qlib.data_handler.PoseidonDataHandler",
    "Alpha360Handler": "poseidon.qlib.data_handler.PoseidonDataHandler",
}

ALLOWED_MODEL_CLASSES: dict[str, str] = {
    "LGBMModel": "qlib.contrib.model.gbdt.LGBMModel",
    "LinearModel": "qlib.contrib.model.linear.LinearModel",
    "XGBModel": "qlib.contrib.model.xgboost.XGBModel",
}


def resolve_handler(name: str) -> str:
    """Return the import path for the given handler class name.

    Raises ``ValueError`` if the name is not in the allowlist.
    """
    if name not in ALLOWED_HANDLER_CLASSES:
        raise ValueError(
            f"Unknown handler_class: {name!r}. "
            f"Allowed: {sorted(ALLOWED_HANDLER_CLASSES)}"
        )
    return ALLOWED_HANDLER_CLASSES[name]


def resolve_model(name: str) -> str:
    """Return the import path for the given model class name.

    Raises ``ValueError`` if the name is not in the allowlist.
    """
    if name not in ALLOWED_MODEL_CLASSES:
        raise ValueError(
            f"Unknown model_class: {name!r}. "
            f"Allowed: {sorted(ALLOWED_MODEL_CLASSES)}"
        )
    return ALLOWED_MODEL_CLASSES[name]
