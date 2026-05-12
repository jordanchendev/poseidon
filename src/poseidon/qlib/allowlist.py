"""Handler/model class allowlist for Research API (RCE prevention).

Static Python dicts -- NOT a database table. Changes require
code deployment, not API calls. API callers cannot expand the execution surface.

The ``resolve_handler`` / ``resolve_model`` functions validate user input and
return the fully-qualified import path for the allowed class. Unknown names
raise ``ValueError`` with a descriptive message listing all allowed options.
"""

ALLOWED_HANDLER_CLASSES: dict[str, str] = {
    "Alpha158Handler": "poseidon.qlib.data_handler.PoseidonDataHandler",
    "Alpha360Handler": "poseidon.qlib.data_handler.PoseidonDataHandler",
    # qrun-YAML adapter (Pitfall 8). Allowlisted so the tx_basis_vol.yml
    # config resolves through the RCE boundary.
    "PoseidonDataHandlerForQrun": "poseidon.qlib.data_handler_qrun.PoseidonDataHandlerForQrun",
    # qlib stock Alpha158 handler. PoseidonDataHandler (the existing
    # "Alpha158Handler" entry) does not accept the start_time/end_time/
    # fit_start_time/fit_end_time/instruments/label kwargs that qlib's
    # Rolling driver injects via init_instance_by_config. The qlib stock
    # handler does. The qrun work path continues to use "Alpha158Handler"
    # → PoseidonDataHandler for backward compat.
    "Alpha158": "qlib.contrib.data.handler.Alpha158",
}

ALLOWED_MODEL_CLASSES: dict[str, str] = {
    "LGBModel": "qlib.contrib.model.gbdt.LGBModel",
    "LinearModel": "qlib.contrib.model.linear.LinearModel",
    "XGBModel": "qlib.contrib.model.xgboost.XGBModel",
    # Deep-learning additions (LocalformerModel substitutes for TFT —
    # pyqlib 0.9.7 ships no compatible TFT path).
    "ALSTM": "qlib.contrib.model.pytorch_alstm.ALSTM",
    "TRAModel": "qlib.contrib.model.pytorch_tra.TRAModel",
    "LocalformerModel": "qlib.contrib.model.pytorch_localformer.LocalformerModel",
}


def resolve_handler(name: str) -> str:
    """Return the import path for the given handler class name.

    Raises ``ValueError`` if the name is not in the allowlist.
    """
    if name not in ALLOWED_HANDLER_CLASSES:
        raise ValueError(f"Unknown handler_class: {name!r}. Allowed: {sorted(ALLOWED_HANDLER_CLASSES)}")
    return ALLOWED_HANDLER_CLASSES[name]


def resolve_model(name: str) -> str:
    """Return the import path for the given model class name.

    Raises ``ValueError`` if the name is not in the allowlist.
    """
    if name not in ALLOWED_MODEL_CLASSES:
        raise ValueError(f"Unknown model_class: {name!r}. Allowed: {sorted(ALLOWED_MODEL_CLASSES)}")
    return ALLOWED_MODEL_CLASSES[name]
