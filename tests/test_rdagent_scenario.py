from __future__ import annotations

import pytest

from poseidon.rdagent.scenario import ALLOWED_THESIS_CLASSES


def test_allowed_thesis_classes_are_the_exact_six():
    expected = {
        "basis-style",
        "momentum_trend",
        "mean_reversion",
        "vol_regime",
        "cross_asset_spread",
        "calendar_effect",
    }
    assert expected == ALLOWED_THESIS_CLASSES


def test_scenario_class_pickle_roundtrip():
    import pickle

    from poseidon.rdagent.scenario import PoseidonQuantScenario

    assert pickle.loads(pickle.dumps(PoseidonQuantScenario)) is PoseidonQuantScenario


def test_upstream_quant_scenario_keeps_factor_and_model_coder_contracts(tmp_path):
    """No-paid construction uses run-scoped HDF data and upstream coder prompts."""
    qlib = pytest.importorskip("qlib")
    pytest.importorskip("rdagent")

    from poseidon.rdagent.dataset_builder import ensure_qlib_bin_dump
    from poseidon.workers.qlib_rdagent_tasks import _install_factor_data

    provider = ensure_qlib_bin_dump(["TX", "0050"])
    qlib.init(provider_uri=str(provider), region="cn")
    restore = _install_factor_data(tmp_path / "run", provider)
    try:
        from poseidon.rdagent.scenario import PoseidonQuantScenario

        PoseidonQuantScenario.set_challenge("mean reversion")
        scenario = PoseidonQuantScenario()
        factor = scenario.get_scenario_all_desc(action="factor")
        model = scenario.get_scenario_all_desc(action="model")
        assert "daily_pv.h5" in factor
        assert "factor" in factor.lower() and "output" in factor.lower()
        assert "model" in model.lower() and "output" in model.lower()
        assert "cn_data" not in factor
        assert "mean reversion" in factor
    finally:
        restore()
