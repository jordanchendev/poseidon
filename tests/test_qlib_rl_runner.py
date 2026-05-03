"""Phase 90 — qlib RL runner + bridge-module tests.

Wave 1 (Plan 90-02) added the two bridge module tests
(``test_dataset_adapter_roundtrip``, ``test_order_builder_*``).

Wave 2 (Plan 90-03) replaces the ``test_run_all_four_algos`` skip stub with:

* ``test_run_all_four_algos_unit`` — monkey-patches
  ``qlib.rl.contrib.backtest.backtest`` to a stub that writes a 5-row dummy
  CSV. Exercises the orchestration logic (config emission, output paths,
  D-11 partial-tolerance for PPO/OPDS) on Mac dev without qlib installed.
* ``test_run_all_four_algos_stormtrooper`` — gated by ``STORMTROOPER=1``
  env var; runs the real qlib RL backtest in the qlib-research container.
* Additional unit tests covering ``_emit_backtest_config`` (TWSE
  09:00/13:30/270 ticks override) and ``run_one`` (NotImplementedError for
  PPO/OPDS).
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from poseidon.qlib.rl_dataset_adapter import write_pickle
from poseidon.qlib.rl_order_builder import build_orders

# Note: the two bridge tests below run BEFORE any qlib import — they are
# pure-pandas and must be collectable on Mac dev. The qlib runner tests are
# gated separately at function level.


def test_dataset_adapter_roundtrip(tmp_path: Path):
    """write_pickle round-trips a 270-bar synthetic 1m fixture."""
    from tests.conftest import make_synthetic_1m_ohlcv

    df = make_synthetic_1m_ohlcv(periods=270, base_price=16500.0)
    out = tmp_path / "data.pkl"
    written = write_pickle(df, instrument="TX", out_path=out)

    assert written == out
    loaded = pd.read_pickle(out)
    assert len(loaded) == 270
    assert list(loaded.columns) == ["$open", "$high", "$low", "$close", "$volume"]
    assert loaded.index.names == ["datetime", "instrument"]
    # Inner level all "TX"
    assert (loaded.index.get_level_values("instrument") == "TX").all()


def test_order_builder_shape(tmp_path: Path):
    """build_orders emits 3 rows × [amount, order_type] for 3 trigger days."""
    triggers = [
        pd.Timestamp("2024-03-04"),
        pd.Timestamp("2024-04-15"),
        pd.Timestamp("2024-05-22"),
    ]
    out = tmp_path / "orders.pkl"
    written = build_orders(
        triggers,
        leg="TX",
        notional=1_000_000.0,
        out_path=out,
        fill_time="09:00",
    )

    assert written == out
    loaded = pd.read_pickle(out)
    assert len(loaded) == 3
    assert list(loaded.columns) == ["amount", "order_type"]
    assert loaded.index.names == ["datetime", "instrument"]
    assert (loaded["order_type"] == "BUY").all()
    # All instruments == "TX"
    assert (loaded.index.get_level_values("instrument") == "TX").all()


def test_order_builder_sell_leg(tmp_path: Path):
    """leg='0050' → order_type='SELL' (short the spot ETF)."""
    triggers = [pd.Timestamp("2024-03-04")]
    out = tmp_path / "orders_sell.pkl"
    build_orders(triggers, leg="0050", notional=1_000_000.0, out_path=out)
    loaded = pd.read_pickle(out)
    assert loaded["order_type"].iloc[0] == "SELL"
    assert loaded.index.get_level_values("instrument")[0] == "0050"


# --- rl_runner.py unit tests (no qlib required) ---


def test_run_one_ppo_raises_not_implemented(tmp_path: Path):
    """run_one("ppo", ...) signals Wave 3 boundary cleanly."""
    from poseidon.qlib.rl_runner import run_one

    with pytest.raises(NotImplementedError, match="Wave 3"):
        run_one(
            algo="ppo",
            leg="TX",
            run_dir=tmp_path,
            ohlcv_pickle_path=tmp_path / "x.pkl",
            orders_pickle_path=tmp_path / "y.pkl",
        )


def test_run_one_opds_raises_not_implemented(tmp_path: Path):
    """run_one("opds", ...) signals Wave 3 boundary cleanly."""
    from poseidon.qlib.rl_runner import run_one

    with pytest.raises(NotImplementedError, match="Wave 3"):
        run_one(
            algo="opds",
            leg="0050",
            run_dir=tmp_path,
            ohlcv_pickle_path=tmp_path / "x.pkl",
            orders_pickle_path=tmp_path / "y.pkl",
        )


def test_run_one_unknown_algo_raises_value_error(tmp_path: Path):
    """run_one rejects unknown algo names with ValueError."""
    from poseidon.qlib.rl_runner import run_one

    with pytest.raises(ValueError, match="unknown algo"):
        run_one(
            algo="not_an_algo",
            leg="TX",
            run_dir=tmp_path,
            ohlcv_pickle_path=tmp_path / "x.pkl",
            orders_pickle_path=tmp_path / "y.pkl",
        )


def test_emit_backtest_config_twse_overrides(tmp_path: Path):
    """Emitted YAML must encode TWSE 09:00–13:30 / 270 ticks (RESEARCH Pitfall 4)."""
    from poseidon.qlib.rl_runner import _emit_backtest_config

    cfg = _emit_backtest_config(
        algo="twap",
        leg="TX",
        ohlcv_pickle_path=tmp_path / "ohlcv.pkl",
        orders_pickle_path=tmp_path / "orders.pkl",
        run_dir=tmp_path,
    )

    assert cfg.exists()
    text = cfg.read_text()
    assert 'start_time: "09:00"' in text
    assert 'end_time: "13:30"' in text
    assert "data_ticks: 270" in text


def test_emit_backtest_config_vwap_uses_hlc3(tmp_path: Path):
    """VWAP config encodes HLC/3 fallback for $vwap (matches dataset_builder)."""
    from poseidon.qlib.rl_runner import _emit_backtest_config

    cfg = _emit_backtest_config(
        algo="vwap",
        leg="0050",
        ohlcv_pickle_path=tmp_path / "ohlcv.pkl",
        orders_pickle_path=tmp_path / "orders.pkl",
        run_dir=tmp_path,
    )
    text = cfg.read_text()
    assert "($high + $low + $close) / 3" in text


def test_emit_backtest_config_ppo_raises(tmp_path: Path):
    """_emit_backtest_config refuses PPO/OPDS too (defense in depth)."""
    from poseidon.qlib.rl_runner import _emit_backtest_config

    with pytest.raises(NotImplementedError):
        _emit_backtest_config(
            algo="ppo",
            leg="TX",
            ohlcv_pickle_path=tmp_path / "x.pkl",
            orders_pickle_path=tmp_path / "y.pkl",
            run_dir=tmp_path,
        )


def _stub_qlib_backtest_module(monkeypatch, tmp_run_dir: Path) -> list[str]:
    """Install a stub `qlib.rl.contrib.backtest` that writes a 5-row dummy CSV.

    Returns the list of config-paths the stub was invoked with (mutated in
    place by the stub on each call) so tests can assert call ordering.
    """
    calls: list[str] = []

    def _stub_backtest(config_path: str) -> None:
        calls.append(config_path)
        # Parse the run_dir / algo / leg from the config path so the dummy
        # CSV lands at the same place run_one will look for it.
        cfg_text = Path(config_path).read_text()
        # Extract output_dir line.
        out_line = next(ln for ln in cfg_text.splitlines() if ln.startswith("output_dir:"))
        output_dir = Path(out_line.split(":", 1)[1].strip())
        ckpt_dir = output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # 5-row dummy backtest_result.csv with the qlib upstream columns
        # (pa = price advantage ×10⁴ = bps; ffr = fill fulfillment ratio).
        df = pd.DataFrame(
            {
                "instrument": ["TX"] * 5,
                "datetime": pd.date_range("2026-01-01", periods=5, freq="D"),
                "pa": [1.5, 2.0, 0.8, -0.5, 1.1],
                "ffr": [1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        df.set_index(["instrument", "datetime"]).to_csv(ckpt_dir / "backtest_result.csv")

    # Build stub package tree: qlib → qlib.rl → qlib.rl.contrib → qlib.rl.contrib.backtest
    pkgs = ["qlib", "qlib.rl", "qlib.rl.contrib", "qlib.rl.contrib.backtest"]
    for name in pkgs:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = mod
            monkeypatch.setattr(sys.modules[name], "__path__", [], raising=False)
            # Ensure cleanup
            monkeypatch.delitem(sys.modules, name, raising=False)
            sys.modules[name] = mod
    # Attach the callable to qlib.rl.contrib.backtest
    sys.modules["qlib.rl.contrib.backtest"].backtest = _stub_backtest  # type: ignore[attr-defined]

    return calls


def test_run_all_four_algos_unit(tmp_path: Path, monkeypatch):
    """Unit variant: stub qlib.rl.contrib.backtest and exercise orchestration.

    Asserts:
      * TWAP + VWAP each emit a backtest_result.csv via the stub.
      * PPO/OPDS surface as PARTIAL with "Wave 3" in the error.
      * No exception escapes run_all_algos_legs (D-11 partial-tolerance).
    """
    from poseidon.qlib.rl_runner import run_all_algos_legs
    from tests.conftest import make_synthetic_1m_ohlcv

    # Build TX + ETF synthetic pickles.
    tx_df = make_synthetic_1m_ohlcv(periods=270, base_price=16500.0)
    etf_df = make_synthetic_1m_ohlcv(periods=270, base_price=130.0)
    tx_pkl = write_pickle(tx_df, instrument="TX", out_path=tmp_path / "tx.pkl")
    etf_pkl = write_pickle(etf_df, instrument="0050", out_path=tmp_path / "etf.pkl")

    # Build TX + ETF orders.
    triggers = [pd.Timestamp("2026-01-01")]
    tx_orders = build_orders(triggers, leg="TX", notional=1_000_000.0, out_path=tmp_path / "tx_o.pkl")
    etf_orders = build_orders(triggers, leg="0050", notional=1_000_000.0, out_path=tmp_path / "etf_o.pkl")

    # Stub qlib.rl.contrib.backtest with a callable that writes a dummy CSV.
    calls = _stub_qlib_backtest_module(monkeypatch, tmp_path)

    summary = run_all_algos_legs(
        run_dir=tmp_path,
        ohlcv_pickles={"TX": tx_pkl, "0050": etf_pkl},
        orders_pickles={"TX": tx_orders, "0050": etf_orders},
    )

    # TWAP + VWAP should each have called the stub for TX + 0050 (4 total).
    assert len(calls) == 4
    assert summary["twap"]["status"] == "OK"
    assert summary["vwap"]["status"] == "OK"
    assert "TX" in summary["twap"]["results"]
    assert "0050" in summary["twap"]["results"]

    # PPO + OPDS: PARTIAL with "Wave 3" in the error message.
    assert summary["ppo"]["status"] == "PARTIAL"
    assert "Wave 3" in summary["ppo"]["error"]
    assert summary["opds"]["status"] == "PARTIAL"
    assert "Wave 3" in summary["opds"]["error"]


@pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="Real qlib backtest only runs in stormtrooper qlib-research container",
)
def test_run_all_four_algos_stormtrooper(tmp_path: Path):
    """Integration variant: run real qlib.rl.contrib.backtest on synthetic data.

    Accepts PARTIAL outcomes for TWAP/VWAP (real qlib may fail on synthetic
    data depending on internal validations) but no exception must escape
    run_all_algos_legs (D-11 partial-tolerance).
    """
    pytest.importorskip("qlib")
    from poseidon.qlib.rl_runner import run_all_algos_legs
    from tests.conftest import make_synthetic_1m_ohlcv

    tx_df = make_synthetic_1m_ohlcv(periods=270, base_price=16500.0)
    etf_df = make_synthetic_1m_ohlcv(periods=270, base_price=130.0)
    tx_pkl = write_pickle(tx_df, instrument="TX", out_path=tmp_path / "tx.pkl")
    etf_pkl = write_pickle(etf_df, instrument="0050", out_path=tmp_path / "etf.pkl")

    triggers = [pd.Timestamp("2026-01-01")]
    tx_orders = build_orders(triggers, leg="TX", notional=1_000_000.0, out_path=tmp_path / "tx_o.pkl")
    etf_orders = build_orders(triggers, leg="0050", notional=1_000_000.0, out_path=tmp_path / "etf_o.pkl")

    summary = run_all_algos_legs(
        run_dir=tmp_path,
        ohlcv_pickles={"TX": tx_pkl, "0050": etf_pkl},
        orders_pickles={"TX": tx_orders, "0050": etf_orders},
    )

    assert summary["twap"]["status"] in ("OK", "PARTIAL")
    assert summary["vwap"]["status"] in ("OK", "PARTIAL")
    assert summary["ppo"]["status"] == "PARTIAL"
    assert "Wave 3" in summary["ppo"]["error"]
    assert summary["opds"]["status"] == "PARTIAL"
    assert "Wave 3" in summary["opds"]["error"]
