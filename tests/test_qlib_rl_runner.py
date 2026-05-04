"""Phase 90 — qlib RL runner + bridge-module tests.

Plan 90-04.1 rewrite. After the qlib data-format adapter (Task 1) and
order-builder OrderDir-int rewrite (Task 2), this test suite covers:

* Wave 1 bridge module sanity (rl_dataset_adapter / rl_order_builder).
* Plan 90-04.1 emitter/runner unit tests with the new signatures
  (``bin_dir`` / ``pickle_dir`` / ``order_dir`` instead of pickle paths).
* Plan 90-04.1 ``test_run_all_four_algos_unit`` — full DataFrame +
  triggers entry path with stubs covering ``write_qlib_data_dir`` (real),
  ``build_orders_split`` (real), ``train_one`` (stub), and
  ``qlib.rl.contrib.backtest`` (stub).
* ``test_run_all_four_algos_stormtrooper`` — STORMTROOPER=1-gated;
  attempts the real qlib RL backtest on synthetic data.
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
    assert (loaded.index.get_level_values("instrument") == "TX").all()


def test_order_builder_shape(tmp_path: Path):
    """build_orders emits 3 rows × [amount, order_type] for 3 trigger days.

    Plan 90-04.1 schema: index ``[date, instrument]``, order_type int.
    """
    from poseidon.qlib.rl_order_builder import ORDER_DIR_BUY

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
    )

    assert written == out
    loaded = pd.read_pickle(out)
    assert len(loaded) == 3
    assert list(loaded.columns) == ["amount", "order_type"]
    assert loaded.index.names == ["date", "instrument"]
    assert (loaded["order_type"] == ORDER_DIR_BUY).all()
    assert (loaded.index.get_level_values("instrument") == "TX").all()


def test_order_builder_sell_leg(tmp_path: Path):
    """leg='0050' → order_type=OrderDir.SELL (0)."""
    from poseidon.qlib.rl_order_builder import ORDER_DIR_SELL

    triggers = [pd.Timestamp("2024-03-04")]
    out = tmp_path / "orders_sell.pkl"
    build_orders(triggers, leg="0050", notional=1_000_000.0, out_path=out)
    loaded = pd.read_pickle(out)
    assert loaded["order_type"].iloc[0] == ORDER_DIR_SELL
    assert loaded.index.get_level_values("instrument")[0] == "0050"


def test_build_orders_split_chronological(tmp_path: Path):
    """build_orders_split chunks 100 trigger days chronologically into 70/15/15."""
    from poseidon.qlib.rl_order_builder import build_orders_split

    triggers = pd.date_range("2024-01-01", periods=100, freq="3D").tolist()
    paths = build_orders_split(
        triggers,
        legs={"TX": 1_000_000.0, "0050": 1_000_000.0},
        out_dir=tmp_path,
    )

    train = pd.read_pickle(paths["train"])
    valid = pd.read_pickle(paths["valid"])
    test = pd.read_pickle(paths["test"])

    assert len(train) == 70 * 2
    assert len(valid) == 15 * 2
    assert len(test) == 15 * 2

    train_dates = train.index.get_level_values("date")
    valid_dates = valid.index.get_level_values("date")
    test_dates = test.index.get_level_values("date")
    assert train_dates.max() < valid_dates.min()
    assert valid_dates.max() < test_dates.min()


def test_build_orders_multileg_combines_legs(tmp_path: Path):
    """build_orders_multileg writes one pickle with rows for each (date, leg) pair."""
    from poseidon.qlib.rl_order_builder import build_orders_multileg

    triggers = [pd.Timestamp("2024-03-04"), pd.Timestamp("2024-04-15")]
    out = tmp_path / "all_orders.pkl"
    build_orders_multileg(
        triggers,
        legs={"TX": 1_000_000.0, "0050": 500_000.0},
        out_path=out,
    )

    df = pd.read_pickle(out)
    assert len(df) == 4
    assert set(df.index.get_level_values("instrument").unique()) == {"TX", "0050"}
    tx_row = df.xs("TX", level="instrument")
    etf_row = df.xs("0050", level="instrument")
    assert (tx_row["amount"] == 1_000_000.0).all()
    assert (etf_row["amount"] == 500_000.0).all()


def test_orderdir_consistency_probe():
    """If qlib is installed (stormtrooper), the cached OrderDir constants match upstream."""
    from poseidon.qlib.rl_order_builder import _assert_orderdir_consistency

    _assert_orderdir_consistency()


# ---------------------------------------------------------------------------
# rl_runner.py emitter unit tests (no qlib required)
# ---------------------------------------------------------------------------


def test_run_one_unknown_algo_raises_value_error(tmp_path: Path):
    """run_one rejects unknown algo names with ValueError."""
    from poseidon.qlib.rl_runner import run_one

    with pytest.raises(ValueError, match="unknown algo"):
        run_one(
            algo="not_an_algo",
            leg="TX",
            run_dir=tmp_path,
            bin_dir=tmp_path / "bin",
            pickle_dir=tmp_path / "pickle",
            order_dir=tmp_path / "orders",
            test_orders_pickle_path=tmp_path / "orders" / "test",
        )


def test_emit_backtest_config_twap_upstream_schema(tmp_path: Path):
    """TWAP backtest YAML uses upstream schema: provider_uri_1min + order_file."""
    from poseidon.qlib.rl_runner import _emit_backtest_config

    cfg = _emit_backtest_config(
        algo="twap",
        leg="TX",
        bin_dir=tmp_path / "bin",
        pickle_dir=tmp_path / "pickle",
        orders_pickle_path=tmp_path / "test_orders",
        run_dir=tmp_path,
    )

    assert cfg.exists()
    text = cfg.read_text()
    assert 'start_time: "09:00"' in text
    assert 'end_time: "13:30"' in text
    assert 'data_granularity: "1min"' in text
    assert "provider_uri_1min:" in text
    assert "order_file:" in text
    assert "TWAPStrategy" in text


def test_emit_backtest_config_vwap_uses_hlc3(tmp_path: Path):
    """VWAP config encodes HLC/3 fallback in deal_price."""
    from poseidon.qlib.rl_runner import _emit_backtest_config

    cfg = _emit_backtest_config(
        algo="vwap",
        leg="0050",
        bin_dir=tmp_path / "bin",
        pickle_dir=tmp_path / "pickle",
        orders_pickle_path=tmp_path / "test_orders",
        run_dir=tmp_path,
    )
    text = cfg.read_text()
    assert "($high + $low + $close) / 3" in text


def test_emit_backtest_config_ppo_requires_checkpoint(tmp_path: Path):
    """PPO/OPDS branch refuses to emit without a checkpoint."""
    from poseidon.qlib.rl_runner import _emit_backtest_config

    with pytest.raises(ValueError, match="requires checkpoint_path"):
        _emit_backtest_config(
            algo="ppo",
            leg="TX",
            bin_dir=tmp_path / "bin",
            pickle_dir=tmp_path / "pickle",
            orders_pickle_path=tmp_path / "test_orders",
            run_dir=tmp_path,
        )


def test_emit_backtest_config_ppo_with_checkpoint_succeeds(tmp_path: Path):
    """PPO with checkpoint emits SAOEIntStrategy + weight_file pointing at the ckpt."""
    from poseidon.qlib.rl_runner import _emit_backtest_config

    ckpt = tmp_path / "checkpoint.pth"
    ckpt.write_bytes(b"dummy")
    cfg = _emit_backtest_config(
        algo="ppo",
        leg="TX",
        bin_dir=tmp_path / "bin",
        pickle_dir=tmp_path / "pickle",
        orders_pickle_path=tmp_path / "test_orders",
        run_dir=tmp_path,
        checkpoint_path=ckpt,
    )
    text = cfg.read_text()
    assert "SAOEIntStrategy" in text
    assert "class: PPO" in text
    assert f"weight_file: {ckpt}" in text


def test_emit_train_config_upstream_schema(tmp_path: Path):
    """PPO train YAML matches upstream schema (env, simulator, data.source)."""
    from poseidon.qlib.rl_runner import _emit_train_config

    ckpt_dir = tmp_path / "ckpt"
    cfg = _emit_train_config(
        algo="ppo",
        leg="TX",
        pickle_dir=tmp_path / "pickle",
        order_dir=tmp_path / "orders",
        run_dir=tmp_path,
        checkpoint_out_dir=ckpt_dir,
    )

    assert cfg.exists()
    assert ckpt_dir.exists()
    text = cfg.read_text()
    # Upstream schema sections (RESEARCH §Sources train_ppo.yml)
    assert "simulator:" in text
    assert "env:" in text
    assert "data:\n  source:" in text
    assert "feature_root_dir:" in text
    assert "order_dir:" in text
    assert "trainer:" in text
    assert "max_epoch:" in text  # NOT max_iters
    assert "class: PPO" in text
    assert "use_cuda: true" in text
    # TWSE overrides
    assert "data_granularity: 1" in text
    assert "data_ticks: 270" in text


def test_emit_train_config_opds_uses_opds_class(tmp_path: Path):
    """OPDS train YAML carries policy.class: OPDS and reward.class: PAPenaltyReward."""
    from poseidon.qlib.rl_runner import _emit_train_config

    cfg = _emit_train_config(
        algo="opds",
        leg="TX",
        pickle_dir=tmp_path / "pickle",
        order_dir=tmp_path / "orders",
        run_dir=tmp_path,
        checkpoint_out_dir=tmp_path / "ckpt",
    )
    text = cfg.read_text()
    assert "class: OPDS" in text
    assert "class: PAPenaltyReward" in text


def test_emit_train_config_rejects_twap(tmp_path: Path):
    """train_one config emitter refuses TWAP/VWAP."""
    from poseidon.qlib.rl_runner import _emit_train_config

    with pytest.raises(ValueError, match="PPO/OPDS only"):
        _emit_train_config(
            algo="twap",
            leg="TX",
            pickle_dir=tmp_path / "pickle",
            order_dir=tmp_path / "orders",
            run_dir=tmp_path,
            checkpoint_out_dir=tmp_path / "ckpt",
        )


def test_train_one_rejects_twap(tmp_path: Path):
    """train_one refuses TWAP/VWAP."""
    from poseidon.qlib.rl_runner import train_one

    with pytest.raises(ValueError, match="PPO/OPDS only"):
        train_one(
            algo="twap",
            leg="TX",
            run_dir=tmp_path,
            pickle_dir=tmp_path / "pickle",
            order_dir=tmp_path / "orders",
        )


# ---------------------------------------------------------------------------
# Run-loop integration tests (with stubs)
# ---------------------------------------------------------------------------


def _stub_qlib_backtest_module(monkeypatch) -> list[dict]:
    """Stub ``qlib.rl.contrib.backtest`` to write a 5-row dummy CSV.

    Returns a list of config-dicts the stub was invoked with so tests can
    assert call ordering.
    """
    calls: list[dict] = []

    def _stub_backtest(config_dict):
        calls.append(config_dict)
        # Pull the output_dir out of the config the stub get_backtest_config_fromfile
        # returned. Plan 90-04.1 backtest YAML stores it at top-level.
        output_dir_str = config_dict.get("output_dir") or config_dict.get("_output_dir")
        if not output_dir_str:
            raise ValueError("stub backtest: no output_dir in config dict")
        output_dir = Path(output_dir_str)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Match what qlib.rl.utils.log.CsvWriter writes — one CSV per
        # instrument (file name = "<instrument>.csv").
        leg = config_dict.get("_leg", "TX")
        df = pd.DataFrame(
            {
                "instrument": [leg] * 5,
                "datetime": pd.date_range("2026-01-01", periods=5, freq="D"),
                "pa": [1.5, 2.0, 0.8, -0.5, 1.1],
                "ffr": [1.0, 1.0, 1.0, 1.0, 1.0],
            }
        )
        df.set_index(["instrument", "datetime"]).to_csv(output_dir / f"{leg}.csv")

    def _stub_get_backtest_config_fromfile(path: str):
        cfg_text = Path(path).read_text()
        out_line = next(ln for ln in cfg_text.splitlines() if ln.startswith("output_dir:"))
        output_dir = out_line.split(":", 1)[1].strip().rstrip("/")
        # Pull leg from the comment header for stub differentiation.
        leg = "TX"
        for ln in cfg_text.splitlines():
            if "leg=" in ln:
                leg = ln.split("leg=")[1].strip()
                break
        return {
            "output_dir": output_dir,
            "_yaml_path": path,
            "_leg": leg,
        }

    pkgs = ["qlib", "qlib.rl", "qlib.rl.contrib", "qlib.rl.contrib.backtest"]
    for name in pkgs:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = mod
            monkeypatch.setattr(sys.modules[name], "__path__", [], raising=False)
            monkeypatch.delitem(sys.modules, name, raising=False)
            sys.modules[name] = mod
    sys.modules["qlib.rl.contrib.backtest"].backtest = _stub_backtest  # type: ignore[attr-defined]
    sys.modules["qlib.rl.contrib.backtest"].get_backtest_config_fromfile = (  # type: ignore[attr-defined]
        _stub_get_backtest_config_fromfile
    )

    return calls


def _stub_train_one(monkeypatch, dummy_checkpoint_size: int = 32) -> list[tuple[str, str]]:
    """Stub :func:`train_one` to write a dummy checkpoint."""
    from poseidon.qlib import rl_runner

    calls: list[tuple[str, str]] = []

    def _stub(
        algo,
        leg,
        run_dir,
        pickle_dir,
        order_dir,
        time_budget_seconds=14400,
        session=None,
        run_id=None,
        cancel_poll_interval=30.0,
    ):
        calls.append((algo, leg))
        ckpt = rl_runner._checkpoint_path(algo)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"X" * dummy_checkpoint_size)
        return ckpt

    monkeypatch.setattr(rl_runner, "train_one", _stub)
    return calls


def _synthetic_run_inputs(tmp_path: Path):
    """Build the (ohlcv_dataframes, triggers, leg_notionals) tuple a run consumes."""
    from tests.conftest import make_synthetic_1m_ohlcv

    # 20 trigger days so build_orders_split can produce non-empty 70/15/15
    # buckets (14/3/3).
    triggers = [pd.Timestamp("2024-01-01") + pd.Timedelta(days=i * 7) for i in range(20)]
    pieces_tx, pieces_etf = [], []
    for d in triggers:
        idx = pd.date_range(d.strftime("%Y-%m-%d") + " 09:00:00", periods=270, freq="1min")
        pieces_tx.append(make_synthetic_1m_ohlcv(periods=270, base_price=16500.0).set_axis(idx))
        pieces_etf.append(make_synthetic_1m_ohlcv(periods=270, base_price=130.0).set_axis(idx))
    tx_df = pd.concat(pieces_tx)
    etf_df = pd.concat(pieces_etf)
    return {"TX": tx_df, "0050": etf_df}, triggers


def test_run_all_four_algos_unit(tmp_path: Path, monkeypatch):
    """Plan 90-04.1: full DataFrame + triggers entry path with stubs.

    Asserts:
      * materialize_qlib_data writes bin_dir + pickle_dir + order splits.
      * 4 algos × 2 legs = 8 backtest invocations through the qlib stub.
      * train_one stub called exactly twice (once per RL algo; second leg
        reuses cached checkpoint).
      * Every algo summary["status"] == "OK".
    """
    from poseidon.qlib import rl_runner
    from poseidon.qlib.rl_runner import _checkpoint_path, run_all_algos_legs

    monkeypatch.setattr(rl_runner, "_AQUARIUM_ROOT", tmp_path)
    (tmp_path / ".planning").mkdir()

    for algo in ("ppo", "opds"):
        ckpt = _checkpoint_path(algo)
        if ckpt.exists():
            ckpt.unlink()

    ohlcv_dfs, triggers = _synthetic_run_inputs(tmp_path)

    calls = _stub_qlib_backtest_module(monkeypatch)
    train_calls = _stub_train_one(monkeypatch)

    summary = run_all_algos_legs(
        run_dir=tmp_path,
        ohlcv_dataframes=ohlcv_dfs,
        triggers=triggers,
    )

    # 4 algos × 2 legs = 8 backtest invocations.
    assert len(calls) == 8

    # train_one called once per RL algo (second leg reuses cached ckpt).
    assert len(train_calls) == 2
    assert {algo for (algo, _leg) in train_calls} == {"ppo", "opds"}

    for algo in ("twap", "vwap", "ppo", "opds"):
        assert summary[algo]["status"] == "OK", f"{algo} expected OK, got {summary[algo]}"
        assert "TX" in summary[algo]["results"]
        assert "0050" in summary[algo]["results"]

    # qlib data dir was materialized exactly once (bin/pickle/orders all present).
    qlib_data_root = tmp_path / "qlib-data"
    assert (qlib_data_root / "bin" / "calendars" / "1min.txt").exists()
    assert (qlib_data_root / "pickle" / "feature" / "TX.pkl").exists()
    assert (qlib_data_root / "orders" / "train").exists()


def test_run_all_four_algos_partial_tolerance_on_train_failure(tmp_path: Path, monkeypatch):
    """If train_one raises, the algo is recorded PARTIAL and other algos continue (D-12)."""
    from poseidon.qlib import rl_runner
    from poseidon.qlib.rl_runner import run_all_algos_legs

    monkeypatch.setattr(rl_runner, "_AQUARIUM_ROOT", tmp_path)
    (tmp_path / ".planning").mkdir()

    for algo in ("ppo", "opds"):
        ckpt = rl_runner._checkpoint_path(algo)
        if ckpt.exists():
            ckpt.unlink()

    ohlcv_dfs, triggers = _synthetic_run_inputs(tmp_path)
    _stub_qlib_backtest_module(monkeypatch)

    def _exploding_train_one(algo, leg, *args, **kwargs):
        raise RuntimeError(f"{algo}: GPU OOM (synthetic for D-12 test)")

    monkeypatch.setattr(rl_runner, "train_one", _exploding_train_one)

    summary = run_all_algos_legs(
        run_dir=tmp_path,
        ohlcv_dataframes=ohlcv_dfs,
        triggers=triggers,
    )

    assert summary["twap"]["status"] == "OK"
    assert summary["vwap"]["status"] == "OK"
    assert summary["ppo"]["status"] == "PARTIAL"
    assert "OOM" in summary["ppo"]["error"]
    assert summary["opds"]["status"] == "PARTIAL"
    assert "OOM" in summary["opds"]["error"]


@pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="Real qlib backtest only runs in stormtrooper qlib-research container",
)
def test_run_all_four_algos_stormtrooper(tmp_path: Path, monkeypatch):
    """Integration variant: run real qlib.rl.contrib.backtest on synthetic data.

    Accepts PARTIAL outcomes (qlib may fail on synthetic data); just asserts
    no exception escapes (D-11 / D-12 partial-tolerance).
    """
    pytest.importorskip("qlib")
    from poseidon.qlib import rl_runner
    from poseidon.qlib.rl_runner import run_all_algos_legs

    monkeypatch.setattr(rl_runner, "_AQUARIUM_ROOT", tmp_path)
    (tmp_path / ".planning").mkdir()

    ohlcv_dfs, triggers = _synthetic_run_inputs(tmp_path)
    summary = run_all_algos_legs(
        run_dir=tmp_path,
        ohlcv_dataframes=ohlcv_dfs,
        triggers=triggers,
        # Bound training time aggressively for the integration test.
        algos=["twap", "vwap"],
    )

    for algo in ("twap", "vwap"):
        assert summary[algo]["status"] in ("OK", "PARTIAL")
