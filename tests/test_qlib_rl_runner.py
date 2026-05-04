"""Phase 90 — qlib RL runner + bridge-module tests.

Wave 1 (Plan 90-02) added the two bridge module tests
(``test_dataset_adapter_roundtrip``, ``test_order_builder_*``).

Wave 2 (Plan 90-03) added the rule-based runner tests (TWAP+VWAP via
stub-backtest).

Wave 3 (Plan 90-04) lights up PPO + OPDS:

* ``test_run_all_four_algos_unit`` — monkey-patches BOTH
  ``qlib.rl.contrib.backtest.backtest`` (writes a 5-row dummy CSV) AND
  ``subprocess.Popen`` (so :func:`train_one` writes a dummy checkpoint
  without a real qlib container). Asserts all four algos return paths and
  ``summary[algo]["status"] == "OK"``.
* ``test_emit_train_config_twse_overrides`` — emitted PPO train YAML
  carries TWSE 09:00–13:30 / 270 ticks / GPU on.
* ``test_emit_backtest_config_ppo_requires_checkpoint`` — PPO/OPDS branch
  of :func:`_emit_backtest_config` refuses to emit without a checkpoint.
* ``test_run_one_unknown_algo_raises_value_error`` — unchanged.
* ``test_run_all_four_algos_stormtrooper`` — gated by ``STORMTROOPER=1``;
  attempts the real qlib RL backtest in the qlib-research container.
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
    """build_orders emits 3 rows × [amount, order_type] for 3 trigger days.

    Plan 90-04.1 schema:
      * Index: ``[date, instrument]`` (date OUTER per upstream convention)
      * order_type: int (OrderDir.BUY = 1)
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
        fill_time="09:00",
    )

    assert written == out
    loaded = pd.read_pickle(out)
    assert len(loaded) == 3
    assert list(loaded.columns) == ["amount", "order_type"]
    assert loaded.index.names == ["date", "instrument"]
    # OrderDir.BUY = 1 verified live against qlib v0.9.7 in Plan 90-04.1.
    assert (loaded["order_type"] == ORDER_DIR_BUY).all()
    assert (loaded.index.get_level_values("instrument") == "TX").all()


def test_order_builder_sell_leg(tmp_path: Path):
    """leg='0050' → order_type=OrderDir.SELL (0) — short the spot ETF."""
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

    # 100 trigger days × 2 legs = 200 rows total
    assert len(train) == 70 * 2
    assert len(valid) == 15 * 2
    assert len(test) == 15 * 2

    # Chronological — train.max < valid.min < test.min
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
    assert len(df) == 4  # 2 dates × 2 legs
    assert set(df.index.get_level_values("instrument").unique()) == {"TX", "0050"}
    # TX row should have notional 1e6, 0050 should have 5e5
    tx_row = df.xs("TX", level="instrument")
    etf_row = df.xs("0050", level="instrument")
    assert (tx_row["amount"] == 1_000_000.0).all()
    assert (etf_row["amount"] == 500_000.0).all()


def test_orderdir_consistency_probe(tmp_path: Path):
    """If qlib is installed (stormtrooper), the cached OrderDir constants match upstream."""
    from poseidon.qlib.rl_order_builder import _assert_orderdir_consistency

    # Mac dev: qlib not installed — function silently returns. Stormtrooper:
    # raises if OrderDir drifted. Either way, the call must not raise.
    _assert_orderdir_consistency()


# --- rl_runner.py unit tests (no qlib required) ---


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


def test_emit_backtest_config_ppo_requires_checkpoint(tmp_path: Path):
    """PPO/OPDS branch refuses to emit without a checkpoint (defense in depth)."""
    from poseidon.qlib.rl_runner import _emit_backtest_config

    with pytest.raises(ValueError, match="requires checkpoint_path"):
        _emit_backtest_config(
            algo="ppo",
            leg="TX",
            ohlcv_pickle_path=tmp_path / "x.pkl",
            orders_pickle_path=tmp_path / "y.pkl",
            run_dir=tmp_path,
        )

    with pytest.raises(ValueError, match="requires checkpoint_path"):
        _emit_backtest_config(
            algo="opds",
            leg="0050",
            ohlcv_pickle_path=tmp_path / "x.pkl",
            orders_pickle_path=tmp_path / "y.pkl",
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
        ohlcv_pickle_path=tmp_path / "ohlcv.pkl",
        orders_pickle_path=tmp_path / "orders.pkl",
        run_dir=tmp_path,
        checkpoint_path=ckpt,
    )
    text = cfg.read_text()
    assert "SAOEIntStrategy" in text
    assert "class: PPO" in text
    assert f"weight_file: {ckpt}" in text


def test_emit_train_config_twse_overrides(tmp_path: Path):
    """PPO train YAML encodes TWSE 09:00–13:30 / 270 ticks + GPU on."""
    from poseidon.qlib.rl_runner import _emit_train_config

    ckpt_dir = tmp_path / "ckpt"
    cfg = _emit_train_config(
        algo="ppo",
        leg="TX",
        ohlcv_pickle_path=tmp_path / "ohlcv.pkl",
        orders_pickle_path=tmp_path / "orders.pkl",
        run_dir=tmp_path,
        checkpoint_out_dir=ckpt_dir,
    )

    assert cfg.exists()
    assert ckpt_dir.exists()
    text = cfg.read_text()
    assert 'start_time: "09:00"' in text
    assert 'end_time: "13:30"' in text
    assert "data_ticks: 270" in text
    assert "class: PPO" in text
    assert f"checkpoint_path: {ckpt_dir}" in text
    assert "use_cuda: true" in text


def test_emit_train_config_opds_uses_opds_class(tmp_path: Path):
    """OPDS train YAML carries policy.class: OPDS."""
    from poseidon.qlib.rl_runner import _emit_train_config

    cfg = _emit_train_config(
        algo="opds",
        leg="TX",
        ohlcv_pickle_path=tmp_path / "ohlcv.pkl",
        orders_pickle_path=tmp_path / "orders.pkl",
        run_dir=tmp_path,
        checkpoint_out_dir=tmp_path / "ckpt",
    )
    text = cfg.read_text()
    assert "class: OPDS" in text


def test_emit_train_config_rejects_twap(tmp_path: Path):
    """train_one config emitter refuses TWAP/VWAP (rule-based — no train step)."""
    from poseidon.qlib.rl_runner import _emit_train_config

    with pytest.raises(ValueError, match="PPO/OPDS only"):
        _emit_train_config(
            algo="twap",
            leg="TX",
            ohlcv_pickle_path=tmp_path / "x.pkl",
            orders_pickle_path=tmp_path / "y.pkl",
            run_dir=tmp_path,
            checkpoint_out_dir=tmp_path / "ckpt",
        )


def test_train_one_rejects_twap(tmp_path: Path):
    """train_one refuses TWAP/VWAP (rule-based — no train step)."""
    from poseidon.qlib.rl_runner import train_one

    with pytest.raises(ValueError, match="PPO/OPDS only"):
        train_one(
            algo="twap",
            leg="TX",
            run_dir=tmp_path,
            ohlcv_pickle_path=tmp_path / "x.pkl",
            orders_pickle_path=tmp_path / "y.pkl",
        )


def _stub_qlib_backtest_module(monkeypatch, tmp_run_dir: Path) -> list[str]:
    """Install a stub `qlib.rl.contrib.backtest` that writes a 5-row dummy CSV.

    Returns the list of config-paths the stub was invoked with (mutated in
    place by the stub on each call) so tests can assert call ordering.
    """
    calls: list[str] = []

    def _stub_backtest(config_or_path):
        # qlib's real backtest accepts either a config dict (preferred) or
        # the path of a YAML; we accept either to keep the unit test robust.
        if isinstance(config_or_path, dict):
            output_dir_str = config_or_path.get("output_dir") or config_or_path.get("data", {}).get("output_dir")
            if not output_dir_str:
                raise ValueError("stub backtest: no output_dir in config dict")
            calls.append(str(output_dir_str))
            output_dir = Path(output_dir_str)
        else:
            calls.append(str(config_or_path))
            cfg_text = Path(config_or_path).read_text()
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

    def _stub_get_backtest_config_fromfile(path: str):
        # Pass the yaml path through as a string in a dict so the stub backtest
        # has something to grep output_dir out of.
        cfg_text = Path(path).read_text()
        out_line = next(ln for ln in cfg_text.splitlines() if ln.startswith("output_dir:"))
        output_dir = out_line.split(":", 1)[1].strip()
        return {"output_dir": output_dir, "_yaml_path": path}

    # Build stub package tree: qlib → qlib.rl → qlib.rl.contrib → qlib.rl.contrib.backtest
    pkgs = ["qlib", "qlib.rl", "qlib.rl.contrib", "qlib.rl.contrib.backtest"]
    for name in pkgs:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []  # type: ignore[attr-defined]
            sys.modules[name] = mod
            monkeypatch.setattr(sys.modules[name], "__path__", [], raising=False)
            monkeypatch.delitem(sys.modules, name, raising=False)
            sys.modules[name] = mod
    # Attach the callable to qlib.rl.contrib.backtest
    sys.modules["qlib.rl.contrib.backtest"].backtest = _stub_backtest  # type: ignore[attr-defined]
    sys.modules["qlib.rl.contrib.backtest"].get_backtest_config_fromfile = _stub_get_backtest_config_fromfile  # type: ignore[attr-defined]

    return calls


def _stub_train_one(monkeypatch, dummy_checkpoint_size: int = 32) -> list[tuple[str, str]]:
    """Stub :func:`train_one` to write a dummy checkpoint and return its path.

    Returns the list of (algo, leg) tuples the stub was called with.
    """
    from poseidon.qlib import rl_runner

    calls: list[tuple[str, str]] = []

    def _stub(
        algo,
        leg,
        run_dir,
        ohlcv_pickle_path,
        orders_pickle_path,
        time_budget_seconds=14400,
        session=None,
        run_id=None,
        cancel_poll_interval=30.0,
    ):
        calls.append((algo, leg))
        # Write the dummy checkpoint at the canonical location so run_one
        # finds it on the second algo iteration via the cache path.
        ckpt = rl_runner._checkpoint_path(algo)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_bytes(b"X" * dummy_checkpoint_size)
        return ckpt

    monkeypatch.setattr(rl_runner, "train_one", _stub)
    return calls


def test_run_all_four_algos_unit(tmp_path: Path, monkeypatch):
    """Unit variant: stub qlib.rl.contrib.backtest + train_one and exercise orchestration.

    Asserts:
      * TWAP + VWAP each emit a backtest_result.csv via the stub (4 calls).
      * PPO + OPDS each emit a backtest_result.csv via the stub (4 more calls).
      * train_one stub called exactly twice (PPO once, OPDS once — second
        leg reuses the cached checkpoint).
      * Every algo summary["status"] == "OK".
      * D-11/D-12 partial-tolerance still in place (no exception escapes).
    """
    # Redirect AQUARIUM_ROOT to tmp_path so the cache dir is isolated.
    from poseidon.qlib import rl_runner
    from poseidon.qlib.rl_runner import _checkpoint_path, run_all_algos_legs
    from tests.conftest import make_synthetic_1m_ohlcv

    monkeypatch.setattr(rl_runner, "_AQUARIUM_ROOT", tmp_path)
    (tmp_path / ".planning").mkdir()

    # Pre-clean any cached checkpoints (idempotency).
    for algo in ("ppo", "opds"):
        ckpt = _checkpoint_path(algo)
        if ckpt.exists():
            ckpt.unlink()

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
    train_calls = _stub_train_one(monkeypatch)

    summary = run_all_algos_legs(
        run_dir=tmp_path,
        ohlcv_pickles={"TX": tx_pkl, "0050": etf_pkl},
        orders_pickles={"TX": tx_orders, "0050": etf_orders},
    )

    # 4 algos × 2 legs = 8 backtest invocations.
    assert len(calls) == 8

    # train_one called once per RL algo (2 total) — the second leg reuses
    # the cached checkpoint written by the first leg's call.
    assert len(train_calls) == 2
    train_algos = {algo for (algo, _leg) in train_calls}
    assert train_algos == {"ppo", "opds"}

    for algo in ("twap", "vwap", "ppo", "opds"):
        assert summary[algo]["status"] == "OK", f"{algo} expected OK, got {summary[algo]}"
        assert "TX" in summary[algo]["results"]
        assert "0050" in summary[algo]["results"]


def test_run_all_four_algos_partial_tolerance_on_train_failure(tmp_path: Path, monkeypatch):
    """If train_one raises, the algo is recorded PARTIAL and other algos continue.

    D-12 partial-tolerance gate.
    """
    from poseidon.qlib import rl_runner
    from poseidon.qlib.rl_runner import run_all_algos_legs

    monkeypatch.setattr(rl_runner, "_AQUARIUM_ROOT", tmp_path)
    (tmp_path / ".planning").mkdir()

    # Pre-clean any cached checkpoints.
    for algo in ("ppo", "opds"):
        ckpt = rl_runner._checkpoint_path(algo)
        if ckpt.exists():
            ckpt.unlink()

    tx_pkl = tmp_path / "tx.pkl"
    etf_pkl = tmp_path / "etf.pkl"
    tx_orders = tmp_path / "tx_o.pkl"
    etf_orders = tmp_path / "etf_o.pkl"
    for p in (tx_pkl, etf_pkl, tx_orders, etf_orders):
        p.write_bytes(b"")

    _stub_qlib_backtest_module(monkeypatch, tmp_path)

    def _exploding_train_one(algo, leg, *args, **kwargs):
        raise RuntimeError(f"{algo}: GPU OOM (synthetic for D-12 test)")

    monkeypatch.setattr(rl_runner, "train_one", _exploding_train_one)

    summary = run_all_algos_legs(
        run_dir=tmp_path,
        ohlcv_pickles={"TX": tx_pkl, "0050": etf_pkl},
        orders_pickles={"TX": tx_orders, "0050": etf_orders},
    )

    # TWAP/VWAP unaffected — no train step.
    assert summary["twap"]["status"] == "OK"
    assert summary["vwap"]["status"] == "OK"
    # PPO/OPDS PARTIAL with the exploding-train error message surfaced.
    assert summary["ppo"]["status"] == "PARTIAL"
    assert "OOM" in summary["ppo"]["error"]
    assert summary["opds"]["status"] == "PARTIAL"
    assert "OOM" in summary["opds"]["error"]


@pytest.mark.skipif(
    os.environ.get("STORMTROOPER") != "1",
    reason="Real qlib backtest only runs in stormtrooper qlib-research container",
)
def test_run_all_four_algos_stormtrooper(tmp_path: Path):
    """Integration variant: run real qlib.rl.contrib.backtest on synthetic data.

    Accepts PARTIAL outcomes for any algo (real qlib may fail on synthetic
    data depending on internal validations) but no exception must escape
    run_all_algos_legs (D-11 / D-12 partial-tolerance).
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

    for algo in ("twap", "vwap", "ppo", "opds"):
        assert summary[algo]["status"] in ("OK", "PARTIAL")
