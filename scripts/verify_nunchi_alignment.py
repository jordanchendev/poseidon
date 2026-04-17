#!/usr/bin/env python3
"""Phase 65 FIX-01: VotingStrategy Nunchi alignment verification.

Runs BTCUSDT 1h backtest with nunchi_crypto_1h.json config and reports
Sharpe/MaxDD/WFE/trade_count. Per D-01: these metrics ARE the aligned baseline
(the config encodes all Nunchi 103 experiment parameters).

Run inside stormtrooper docker:
  docker compose exec cpu-worker python scripts/verify_nunchi_alignment.py
"""

import json
import sys
from pathlib import Path

from poseidon.backtest.cost_model import COST_MODELS
from poseidon.backtest.metrics import compute_composite_score
from poseidon.backtest.portfolio import SizingConfig, SizingMode
from poseidon.backtest.runner import BacktestRunner
from poseidon.backtest.voting_strategy_factory import VotingStrategyFactory
from poseidon.backtest.walk_forward import WalkForwardAnalyzer, WalkForwardConfig
from poseidon.data.feature_engine import FeatureEngine
from poseidon.data.remote_repository import RemoteDataRepository
from poseidon.risk.engine import RiskEngine


def main() -> int:
    # Load config
    config_path = Path("src/poseidon/strategies/configs/nunchi_crypto_1h.json")
    with open(config_path) as f:
        config = json.load(f)

    print(f"Config: {config_path.name}")
    print(f"  name:            {config.get('name')}")
    print(f"  atr_multiplier:  {config.get('atr_multiplier')}")
    print(f"  min_votes:       {config.get('min_votes')}")
    print(f"  cooldown_bars:   {config.get('cooldown_bars')}")
    print(f"  conviction_gap:  {config.get('conviction_gap')}")
    print(f"  position_pct:    {config.get('position_pct')}")
    print(f"  sub_signals:     {len(config.get('sub_signals', []))}")

    # Build strategy
    strategy = VotingStrategyFactory.from_config(config)

    # Fetch data via Thalassa
    repo = RemoteDataRepository.from_settings()
    ohlcv = repo.read_ohlcv("BTCUSDT", "crypto_spot", "1h")
    print(f"\nOHLCV rows: {len(ohlcv)}, range: {ohlcv.index[0]} to {ohlcv.index[-1]}")

    # Setup dependencies
    feature_engine = FeatureEngine()
    risk_engine = RiskEngine()
    cost_model = COST_MODELS["crypto_spot"]
    sizing = SizingConfig(mode=SizingMode.FIXED_NOTIONAL, notional_pct=0.08)

    # Run backtest
    runner = BacktestRunner(
        strategy=strategy,
        feature_engine=feature_engine,
        risk_engine=risk_engine,
        cost_model=cost_model,
        sizing_config=sizing,
    )
    result = runner.run(ohlcv)

    # Metrics
    metrics = result.metrics
    composite = compute_composite_score(metrics)
    sharpe = metrics.get("sharpe_ratio", 0.0)
    max_dd = metrics.get("max_drawdown", 0.0)
    trade_count = metrics.get("trade_count", 0)
    win_rate = metrics.get("win_rate", 0.0)
    ann_return = metrics.get("annualized_return", 0.0)

    print("\n" + "=" * 60)
    print("FIX-01 VotingStrategy Nunchi Alignment")
    print("=" * 60)
    print(f"Sharpe Ratio:       {sharpe:.4f}")
    print(f"Max Drawdown:       {max_dd:.4f}")
    print(f"Annualized Return:  {ann_return:.4f}")
    print(f"Trade Count:        {trade_count}")
    print(f"Win Rate:           {win_rate:.4f}")
    print(f"Composite Score:    {composite:.4f}")
    print(f"Status:             {result.status}")

    # D-02 checklist
    print("\n--- D-02 Nunchi Alignment Checklist ---")
    print(f"  ATR 5.5x:         {config.get('atr_multiplier')} == 5.5 -> {'PASS' if config.get('atr_multiplier') == 5.5 else 'FAIL'}")
    # Check BB threshold in sub_signals
    bb_threshold = None
    for sig in config.get("sub_signals", []):
        if sig.get("type") == "bollinger_width_percentile":
            bb_threshold = sig.get("threshold")
    print(f"  BB 0.85:           {bb_threshold} == 0.85 -> {'PASS' if bb_threshold == 0.85 else 'FAIL'}")
    print(f"  min_votes:         {config.get('min_votes')}")
    print(f"  cooldown_bars:     {config.get('cooldown_bars')}")
    print(f"  conviction_gap:    {config.get('conviction_gap')}")
    print(f"  Composite Score:   {composite:.4f}")

    # WFE
    print("\n--- Walk-Forward Efficiency ---")
    wf = WalkForwardAnalyzer(
        feature_engine=feature_engine,
        risk_engine=risk_engine,
        cost_model=cost_model,
        sizing_config=sizing,
    )
    wf_config = WalkForwardConfig()
    wf_result = wf.analyze(strategy=strategy, ohlcv=ohlcv, config=wf_config)

    print(f"WFE:                {wf_result.wfe:.4f}")
    print(f"WFE Passed:         {wf_result.passed}")
    print(f"WFE Flags:          {wf_result.flags}")
    print(f"Windows:            {len(wf_result.per_window)}")

    for w in wf_result.per_window:
        print(
            f"  Window {w.window_index}: "
            f"IS ann_ret={w.is_metrics.get('annualized_return', 0):.4f} "
            f"OOS ann_ret={w.oos_metrics.get('annualized_return', 0):.4f} "
            f"IS trades={w.is_trade_count} OOS trades={w.oos_trade_count}"
        )

    # Pass/fail (per D-01: non-degenerate results = pass)
    passed = sharpe > 0 and trade_count > 10
    status = "PASS" if passed else "FAIL"
    print(f"\n>>> FIX-01 Status: {status}")

    if not passed:
        if sharpe <= 0:
            print(f"  FAIL reason: Sharpe ({sharpe:.4f}) <= 0")
        if trade_count <= 10:
            print(f"  FAIL reason: trade_count ({trade_count}) <= 10")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
