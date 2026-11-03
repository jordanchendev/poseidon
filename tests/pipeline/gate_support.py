"""Test-only data sources and strategies for the gate stages (Step 3b).

* :class:`PlantedOrbSource` -- synthetic TX-like 5m day-session bars: a random
  walk with slowly cycling volatility, plus (``edge=True``) an exploitable
  pattern: once the 30-minute opening range breaks, price drifts
  ``drift_points`` further in the break direction over the next
  ``drift_bars`` bars (with the same noise on top).
* :class:`RandomEntryFactory` -- a DSL v2 factory whose strategy enters at
  seeded pseudo-random bars in a random direction (the "random entry" control).
* :class:`DriftOnlySource` + :func:`overnight_long_spec` -- the "pure beta"
  control: no pattern at all, only an upward drift proportional to each move's
  variance (so every holding period earns the same drift per unit of risk as
  buy-and-hold), and a strategy that is long over every overnight gap.
"""

from __future__ import annotations

import copy
import hashlib
import math
import uuid

import numpy as np
import pandas as pd

from poseidon.backtest.dsl_v2_factory import DslV2StrategyFactory
from poseidon.core.canonical import content_sha256
from poseidon.pipeline import artifacts
from poseidon.pipeline.stage_api import StageContext, UpstreamResult
from poseidon.pipeline.state import Verdict
from poseidon.pipeline.window import MARKET_TZ
from poseidon.strategies.dsl.v2.examples import tx_orb30_day
from poseidon.strategies.rule_strategy_v2 import RuleStrategyV2

DAY_BARS = 61  # 08:45 .. 13:45 open-labelled 5m bars


class PlantedOrbSource:
    """Day-session-only TX-like bars for every business day of the requested window."""

    def __init__(
        self,
        *,
        edge: bool = True,
        seed: int = 11,
        drift_per_variance: float = 0.0,
        drift_points: float = 40.0,
        drift_bars: int = 12,
        or_bars: int = 6,
        base_vol: float = 5.0,
        vol_cycle_days: int = 250,
        spread: float = 2.0,
    ):
        self.edge = edge
        self.seed = seed
        # expected move = drift_per_variance x its variance (points^2), gaps and bars alike
        self.drift_per_variance = drift_per_variance
        self.drift_points = drift_points
        self.drift_bars = drift_bars
        self.or_bars = or_bars
        self.base_vol = base_vol
        self.vol_cycle_days = vol_cycle_days
        self.spread = spread
        self.calls: list[tuple] = []
        self.planted_days = 0

    def frame_for(self, first, last) -> pd.DataFrame:
        days = pd.bdate_range(first, last)
        rng = np.random.default_rng(self.seed)
        offsets = pd.to_timedelta(np.arange(DAY_BARS) * 5, unit="min") + pd.Timedelta(hours=8, minutes=45)
        stamps, opens, highs, lows, closes = [], [], [], [], []
        price = 23000.0
        planted = 0
        for k, day in enumerate(days):
            vol = self.base_vol * (1.0 + 0.5 * math.sin(2.0 * math.pi * k / self.vol_cycle_days))
            price += rng.normal(self.drift_per_variance * (3.0 * vol) ** 2, 3.0 * vol)  # overnight gap
            noise = rng.normal(0.0, vol, DAY_BARS)
            o = np.empty(DAY_BARS)
            c = np.empty(DAY_BARS)
            h = np.empty(DAY_BARS)
            low = np.empty(DAY_BARS)
            direction, left = 0, 0
            or_high = or_low = None
            for i in range(DAY_BARS):
                o[i] = price
                step = noise[i] + self.drift_per_variance * vol**2
                if left > 0:
                    step += direction * self.drift_points / self.drift_bars
                    left -= 1
                price = round(price + step)
                c[i] = price
                h[i] = max(o[i], c[i]) + self.spread
                low[i] = min(o[i], c[i]) - self.spread
                if i == self.or_bars - 1:
                    or_high, or_low = h[: self.or_bars].max(), low[: self.or_bars].min()
                elif self.edge and or_high is not None and direction == 0 and i < DAY_BARS - self.drift_bars:
                    if c[i] > or_high:
                        direction, left = 1, self.drift_bars
                    elif c[i] < or_low:
                        direction, left = -1, self.drift_bars
                    if direction:
                        planted += 1
            stamps.append(day + offsets)
            opens.append(o)
            highs.append(h)
            lows.append(low)
            closes.append(c)
        self.planted_days = planted
        index = pd.DatetimeIndex(np.concatenate([s.to_numpy() for s in stamps])).tz_localize(MARKET_TZ)
        return pd.DataFrame(
            {
                "open": np.concatenate(opens),
                "high": np.concatenate(highs),
                "low": np.concatenate(lows),
                "close": np.concatenate(closes),
                "volume": 10.0,
            },
            index=index.tz_convert("UTC"),
        )

    def fetch(self, symbol, market, interval, start, end):
        self.calls.append((symbol, market, interval, start, end))
        first = start.astimezone(MARKET_TZ).date()
        last = (end - pd.Timedelta(seconds=1)).astimezone(MARKET_TZ).date()
        bars = self.frame_for(first, last)
        return bars[(bars.index >= start) & (bars.index < end)]


def planted_orb_spec() -> dict:
    """The design-doc ORB with two ``$param``s: opening-range minutes and the R-multiple target."""

    doc = tx_orb30_day()
    doc["name"] = "tx_orb_planted_control"
    doc["params"] = {
        "or_min": {"type": "int", "low": 15, "high": 45, "step": 15, "default": 30},
        "tp_r": {"type": "float", "low": 1.0, "high": 2.0, "step": 0.5, "default": 1.5},
    }
    for rule in doc["rules"]:
        rule["condition"]["all"][1]["b"]["minutes"] = {"$param": "or_min"}
        rule["risk"]["stop_loss"]["level"]["minutes"] = {"$param": "or_min"}
    doc["risk"]["take_profit"]["value"] = {"$param": "tp_r"}
    return doc


class DriftOnlySource(PlantedOrbSource):
    """No pattern, only buy-and-hold drift: ~50 points/day up (gap ~7, each 5m bar ~0.75 on average).

    Drift per unit of variance is the same for every move, so any subset of
    moves (e.g. only the overnight gaps) has zero alpha over buy-and-hold before
    costs: beta = its share of the variance, expected PnL = beta x buy-and-hold.
    """

    def __init__(self, *, seed: int = 11, drift_per_variance: float = 0.03, **kwargs):
        super().__init__(edge=False, seed=seed, drift_per_variance=drift_per_variance, **kwargs)


def overnight_long_spec() -> dict:
    """Long at the last day bar's close, out at the next day's first bar close; a wide disaster stop."""

    return {
        "dsl_version": 2,
        "name": "overnight_long_pure_beta_control",
        "symbol": "TX",
        "market": "tw_futures",
        "interval": "5m",
        "params": {"sl_pts": {"type": "int", "low": 200, "high": 400, "step": 100, "default": 300}},
        "risk": {
            "stop_loss": {"type": "points", "value": {"$param": "sl_pts"}},
            "max_trades_per_session": 1,
            "close_on_session_change": False,
        },
        "rules": [
            {
                "id": "overnight_long",
                "action": "long",
                "condition": {"type": "time_window", "session": "day", "start": "13:45", "end": "13:45"},
            },
            {
                "id": "morning_exit",
                "action": "close",
                "exit_side": "long",
                "condition": {"type": "time_window", "session": "day", "start": "08:45", "end": "13:00"},
            },
        ],
    }


# ----------------------------------------------------------------- random entry


def _uniform(seed: int, stamp, salt: int) -> float:
    digest = hashlib.sha256(f"{seed}:{pd.Timestamp(stamp).value}:{salt}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


class RandomEntryStrategy(RuleStrategyV2):
    """Enters at seeded pseudo-random bars (probability ``p`` per eligible bar), coin-flip direction.

    The DSL rules only decide *when entries are allowed* (a time window) and
    the exits (points stop / target, flat before the close, one trade per
    session); which bar and which side is pure chance.
    """

    def __init__(self, *, seed: int = 1234, entry_probability: float = 0.04, **kwargs):
        super().__init__(**kwargs)
        self._seed = seed
        self._p = entry_probability

    def _try_entries(self, row):
        stamp = self._cache.index[row]
        if _uniform(self._seed, stamp, 0) >= self._p:
            return []
        side = "long" if _uniform(self._seed, stamp, 1) < 0.5 else "short"
        saved = self._entries
        self._entries = [entry for entry in saved if entry[1].action == side]
        try:
            return super()._try_entries(row)
        finally:
            self._entries = saved


class RandomEntryFactory(DslV2StrategyFactory):
    def build(self, params=None):
        values = self.defaults if params is None else {**self.defaults, **params}
        return RandomEntryStrategy(config=self.template, param_values=values, bar_label=self._bar_label)


def random_entry_spec() -> dict:
    window = {"type": "time_window", "session": "day", "start": "09:15", "end": "13:00"}
    return {
        "dsl_version": 2,
        "name": "random_entry_control",
        "symbol": "TX",
        "market": "tw_futures",
        "interval": "5m",
        "params": {
            "sl_pts": {"type": "int", "low": 20, "high": 40, "step": 10, "default": 30},
            "tp_pts": {"type": "int", "low": 20, "high": 60, "step": 20, "default": 40},
        },
        "risk": {
            "stop_loss": {"type": "points", "value": {"$param": "sl_pts"}},
            "take_profit": {"type": "points", "value": {"$param": "tp_pts"}},
            "flat_before_session_end_minutes": 10,
            "max_trades_per_session": 1,
        },
        "rules": [
            {"id": "rand_long", "action": "long", "condition": {"all": [copy.deepcopy(window)]}},
            {"id": "rand_short", "action": "short", "condition": {"all": [copy.deepcopy(window)]}},
        ],
    }


# ----------------------------------------------------------------- direct stage chain


class ChainLedger:
    """Family ledger for direct stage calls: trial numbers are unique per stage, like per stage_run in the DB."""

    def __init__(self):
        self.stage = None
        self.rows: list[tuple[str, object]] = []

    @property
    def records(self) -> list:
        return [record for _stage, record in self.rows]

    def stage_records(self, stage: str) -> list:
        return [record for name, record in self.rows if name == stage]

    def append(self, record) -> None:
        if any(name == self.stage and old.trial_no == record.trial_no for name, old in self.rows):
            raise ValueError(f"{self.stage} trial {record.trial_no} already recorded")
        self.rows.append((self.stage, record))


class StageChain:
    """Runs stages directly, wiring upstream reports *and files* the way the dispatcher does."""

    def __init__(self, root, spec, config, *, ledger=None, family=None):
        self.root = root
        self.spec = spec
        self.config = config
        self.ledger = ledger if ledger is not None else ChainLedger()
        self.family = family
        self.run_id = uuid.uuid4()
        self.results: dict[str, UpstreamResult] = {
            "spec_generate": UpstreamResult(
                "spec_generate", 1, "0" * 64, "0" * 64, {"spec": spec, "spec_sha256": content_sha256(spec)}
            )
        }
        self.outputs: dict = {}

    def run(self, stage, **extra):
        if isinstance(self.ledger, ChainLedger):
            self.ledger.stage = stage.name
        ctx = StageContext(
            run_id=self.run_id,
            family_id=self.run_id,
            idea_text="gate stage test",
            input_spec_json=self.spec,
            config=self.config,
            config_sha256=content_sha256(self.config),
            stage=stage.name,
            attempt=1,
            upstream={name: self.results[name] for name in stage.upstream_names(self.config)},
            artifacts_root=self.root,
            trials=self.ledger,
            family=self.family,
            **extra,
        )
        ctx.input_sha256 = content_sha256(stage.input_payload(ctx))
        result = stage.run(ctx)
        manifest, uri = [], None
        if result.files:
            directory = self.root / "out" / stage.name / uuid.uuid4().hex
            manifest = artifacts.write_files(directory, result.files)
            uri = directory.as_uri()
        _artifact, sha = artifacts.build_artifact(result.report, manifest)
        self.results[stage.name] = UpstreamResult(
            stage.name, 1, ctx.input_sha256, sha, result.report, artifact_uri=uri, files=tuple(manifest)
        )
        self.outputs[stage.name] = result
        return result

    def through(self, stages) -> list:
        out = []
        for stage in stages:
            result = self.run(stage)
            out.append((stage.name, result))
            if result.verdict != Verdict.PASS:
                break
        return out


# ----------------------------------------------------------------- stub stages


def passing(name: str, requires: tuple[str, ...]):
    """A stub stage that always passes with a summary line."""

    from poseidon.pipeline.stage_api import BaseStage, StageResult

    stage_class = type(
        f"Pass{name.title().replace('_', '')}",
        (BaseStage,),
        {
            "name": name,
            "requires": requires,
            "run": lambda self, ctx: StageResult.passed({"summary": f"{name} ✅ stub"}, reason="stub"),
        },
    )
    return stage_class()


def stub_train_gates(**extra):
    """Synthetic registry whose train gates pass without looking (the real holdout stays)."""

    from poseidon.pipeline.plan import GATE_STAGES

    stages = {name: passing(name, ("optimize",)) for name in GATE_STAGES[: GATE_STAGES.index("wfe")]}
    stages["wfe"] = passing("wfe", ("spec_validate", "optimize"))
    stages.update(extra)
    from tests.pipeline.conftest import synthetic_registry

    return synthetic_registry(**stages)


# ----------------------------------------------------------------- reporting helpers


def diagnose(sessions, run_id, stage_names, registry, *, artifacts_root=None) -> dict:
    """Run gate stages *outside* the run (no attempt rows, wfe trials in memory only).

    For reporting which later train gates a failed candidate would also have
    failed. Never pass ``holdout``: a failed candidate does not get to look.
    ``artifacts_root`` scopes the OHLCV cache; without it the stages use the
    global default root, whose cache is keyed by window only, so a second
    synthetic source in the same process would silently read the first one's bars.
    """

    import time

    from poseidon.pipeline.dispatcher import latest_attempts
    from poseidon.pipeline.family import DbFamily
    from poseidon.pipeline.models import PipelineRun, StageRun
    from poseidon.pipeline.trials import MemoryTrialLedger

    if "holdout" in stage_names:
        raise ValueError("diagnostics never evaluate the locked holdout")
    with sessions() as session:
        run = session.get(PipelineRun, run_id)
        attempts = session.query(StageRun).filter_by(run_id=run_id).all()
        for row in attempts:
            session.expunge(row)
        session.expunge(run)
    latest = latest_attempts(attempts)
    out = {}
    for name in stage_names:
        stage = registry[name]
        upstream = {}
        for required in stage.upstream_names(run.config_json):
            row = latest[required]
            artifact = row.artifact_json or {}
            upstream[required] = UpstreamResult(
                required,
                row.attempt,
                row.input_sha256,
                row.artifact_sha256,
                artifact.get("report"),
                artifact_uri=row.artifact_uri,
                files=tuple(artifact.get("files") or ()),
            )
        ctx = StageContext(
            run_id=run.id,
            family_id=run.family_id,
            idea_text=run.idea_text,
            input_spec_json=run.input_spec_json,
            config=run.config_json,
            config_sha256=run.config_sha256,
            stage=name,
            attempt=0,
            upstream=upstream,
            trials=MemoryTrialLedger(),
            artifacts_root=artifacts_root,
            family=DbFamily(sessions, run_id=run.id, family_id=run.family_id, stage_run_id=uuid.uuid4()),
        )
        started = time.monotonic()
        result = stage.run(ctx)
        out[name] = {
            "verdict": result.verdict.value,
            "reason": result.reason,
            "summary": (result.report or {}).get("summary"),
            "seconds": round(time.monotonic() - started, 1),
            "report": result.report,
        }
    return out


def gate_margins(reports: dict) -> dict:
    """The deciding number of each gate next to its threshold (from stage reports)."""

    out = {}
    for name, report in reports.items():
        if not report or "gate_report" not in report:
            continue
        g = report["gate_report"]
        if name == "param_stability":
            out[name] = {
                "profitable_share": g.get("profitable_share"),
                "min": g.get("profitable_share_min"),
                "profitable/evaluated": f"{g.get('profitable')}/{g.get('evaluated')}",
            }
        elif name == "dsr":
            out[name] = {
                "dsr": g.get("dsr"),
                "min": g.get("dsr_min"),
                "N": g.get("n_trials"),
                "sr0": g.get("sr0"),
                "sharpe_daily": g.get("sharpe"),
                "n_obs": g.get("n_obs"),
            }
        elif name == "cost_stress":
            out[name] = {
                "stressed_total_pnl_twd": g.get("stressed_total_pnl_twd"),
                "min_exclusive": 0,
                "base_total_pnl_twd": g.get("base_total_pnl_twd"),
                "erosion_pct": g.get("pnl_erosion_pct"),
            }
        elif name == "regime_split":
            out[name] = {
                "profitable_buckets": g.get("profitable_buckets"),
                "required": g.get("required_profitable_buckets"),
                "evaluated_buckets": g.get("evaluated_buckets"),
                "min_buckets": g.get("min_buckets"),
                "buckets": {k: (v["trades"], round(v["pnl_twd"])) for k, v in (g.get("buckets") or {}).items()},
            }
        elif name == "wfe":
            out[name] = {
                "wfe": g.get("wfe"),
                "min": g.get("wfe_min"),
                "aggregate_oos_total_pnl_twd": g.get("aggregate_oos_total_pnl_twd"),
                "per_window_wfe": [w.get("wfe") for w in g.get("windows") or []],
            }
        elif name == "benchmark":
            out[name] = {
                "alpha_twd_per_day": g.get("alpha_twd_per_day"),
                "alpha_t_hac": g.get("alpha_t_hac"),
                "t_min": g.get("alpha_t_min"),
                "beta": g.get("beta"),
                "alpha_annualised_twd": g.get("alpha_annualised_twd"),
                "strategy_total_pnl_twd": g.get("strategy_total_pnl_twd"),
                "beta_scaled_benchmark_pnl_twd": g.get("beta_scaled_benchmark_pnl_twd"),
                "time_in_market": g.get("time_in_market"),
                "n_obs": g.get("n_obs"),
            }
        elif name == "holdout":
            out[name] = {
                "sharpe_decay": g.get("sharpe_decay"),
                "max": g.get("sharpe_decay_max"),
                "train_sharpe": g.get("train_sharpe"),
                "holdout_sharpe": g.get("holdout_sharpe"),
                "holdout_total_pnl_twd": g.get("holdout_total_pnl_twd"),
                "holdout_trades": g.get("holdout_trade_count"),
                "min_trades": g.get("min_trades"),
                "passed_criterion": g.get("passed_criterion"),
                "se_criterion": (g.get("criteria") or {}).get("se"),
                "alpha": {
                    key: (g.get("benchmark") or {}).get(key)
                    for key in ("alpha_twd_per_day", "alpha_t_hac", "beta", "time_in_market")
                },
                "alpha_positive_required": g.get("alpha_positive_required"),
            }
        out[name]["summary"] = report.get("summary")
    return out
