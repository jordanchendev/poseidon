from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://poseidon:poseidon@localhost:5432/poseidon"
    redis_url: str = "redis://localhost:6379/0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""
    finmind_token: str = ""
    finlab_api_token: str = ""
    tw_stock_data_backend: Literal["finlab", "finmind", "shioaji"] = "finlab"
    tw_futures_data_backend: Literal["finlab", "finmind"] = "finlab"
    shioaji_data_api_key: str = ""
    shioaji_data_secret_key: str = ""
    shioaji_data_simulation: bool = True
    shioaji_trade_api_key: str = ""
    shioaji_trade_secret_key: str = ""
    shioaji_trade_simulation: bool = True
    shioaji_ca_cert_path: str = ""
    shioaji_ca_password: str = ""
    shioaji_person_id: str = ""
    # Binance credentials — read-only key for data fetching
    binance_api_key: str = ""
    # Binance credentials — trading key pair (added when ready for live trading)
    binance_trade_key: str = ""
    binance_trade_secret: str = ""
    symbols_config: str = "config/symbols.yaml"
    model_artifact_dir: str = "/data/models"

    # Rate limit settings (conservative values per provider)
    ratelimit_finmind_hourly: int = 500
    ratelimit_finlab_per_minute: int = 120
    ratelimit_shioaji_per_minute: int = 60
    ratelimit_yfinance_daily: int = 900
    ratelimit_ccxt_per_minute: int = 1200

    # Phase 39 D-16/D-17: bulk backfill budgets, isolated from live ingest so a
    # one-shot historical job cannot starve the periodic ingest path's quota.
    ratelimit_finmind_backfill_hourly: int = 100
    ratelimit_yfinance_backfill_daily: int = 200
    ratelimit_ccxt_backfill_per_minute: int = 300

    # Circuit breaker settings
    circuit_failure_threshold: int = 5
    circuit_open_timeout: int = 60
    circuit_failure_window: int = 300

    # Cache settings
    cache_enabled: bool = True

    # VaR computation settings
    var_lookback_days: int = 252
    var_min_observations: int = 30
    var_confidence_levels: str = "0.95,0.99"  # comma-separated

    # Drawdown monitoring thresholds (per D-10)
    drawdown_warning_pct: float = 0.05
    drawdown_alert_pct: float = 0.10
    drawdown_critical_pct: float = 0.20

    # VaR cache TTL (2x hourly schedule = 2 hours, per pitfall 5)
    var_cache_ttl: int = 7200

    # Monte Carlo VaR settings (per D-05)
    mc_simulations: int = 10_000

    # Prediction settings
    predict_confidence_threshold: float = 0.6

    # Data quality scoring weights (per D-08, must sum to 1.0)
    quality_weight_completeness: float = 0.30
    quality_weight_consistency: float = 0.25
    quality_weight_anomaly_free: float = 0.25
    quality_weight_timeliness: float = 0.20

    # Phase 38 D-04: ingest cursor mode flag (legacy|cursor). Defaults to legacy
    # for instant rollback; 48h shadow-mode runbook flips to "cursor" via the
    # unprefixed env var INGEST_CURSOR_MODE (see docs/runbooks/phase38-shadow-mode.md).
    ingest_cursor_mode: Literal["legacy", "cursor"] = Field(
        default="legacy",
        validation_alias="INGEST_CURSOR_MODE",
    )

    # Phase 40 D-10/D-11: per-(market, interval) freshness SLA in SECONDS.
    # Read by the ingest_freshness_watchdog beat task (plan 40-03). Keys are
    # tuples serialized as ``"market:interval"`` (pydantic-settings cannot
    # load dict-of-tuple-keys from env vars cleanly); the watchdog splits the
    # colon at lookup time. Values are the maximum tolerated delta between
    # ``now()`` and ``ingest_state.last_successful_ts`` before the tuple
    # counts as a violation — HC.io then fires a Telegram alert via the
    # dead-man's-switch topology (D-12).
    freshness_sla: dict[str, int] = Field(
        default_factory=lambda: {
            "crypto_perp:4h": 18000,  # 5h
            "crypto_spot:1h": 7200,  # 2h
            "crypto_spot:1d": 108000,  # 30h
            "tw_stock:1d": 108000,  # 30h (covers weekends + holidays)
            "tw_futures:1d": 108000,  # 30h
            "us_stock:1d": 108000,  # 30h
        }
    )

    # Phase 40 D-12..D-14: Healthchecks.io single-check URL covering both
    # FRESH-03 (violation -> /fail) and FRESH-04 (dead-man's-switch via
    # silence). Empty string = no-op (local/dev safe; watchdog logs the
    # summary but skips the HTTP call). Ping is fire-and-forget with a
    # short timeout — HC.io outages must NOT poison the watchdog task.
    healthchecks_freshness_url: str = Field(
        default="",
        validation_alias="HEALTHCHECKS_FRESHNESS_URL",
    )

    # Phase 40 D-20: which markets dispatch ``read_ohlcv(interval='1d')``
    # to ``ohlcv_1d_cagg`` (Phase 40 plan 40-04). Markets NOT in this list
    # fall through to raw ``ohlcv`` rows storing native 1d data
    # (tw_stock, tw_futures, us_stock). The default list matches the
    # only markets with sub-daily raw data today.
    cagg_1d_markets: list[str] = Field(
        default_factory=lambda: ["crypto_perp", "crypto_spot"]
    )

    model_config = {
        "env_prefix": "POSEIDON_",
        "env_file": ".env",
        "populate_by_name": True,
        "extra": "ignore",
    }


settings = Settings()
