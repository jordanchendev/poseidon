from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://poseidon:poseidon@localhost:5432/poseidon"
    redis_url: str = "redis://localhost:6379/0"
    redis_celery_url: str = "redis://localhost:6379/0"  # DB 0: Celery broker/backend + RedBeat
    redis_cache_url: str = "redis://localhost:6379/1"  # DB 1: OHLCV + VaR cache + alert streams
    redis_stream_url: str = "redis://localhost:6379/2"  # DB 2: Signal delivery to Thalassa
    redis_ratelimit_url: str = "redis://localhost:6379/3"  # DB 3: Rate limiter + circuit breaker
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""
    api_principals_json: str = "{}"  # SHA-256 credential fingerprints -> fixed principals
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

    decision_loop_execution_enabled: bool = False
    decision_loop_execution_mode: Literal["legacy", "shadow", "decision", "halted"] = "legacy"
    decision_loop_approved_account_scope: str = ""
    decision_loop_approved_market: Literal["", "tw_stock", "crypto_perp"] = ""
    decision_loop_approved_account_generation: str = ""
    decision_loop_legacy_protective_scopes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_execution_gate(self):
        identity = (
            self.decision_loop_approved_account_scope,
            self.decision_loop_approved_market,
            self.decision_loop_approved_account_generation,
        )
        if any(identity) and not all(identity):
            raise ValueError("approved execution identity must be complete")
        for value in (*identity, *self.decision_loop_legacy_protective_scopes):
            if value and (value != value.strip() or "*" in value):
                raise ValueError("execution identities must be exact trimmed text")
        if any(not value for value in self.decision_loop_legacy_protective_scopes):
            raise ValueError("legacy protective scopes must be explicit")
        if len(set(self.decision_loop_legacy_protective_scopes)) != len(self.decision_loop_legacy_protective_scopes):
            raise ValueError("legacy protective scopes must be unique")
        if self.decision_loop_execution_mode in {"legacy", "shadow"} and self.decision_loop_execution_enabled:
            raise ValueError("legacy/shadow cannot enable decision execution")
        if self.decision_loop_execution_mode == "decision" and not self.decision_loop_execution_enabled:
            raise ValueError("decision mode requires the execution gate")
        if self.decision_loop_execution_enabled and not all(identity):
            raise ValueError("enabled execution requires approved identity")
        return self

    # Cache settings
    cache_enabled: bool = True

    # VaR computation settings
    var_lookback_days: int = 252
    var_min_observations: int = 30
    var_confidence_levels: str = "0.95,0.99"  # comma-separated

    # Drawdown monitoring thresholds
    drawdown_warning_pct: float = 0.05
    drawdown_alert_pct: float = 0.10
    drawdown_critical_pct: float = 0.20

    # VaR cache TTL (2x hourly schedule = 2 hours, per pitfall 5)
    var_cache_ttl: int = 7200

    # Monte Carlo VaR settings
    mc_simulations: int = 10_000

    # Prediction settings
    predict_confidence_threshold: float = 0.6

    # Data quality scoring weights (must sum to 1.0)
    quality_weight_completeness: float = 0.30
    quality_weight_consistency: float = 0.25
    quality_weight_anomaly_free: float = 0.25
    quality_weight_timeliness: float = 0.20

    # Thalassa connectivity (always remote, no feature flag)
    thalassa_base_url: str = ""
    thalassa_api_key: str = ""
    thalassa_timeout: float = 30.0
    thalassa_cb_threshold: int = 5
    thalassa_cb_recovery_timeout: float = 60.0

    model_config = {
        "env_prefix": "POSEIDON_",
        "env_file": ".env",
        "populate_by_name": True,
        "extra": "ignore",
    }


settings = Settings()
