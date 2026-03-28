from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://poseidon:poseidon@localhost:5432/poseidon"
    redis_url: str = "redis://localhost:6379/0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""
    finmind_token: str = ""
    finlab_api_token: str = ""
    symbols_config: str = "config/symbols.yaml"
    model_artifact_dir: str = "/data/models"

    # Rate limit settings (conservative values per provider)
    ratelimit_finmind_hourly: int = 500
    ratelimit_yfinance_daily: int = 900
    ratelimit_ccxt_per_minute: int = 1200

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

    # Data quality scoring weights (per D-08, must sum to 1.0)
    quality_weight_completeness: float = 0.30
    quality_weight_consistency: float = 0.25
    quality_weight_anomaly_free: float = 0.25
    quality_weight_timeliness: float = 0.20

    model_config = {"env_prefix": "POSEIDON_", "env_file": ".env"}


settings = Settings()
