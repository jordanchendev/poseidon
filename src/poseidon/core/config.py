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

    model_config = {"env_prefix": "POSEIDON_", "env_file": ".env"}


settings = Settings()
