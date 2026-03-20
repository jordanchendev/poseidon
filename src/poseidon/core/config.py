from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://poseidon:poseidon@localhost:5432/poseidon"
    redis_url: str = "redis://localhost:6379/0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""
    finmind_token: str = ""
    symbols_config: str = "config/symbols.yaml"

    model_config = {"env_prefix": "POSEIDON_", "env_file": ".env"}


settings = Settings()
