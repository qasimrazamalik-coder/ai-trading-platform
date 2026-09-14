"""Application configuration with safe live-trading defaults."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QUANT_OS_",
        env_file=".env",
        extra="ignore",
    )

    env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    live_trading_enabled: bool = False
    live_trading_ack: str = ""
    database_url: str = "postgresql://quant:quant@localhost:5432/quant"
    nats_url: str = "nats://localhost:4222"
    valkey_url: str = "redis://localhost:6379/0"
    clickhouse_url: str = "http://localhost:8123"
    service_name: str = Field(default="quant-os-api", min_length=1)

    @model_validator(mode="after")
    def enforce_live_trading_interlock(self) -> "Settings":
        if not self.live_trading_enabled:
            return self
        required_ack = "I_UNDERSTAND_REAL_MONEY_RISK"
        if self.env != "production" or self.live_trading_ack != required_ack:
            raise ValueError(
                "live trading requires production mode and the explicit risk acknowledgement"
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
