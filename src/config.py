from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    db_host: str = Field(..., env="DB_HOST")
    db_port: int = Field(5432, env="DB_PORT")
    db_name: str = Field("bybit_shadow", env="DB_NAME")
    db_user: str = Field(..., env="DB_USER")
    db_password: str = Field(..., env="DB_PASSWORD")

    symbols: list[str] = Field(
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        env="SYMBOLS",
    )

    tardis_api_key: str = Field(..., env="TARDIS_API_KEY")

    bybit_api_key: str = Field("", env="BYBIT_API_KEY")
    bybit_api_secret: str = Field("", env="BYBIT_API_SECRET")
    bybit_testnet: bool = Field(True, env="BYBIT_TESTNET")
    bybit_demo: bool = Field(False, env="BYBIT_DEMO")

    live_strategy: str = Field("all", env="LIVE_STRATEGY")
    live_equity: float = Field(1_000.0, env="LIVE_EQUITY")
    live_risk_pct: float = Field(0.02, env="LIVE_RISK_PCT")

    trail_activate_pct: float = Field(0.10, env="TRAIL_ACTIVATE_PCT")
    trail_offset_pct: float = Field(0.03, env="TRAIL_OFFSET_PCT")

    telegram_bot_token: str = Field("", env="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field("", env="TELEGRAM_CHAT_ID")

    # Optional: CoinGlass Crypto API (see src/data/coinglass_liquidation.py)
    coinglass_api_key: str = Field("", env="COINGLASS_API_KEY")
    coinglass_base_url: str = Field(
        "https://open-api-v4.coinglass.com",
        env="COINGLASS_BASE_URL",
    )
    # Hobbyist ~30 RPM; stay slightly under to avoid 429
    coinglass_max_requests_per_minute: int = Field(28, env="COINGLASS_MAX_RPM")
    # Aggregated liquidation history: Hobbyist requires interval >= 4h
    coinglass_min_liquidation_interval: str = Field("4h", env="COINGLASS_MIN_INTERVAL")
    coinglass_exchange_list: str = Field(
        "Binance,Bybit,OKX",
        env="COINGLASS_EXCHANGE_LIST",
    )
    coinglass_enforce_hobbyist_limits: bool = Field(
        True,
        env="COINGLASS_ENFORCE_HOBBYIST_LIMITS",
    )
    # Max bars per symbol per sync (API max 1000; keep moderate for Hobbyist RPM)
    coinglass_sync_history_limit: int = Field(240, env="COINGLASS_SYNC_LIMIT")

    @property
    def is_paper_trading(self) -> bool:
        """True when using testnet OR demo — no real money at risk."""
        return self.bybit_testnet or self.bybit_demo

    @property
    def async_db_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def sync_db_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
