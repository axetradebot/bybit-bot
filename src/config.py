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
