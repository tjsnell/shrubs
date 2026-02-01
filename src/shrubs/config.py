"""Configuration management via environment variables."""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # IB Connection
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497  # 7497=TWS paper, 7496=TWS live, 4001=Gateway paper, 4002=Gateway live
    ib_client_id: int = 1

    # Polygon
    polygon_api_key: str = ""

    # Paths
    data_dir: Path = Path("data")

    # Safety
    dry_run: bool = True
    max_position_pct: float = 0.10  # Max 10% in any single position

    class Config:
        env_file = ".env"
        env_prefix = "SHRUBS_"


settings = Settings()
