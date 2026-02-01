"""Parquet-based data storage."""
from pathlib import Path
from typing import Literal

import pandas as pd


AssetType = Literal["equities", "futures", "options"]
Timeframe = Literal["daily", "minute"]


class DataStore:
    """Manages Parquet file storage for market data."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    def _get_path(
        self, symbol: str, asset_type: AssetType, timeframe: Timeframe
    ) -> Path:
        """Get file path for a symbol."""
        return self.base_dir / asset_type / timeframe / f"{symbol}.parquet"

    def save_ohlcv(
        self,
        symbol: str,
        data: pd.DataFrame,
        asset_type: AssetType,
        timeframe: Timeframe,
    ) -> None:
        """Save OHLCV data to Parquet."""
        path = self._get_path(symbol, asset_type, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(path)

    def load_ohlcv(
        self, symbol: str, asset_type: AssetType, timeframe: Timeframe
    ) -> pd.DataFrame | None:
        """Load OHLCV data from Parquet. Returns None if not found."""
        path = self._get_path(symbol, asset_type, timeframe)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def list_symbols(self, asset_type: AssetType, timeframe: Timeframe) -> list[str]:
        """List available symbols for asset type and timeframe."""
        dir_path = self.base_dir / asset_type / timeframe
        if not dir_path.exists():
            return []
        return [p.stem for p in dir_path.glob("*.parquet")]
