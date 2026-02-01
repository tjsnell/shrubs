"""Shared test fixtures."""
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Sample OHLCV data for testing."""
    return pd.DataFrame({
        "open": [100.0, 101.0, 102.0, 101.5, 103.0],
        "high": [101.0, 102.5, 103.0, 102.0, 104.0],
        "low": [99.5, 100.5, 101.0, 100.0, 102.5],
        "close": [100.5, 102.0, 101.5, 101.0, 103.5],
        "volume": [1000, 1100, 900, 1200, 1050],
    }, index=pd.date_range("2024-01-01", periods=5, freq="D"))


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "equities" / "daily").mkdir(parents=True)
    (data_dir / "futures" / "daily").mkdir(parents=True)
    (data_dir / "options" / "chains").mkdir(parents=True)
    return data_dir
