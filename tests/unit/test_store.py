"""Tests for data store."""
import pandas as pd
import pytest
from pathlib import Path
from shrubs.data.store import DataStore


def test_save_and_load_ohlcv(tmp_data_dir: Path, sample_ohlcv: pd.DataFrame):
    """Round-trip save and load of OHLCV data."""
    store = DataStore(tmp_data_dir)

    store.save_ohlcv("AAPL", sample_ohlcv, asset_type="equities", timeframe="daily")
    loaded = store.load_ohlcv("AAPL", asset_type="equities", timeframe="daily")

    pd.testing.assert_frame_equal(loaded, sample_ohlcv, check_freq=False)


def test_load_nonexistent_returns_none(tmp_data_dir: Path):
    """Loading nonexistent symbol returns None."""
    store = DataStore(tmp_data_dir)
    result = store.load_ohlcv("FAKE", asset_type="equities", timeframe="daily")
    assert result is None


def test_list_symbols(tmp_data_dir: Path, sample_ohlcv: pd.DataFrame):
    """List available symbols for asset type."""
    store = DataStore(tmp_data_dir)
    store.save_ohlcv("AAPL", sample_ohlcv, asset_type="equities", timeframe="daily")
    store.save_ohlcv("MSFT", sample_ohlcv, asset_type="equities", timeframe="daily")

    symbols = store.list_symbols(asset_type="equities", timeframe="daily")
    assert set(symbols) == {"AAPL", "MSFT"}
