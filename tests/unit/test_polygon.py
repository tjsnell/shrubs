"""Tests for Polygon client."""
import pandas as pd
import pytest
from unittest.mock import Mock, patch
from shrubs.data.polygon import PolygonClient


@patch("shrubs.data.polygon.RESTClient")
def test_fetch_daily_bars(mock_rest_class):
    """Fetch daily bars from Polygon."""
    # Mock the API response
    mock_client = Mock()
    mock_rest_class.return_value = mock_client

    mock_agg = Mock()
    mock_agg.open = 100.0
    mock_agg.high = 101.0
    mock_agg.low = 99.0
    mock_agg.close = 100.5
    mock_agg.volume = 1000
    mock_agg.timestamp = 1704067200000  # 2024-01-01

    mock_client.list_aggs.return_value = [mock_agg]

    client = PolygonClient(api_key="test_key")
    df = client.fetch_daily("AAPL", start="2024-01-01", end="2024-01-01")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["close"] == 100.5


@patch("shrubs.data.polygon.RESTClient")
def test_empty_response_returns_empty_df(mock_rest_class):
    """Empty API response returns empty DataFrame."""
    mock_client = Mock()
    mock_rest_class.return_value = mock_client
    mock_client.list_aggs.return_value = []

    client = PolygonClient(api_key="test_key")
    df = client.fetch_daily("FAKE", start="2024-01-01", end="2024-01-01")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
