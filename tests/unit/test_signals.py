"""Tests for signal generators."""
import pandas as pd
import pytest
import numpy as np
from shrubs.backtest.signals import momentum, mean_reversion, combine_signals


@pytest.fixture
def price_series() -> pd.Series:
    """Sample price series for testing."""
    return pd.Series(
        [100, 102, 101, 105, 103, 108, 107, 110, 112, 115],
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )


def test_momentum_signal(price_series: pd.Series):
    """Momentum signal based on N-day returns."""
    signal = momentum(price_series, lookback=5)

    # Signal should be NaN for first lookback periods
    assert pd.isna(signal.iloc[:5]).all()
    # After lookback, should have values
    assert not pd.isna(signal.iloc[5:]).any()


def test_mean_reversion_signal(price_series: pd.Series):
    """Mean reversion signal based on distance from MA."""
    signal = mean_reversion(price_series, window=3)

    # Signal should be NaN for first window-1 periods
    assert pd.isna(signal.iloc[:2]).all()
    # Positive = above MA (sell signal), negative = below MA (buy signal)
    assert isinstance(signal.iloc[-1], float)


def test_combine_signals():
    """Combine multiple signals with weights."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    signal_a = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=idx)
    signal_b = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1], index=idx)

    combined = combine_signals({"a": signal_a, "b": signal_b}, {"a": 0.6, "b": 0.4})

    expected = 0.6 * signal_a + 0.4 * signal_b
    pd.testing.assert_series_equal(combined, expected)
