"""Signal generation functions."""
import pandas as pd


def momentum(prices: pd.Series, lookback: int = 20) -> pd.Series:
    """Calculate momentum signal as N-day returns.

    Args:
        prices: Price series
        lookback: Number of days for return calculation

    Returns:
        Series of momentum values (positive = uptrend)
    """
    return prices.pct_change(periods=lookback)


def mean_reversion(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate mean reversion signal as deviation from moving average.

    Args:
        prices: Price series
        window: Moving average window

    Returns:
        Series of z-scores from MA (positive = above MA, negative = below)
    """
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    return (prices - ma) / std


def volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate realized volatility.

    Args:
        prices: Price series
        window: Rolling window for volatility

    Returns:
        Series of annualized volatility values
    """
    returns = prices.pct_change()
    return returns.rolling(window=window).std() * (252 ** 0.5)


def combine_signals(
    signals: dict[str, pd.Series],
    weights: dict[str, float],
) -> pd.Series:
    """Combine multiple signals with weights.

    Args:
        signals: Dict of signal name to signal series
        weights: Dict of signal name to weight (should sum to 1)

    Returns:
        Weighted combination of signals
    """
    result = None
    for name, signal in signals.items():
        weight = weights.get(name, 0)
        weighted = signal * weight
        if result is None:
            result = weighted
        else:
            result = result + weighted
    return result
