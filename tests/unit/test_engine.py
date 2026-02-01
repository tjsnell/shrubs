"""Tests for backtest engine."""
import pandas as pd
import pytest
from shrubs.backtest.engine import BacktestEngine, BacktestResult


@pytest.fixture
def multi_symbol_data() -> dict[str, pd.DataFrame]:
    """Sample data for multiple symbols."""
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    return {
        "AAPL": pd.DataFrame({
            "open": range(100, 120),
            "high": range(101, 121),
            "low": range(99, 119),
            "close": range(100, 120),
            "volume": [1000] * 20,
        }, index=idx),
        "MSFT": pd.DataFrame({
            "open": range(200, 220),
            "high": range(201, 221),
            "low": range(199, 219),
            "close": range(200, 220),
            "volume": [2000] * 20,
        }, index=idx),
    }


def dummy_strategy(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal weight strategy for testing."""
    symbols = list(data.keys())
    return pd.DataFrame({
        "symbol": symbols,
        "weight": [1.0 / len(symbols)] * len(symbols),
    })


def test_run_backtest(multi_symbol_data: dict[str, pd.DataFrame]):
    """Run a backtest with dummy strategy."""
    engine = BacktestEngine()
    result = engine.run(multi_symbol_data, dummy_strategy)

    assert isinstance(result, BacktestResult)
    assert result.target_positions is not None
    assert len(result.target_positions) == 2


def test_backtest_result_has_metrics(multi_symbol_data: dict[str, pd.DataFrame]):
    """Backtest result includes performance metrics."""
    engine = BacktestEngine()
    result = engine.run(multi_symbol_data, dummy_strategy)

    assert hasattr(result, "sharpe_ratio")
    assert hasattr(result, "max_drawdown")
    assert hasattr(result, "total_return")
