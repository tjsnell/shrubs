"""End-to-end integration test for the pipeline."""
import pytest
from pathlib import Path

import pandas as pd

from shrubs.data.store import DataStore
from shrubs.backtest.engine import BacktestEngine
from shrubs.execution.portfolio import Portfolio


pytestmark = pytest.mark.integration


def test_full_pipeline_dry_run(tmp_path: Path, sample_ohlcv):
    """Run full pipeline in dry-run mode with sample data."""
    # Setup: Create data store with sample data
    store = DataStore(tmp_path)
    symbols = ["AAPL", "MSFT"]

    for symbol in symbols:
        store.save_ohlcv(symbol, sample_ohlcv, "equities", "daily")

    # Step 1: Load data
    data = {}
    for symbol in symbols:
        data[symbol] = store.load_ohlcv(symbol, "equities", "daily")

    assert len(data) == 2

    # Step 2: Run backtest
    def equal_weight(data):
        return pd.DataFrame({
            "symbol": list(data.keys()),
            "weight": [1.0 / len(data)] * len(data),
        })

    engine = BacktestEngine()
    result = engine.run(data, equal_weight)

    assert result.target_positions is not None
    assert len(result.target_positions) == 2

    # Step 3: Compute trades
    portfolio = Portfolio(account_value=100_000)
    prices = {s: data[s]["close"].iloc[-1] for s in symbols}
    trades = portfolio.compute_trades(result.target_positions, prices)

    assert len(trades) == 2
    assert all(t["action"] == "BUY" for t in trades)
