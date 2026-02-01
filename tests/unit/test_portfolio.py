"""Tests for portfolio management."""
import pandas as pd
import pytest
from shrubs.execution.portfolio import Portfolio, Position


def test_compute_trades_from_scratch():
    """Compute trades when starting with empty portfolio."""
    portfolio = Portfolio(account_value=100_000)
    target = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "weight": [0.5, 0.5],
    })
    prices = {"AAPL": 150.0, "MSFT": 300.0}

    trades = portfolio.compute_trades(target, prices)

    assert len(trades) == 2
    aapl_trade = next(t for t in trades if t["symbol"] == "AAPL")
    assert aapl_trade["shares"] > 0  # Buy
    assert aapl_trade["action"] == "BUY"


def test_compute_trades_with_existing_positions():
    """Compute trades to rebalance existing positions."""
    portfolio = Portfolio(account_value=100_000)
    portfolio.positions = {
        "AAPL": Position("AAPL", shares=100, avg_cost=140.0),
    }

    target = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "weight": [0.3, 0.7],
    })
    prices = {"AAPL": 150.0, "MSFT": 300.0}

    trades = portfolio.compute_trades(target, prices)

    # Should sell some AAPL (has 15k, wants 30k worth)
    # Should buy MSFT (has 0, wants 70k worth)
    assert any(t["symbol"] == "MSFT" and t["action"] == "BUY" for t in trades)


def test_compute_trades_liquidation():
    """Compute trades to liquidate unwanted positions."""
    portfolio = Portfolio(account_value=100_000)
    portfolio.positions = {
        "AAPL": Position("AAPL", shares=100, avg_cost=140.0),
        "GOOG": Position("GOOG", shares=50, avg_cost=100.0),  # Not in target
    }

    target = pd.DataFrame({
        "symbol": ["AAPL"],
        "weight": [1.0],
    })
    prices = {"AAPL": 150.0, "GOOG": 120.0}

    trades = portfolio.compute_trades(target, prices)

    goog_trade = next(t for t in trades if t["symbol"] == "GOOG")
    assert goog_trade["action"] == "SELL"
    assert goog_trade["shares"] == 50  # Liquidate all
