"""Tests for order management."""
import pytest
from unittest.mock import Mock, patch
from shrubs.execution.orders import OrderManager, OrderStatus


def test_create_market_order():
    """Create a market order."""
    manager = OrderManager(ib_client=None, dry_run=True)
    order = manager.create_order("AAPL", "BUY", 100, order_type="MARKET")

    assert order["symbol"] == "AAPL"
    assert order["action"] == "BUY"
    assert order["shares"] == 100
    assert order["order_type"] == "MARKET"


def test_dry_run_does_not_submit():
    """Dry run logs but doesn't submit orders."""
    manager = OrderManager(ib_client=None, dry_run=True)
    order = manager.create_order("AAPL", "BUY", 100)

    result = manager.submit(order)

    assert result["status"] == OrderStatus.DRY_RUN
    assert result["submitted"] is False


def test_order_target_percent():
    """Calculate order for target percentage allocation."""
    manager = OrderManager(ib_client=None, dry_run=True)

    order = manager.order_target_percent(
        symbol="AAPL",
        target_pct=0.25,
        account_value=100_000,
        current_shares=0,
        current_price=150.0,
    )

    # 25% of 100k = 25k, at $150 = 166 shares
    assert order["shares"] == 166
    assert order["action"] == "BUY"


def test_order_target_percent_sell():
    """Target percent below current triggers sell."""
    manager = OrderManager(ib_client=None, dry_run=True)

    order = manager.order_target_percent(
        symbol="AAPL",
        target_pct=0.10,
        account_value=100_000,
        current_shares=100,  # Currently have 100 shares @ $150 = $15k = 15%
        current_price=150.0,
    )

    # Target 10% = $10k = 66 shares, currently have 100, need to sell 34
    assert order["shares"] == 34
    assert order["action"] == "SELL"
