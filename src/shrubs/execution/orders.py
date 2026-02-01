"""Order management and submission."""
from dataclasses import dataclass
from enum import Enum
from typing import Literal
import logging

from shrubs.config import settings


logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    REJECTED = "rejected"
    DRY_RUN = "dry_run"


class OrderManager:
    """Manages order creation and submission to IB."""

    def __init__(self, ib_client, dry_run: bool | None = None):
        self.ib_client = ib_client
        self.dry_run = dry_run if dry_run is not None else settings.dry_run

    def create_order(
        self,
        symbol: str,
        action: Literal["BUY", "SELL"],
        shares: int,
        order_type: str = "MARKET",
        limit_price: float | None = None,
    ) -> dict:
        """Create an order specification.

        Args:
            symbol: Stock symbol
            action: BUY or SELL
            shares: Number of shares
            order_type: MARKET, LIMIT, MOC
            limit_price: Price for limit orders

        Returns:
            Order dict
        """
        return {
            "symbol": symbol,
            "action": action,
            "shares": shares,
            "order_type": order_type,
            "limit_price": limit_price,
            "status": OrderStatus.PENDING,
        }

    def submit(self, order: dict) -> dict:
        """Submit an order to IB.

        Args:
            order: Order dict from create_order

        Returns:
            Order dict with updated status
        """
        if self.dry_run:
            logger.info(f"[DRY RUN] Would submit: {order}")
            return {
                **order,
                "status": OrderStatus.DRY_RUN,
                "submitted": False,
            }

        # TODO: Actual IB submission
        # contract = self.ib_client.create_contract(order["symbol"], ContractType.STOCK)
        # ib_order = MarketOrder(order["action"], order["shares"])
        # trade = self.ib_client.ib.placeOrder(contract, ib_order)

        return {
            **order,
            "status": OrderStatus.SUBMITTED,
            "submitted": True,
        }

    def order_target_percent(
        self,
        symbol: str,
        target_pct: float,
        account_value: float,
        current_shares: int,
        current_price: float,
    ) -> dict:
        """Create order to reach target percentage allocation.

        Args:
            symbol: Stock symbol
            target_pct: Target allocation (0.0 to 1.0)
            account_value: Total account value
            current_shares: Currently held shares
            current_price: Current stock price

        Returns:
            Order dict
        """
        target_value = account_value * target_pct
        target_shares = int(target_value / current_price)
        diff = target_shares - current_shares

        if diff > 0:
            return self.create_order(symbol, "BUY", diff)
        elif diff < 0:
            return self.create_order(symbol, "SELL", abs(diff))
        else:
            return self.create_order(symbol, "BUY", 0)  # No action needed
