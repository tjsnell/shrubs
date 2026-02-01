"""Portfolio state management."""
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


@dataclass
class Position:
    """A single position in the portfolio."""

    symbol: str
    shares: int
    avg_cost: float


@dataclass
class Trade:
    """A trade to execute."""

    symbol: str
    action: Literal["BUY", "SELL"]
    shares: int
    price: float


class Portfolio:
    """Manages portfolio state and computes required trades."""

    def __init__(self, account_value: float):
        self.account_value = account_value
        self.positions: dict[str, Position] = {}

    def compute_trades(
        self,
        target: pd.DataFrame,
        prices: dict[str, float],
    ) -> list[dict]:
        """Compute trades needed to reach target allocation.

        Args:
            target: DataFrame with symbol, weight columns
            prices: Current prices for each symbol

        Returns:
            List of trade dicts with symbol, action, shares, price
        """
        trades = []

        # Build set of target symbols
        target_symbols = set(target["symbol"].tolist())

        # First: liquidate positions not in target
        for symbol, position in self.positions.items():
            if symbol not in target_symbols:
                price = prices.get(symbol, position.avg_cost)
                trades.append({
                    "symbol": symbol,
                    "action": "SELL",
                    "shares": position.shares,
                    "price": price,
                })

        # Second: compute trades for target positions
        for _, row in target.iterrows():
            symbol = row["symbol"]
            target_weight = row["weight"]
            price = prices.get(symbol)

            if price is None or price <= 0:
                continue

            target_value = self.account_value * target_weight
            target_shares = int(target_value / price)

            current_shares = 0
            if symbol in self.positions:
                current_shares = self.positions[symbol].shares

            diff = target_shares - current_shares

            if diff > 0:
                trades.append({
                    "symbol": symbol,
                    "action": "BUY",
                    "shares": diff,
                    "price": price,
                })
            elif diff < 0:
                trades.append({
                    "symbol": symbol,
                    "action": "SELL",
                    "shares": abs(diff),
                    "price": price,
                })

        return trades
