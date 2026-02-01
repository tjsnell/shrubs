"""Backtest engine using VectorBT."""
from dataclasses import dataclass
from typing import Callable

import pandas as pd
import numpy as np


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    target_positions: pd.DataFrame  # symbol, weight columns
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    trade_log: pd.DataFrame | None = None


StrategyFn = Callable[[dict[str, pd.DataFrame]], pd.DataFrame]


class BacktestEngine:
    """Simple backtest engine."""

    def __init__(self, initial_capital: float = 100_000):
        self.initial_capital = initial_capital

    def run(
        self,
        data: dict[str, pd.DataFrame],
        strategy: StrategyFn,
    ) -> BacktestResult:
        """Run a backtest.

        Args:
            data: Dict of symbol to OHLCV DataFrame
            strategy: Function that takes data and returns target weights

        Returns:
            BacktestResult with positions and metrics
        """
        # Get target weights from strategy
        target_positions = strategy(data)

        # Calculate simple metrics based on buy-and-hold of the portfolio
        returns_list = []
        for symbol, df in data.items():
            weight_row = target_positions[target_positions["symbol"] == symbol]
            if len(weight_row) > 0:
                weight = weight_row["weight"].iloc[0]
                symbol_returns = df["close"].pct_change() * weight
                returns_list.append(symbol_returns)

        if returns_list:
            portfolio_returns = pd.concat(returns_list, axis=1).sum(axis=1)
            portfolio_returns = portfolio_returns.dropna()

            sharpe = self._calculate_sharpe(portfolio_returns)
            drawdown = self._calculate_max_drawdown(portfolio_returns)
            total_return = (1 + portfolio_returns).prod() - 1
        else:
            sharpe = 0.0
            drawdown = 0.0
            total_return = 0.0

        return BacktestResult(
            target_positions=target_positions,
            sharpe_ratio=sharpe,
            max_drawdown=drawdown,
            total_return=total_return,
        )

    def _calculate_sharpe(self, returns: pd.Series, risk_free: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        excess = returns - risk_free / 252
        return (excess.mean() / excess.std()) * np.sqrt(252)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min()) if len(drawdown) > 0 else 0.0
