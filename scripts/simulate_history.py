#!/usr/bin/env python3
"""Simulate historical trading with day-by-day rebalancing."""
import argparse
import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from shrubs.config import settings
from shrubs.data.store import DataStore
from shrubs.data.ib_client import IBClient, ContractType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

UNIVERSES = {
    "test": ["AAPL", "MSFT", "GOOGL"],
    "faang": ["META", "AAPL", "AMZN", "NFLX", "GOOGL"],
    "mag7": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
}


def fetch_historical_data(symbols: list[str], days: int = 180) -> dict[str, pd.DataFrame]:
    """Fetch historical data for all symbols."""
    store = DataStore(settings.data_dir)
    data = {}

    logger.info(f"Fetching {days} days of data for {len(symbols)} symbols...")

    with IBClient(connect=True) as client:
        for symbol in symbols:
            contract = client.create_contract(symbol, ContractType.STOCK)
            df = client.fetch_historical(contract, duration=f"{days} D", bar_size="1 day")
            if len(df) > 0:
                data[symbol] = df
                logger.info(f"  {symbol}: {len(df)} bars")
            else:
                logger.warning(f"  {symbol}: No data")

    return data


def simulate_trading(
    data: dict[str, pd.DataFrame],
    initial_capital: float = 100_000,
    rebalance_threshold: float = 0.02,  # 2% drift triggers rebalance
) -> pd.DataFrame:
    """Simulate day-by-day trading with rebalancing.

    Args:
        data: Dict of symbol to OHLCV DataFrame
        initial_capital: Starting account value
        rebalance_threshold: Rebalance when any position drifts this much from target

    Returns:
        DataFrame with trade log
    """
    symbols = list(data.keys())
    target_weight = 1.0 / len(symbols)

    # Align all data to common dates
    all_dates = None
    for symbol, df in data.items():
        dates = set(df.index.date if hasattr(df.index, 'date') else df.index)
        if all_dates is None:
            all_dates = dates
        else:
            all_dates = all_dates.intersection(dates)

    all_dates = sorted(all_dates)
    logger.info(f"Simulating {len(all_dates)} trading days")

    # Build price matrix
    prices = pd.DataFrame(index=all_dates)
    for symbol, df in data.items():
        df_copy = df.copy()
        df_copy.index = df_copy.index.date if hasattr(df_copy.index, 'date') else df_copy.index
        prices[symbol] = df_copy["close"]

    prices = prices.dropna()

    # Initialize portfolio
    holdings = {sym: 0 for sym in symbols}  # shares held
    cash = initial_capital
    trades = []

    # Day 0: Initial purchase
    day0 = prices.index[0]
    day0_prices = prices.loc[day0]

    logger.info(f"\n{'='*60}")
    logger.info(f"DAY 0: {day0} - INITIAL PORTFOLIO")
    logger.info(f"{'='*60}")

    for symbol in symbols:
        price = day0_prices[symbol]
        target_value = initial_capital * target_weight
        shares = int(target_value / price)
        holdings[symbol] = shares
        cost = shares * price
        cash -= cost
        trades.append({
            "date": day0,
            "symbol": symbol,
            "action": "BUY",
            "shares": shares,
            "price": price,
            "value": cost,
            "reason": "Initial purchase",
        })
        logger.info(f"  BUY {shares:4} {symbol:5} @ ${price:>8.2f} = ${cost:>10,.2f}")

    logger.info(f"  Cash remaining: ${cash:,.2f}")

    # Simulate each subsequent day
    for i, date in enumerate(prices.index[1:], 1):
        day_prices = prices.loc[date]

        # Calculate current portfolio value and weights
        portfolio_value = cash
        current_values = {}
        for symbol in symbols:
            value = holdings[symbol] * day_prices[symbol]
            current_values[symbol] = value
            portfolio_value += value

        current_weights = {sym: val / portfolio_value for sym, val in current_values.items()}

        # Check for drift
        max_drift = max(abs(current_weights[sym] - target_weight) for sym in symbols)

        if max_drift >= rebalance_threshold:
            # Rebalance needed
            logger.info(f"\n{'='*60}")
            logger.info(f"DAY {i}: {date} - REBALANCE (max drift: {max_drift:.1%})")
            logger.info(f"{'='*60}")
            logger.info(f"  Portfolio value: ${portfolio_value:,.2f}")

            # Show current state
            logger.info("  Current allocations:")
            for symbol in symbols:
                drift = current_weights[symbol] - target_weight
                drift_str = f"+{drift:.1%}" if drift > 0 else f"{drift:.1%}"
                logger.info(f"    {symbol:5}: {current_weights[symbol]:.1%} (target: {target_weight:.1%}, drift: {drift_str})")

            # Calculate trades needed
            logger.info("  Trades:")
            day_trades = []
            for symbol in symbols:
                target_value = portfolio_value * target_weight
                current_value = current_values[symbol]
                diff = target_value - current_value
                price = day_prices[symbol]
                shares_diff = int(diff / price)

                if shares_diff != 0:
                    action = "BUY" if shares_diff > 0 else "SELL"
                    shares = abs(shares_diff)
                    trade_value = shares * price

                    if action == "BUY":
                        holdings[symbol] += shares
                        cash -= trade_value
                    else:
                        holdings[symbol] -= shares
                        cash += trade_value

                    day_trades.append({
                        "date": date,
                        "symbol": symbol,
                        "action": action,
                        "shares": shares,
                        "price": price,
                        "value": trade_value,
                        "reason": f"Rebalance ({drift_str} drift)",
                    })
                    logger.info(f"    {action:4} {shares:4} {symbol:5} @ ${price:>8.2f} = ${trade_value:>10,.2f}")

            trades.extend(day_trades)

            if not day_trades:
                logger.info("    (no trades - drift too small for whole shares)")

    # Final summary
    final_prices = prices.iloc[-1]
    final_value = cash
    for symbol in symbols:
        final_value += holdings[symbol] * final_prices[symbol]

    total_return = (final_value - initial_capital) / initial_capital

    logger.info(f"\n{'='*60}")
    logger.info("SIMULATION COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Period: {prices.index[0]} to {prices.index[-1]} ({len(prices)} days)")
    logger.info(f"Initial capital: ${initial_capital:,.2f}")
    logger.info(f"Final value: ${final_value:,.2f}")
    logger.info(f"Total return: {total_return:.2%}")
    logger.info(f"Total trades: {len(trades)}")

    # Final holdings
    logger.info("\nFinal holdings:")
    for symbol in symbols:
        shares = holdings[symbol]
        price = final_prices[symbol]
        value = shares * price
        weight = value / final_value
        logger.info(f"  {symbol:5}: {shares:4} shares @ ${price:>8.2f} = ${value:>10,.2f} ({weight:.1%})")
    logger.info(f"  Cash: ${cash:,.2f}")

    return pd.DataFrame(trades)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate historical trading")
    parser.add_argument(
        "--universe", "-u",
        default="test",
        help=f"Universe name ({', '.join(UNIVERSES.keys())}) or comma-separated symbols",
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=180,
        help="Days of history to simulate (default: 180)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.02,
        help="Rebalance threshold as decimal (default: 0.02 = 2%%)",
    )
    parser.add_argument(
        "--capital", "-c",
        type=float,
        default=100_000,
        help="Initial capital (default: 100000)",
    )
    args = parser.parse_args()

    # Resolve universe
    if args.universe in UNIVERSES:
        universe = UNIVERSES[args.universe]
    else:
        universe = [s.strip() for s in args.universe.split(",")]

    # Fetch data
    data = fetch_historical_data(universe, days=args.days)

    if not data:
        logger.error("No data fetched")
        exit(1)

    # Run simulation
    trades_df = simulate_trading(
        data,
        initial_capital=args.capital,
        rebalance_threshold=args.threshold,
    )
