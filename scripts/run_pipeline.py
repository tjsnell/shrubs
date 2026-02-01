#!/usr/bin/env python3
"""Full pipeline runner - fetch, backtest, execute."""
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from shrubs.config import settings
from shrubs.data.store import DataStore
from shrubs.data.ib_client import IBClient, ContractType
from shrubs.backtest.engine import BacktestEngine
from shrubs.execution.portfolio import Portfolio
from shrubs.execution.orders import OrderManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_pipeline(
    universe: list[str],
    strategy_name: str = "equal_weight",
    dry_run: bool = True,
) -> bool:
    """Run the full trading pipeline.

    Args:
        universe: List of symbols to trade
        strategy_name: Strategy to use
        dry_run: If True, don't submit orders

    Returns:
        True if successful
    """
    logger.info("=" * 50)
    logger.info(f"Pipeline started at {datetime.now()}")
    logger.info(f"Universe: {len(universe)} symbols")
    logger.info(f"Strategy: {strategy_name}")
    logger.info(f"Dry run: {dry_run}")
    logger.info("=" * 50)

    # Step 1: Fetch data
    logger.info("Step 1: Fetching data...")
    store = DataStore(settings.data_dir)

    try:
        with IBClient(connect=True) as client:
            for symbol in universe:
                logger.info(f"  Fetching {symbol}...")
                contract = client.create_contract(symbol, ContractType.STOCK)
                df = client.fetch_historical(contract, duration="30 D", bar_size="1 day")
                if len(df) > 0:
                    store.save_ohlcv(symbol, df, asset_type="equities", timeframe="daily")
                    logger.info(f"    Saved {len(df)} bars")
                else:
                    logger.warning(f"    No data for {symbol}")
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        return False

    logger.info("Data fetch complete.")

    # Step 2: Run backtest
    logger.info("Step 2: Running backtest...")

    # Load data from store
    data = {}
    for symbol in universe:
        df = store.load_ohlcv(symbol, asset_type="equities", timeframe="daily")
        if df is not None and len(df) > 0:
            data[symbol] = df

    if not data:
        logger.error("No data available for backtest")
        return False

    logger.info(f"  Loaded data for {len(data)} symbols")

    # Define strategy
    def equal_weight_strategy(data: dict) -> pd.DataFrame:
        symbols = list(data.keys())
        weight = 1.0 / len(symbols)
        return pd.DataFrame({
            "symbol": symbols,
            "weight": [weight] * len(symbols),
        })

    engine = BacktestEngine()
    result = engine.run(data, equal_weight_strategy)

    logger.info(f"  Sharpe: {result.sharpe_ratio:.2f}")
    logger.info(f"  Max Drawdown: {result.max_drawdown:.2%}")
    logger.info(f"  Total Return: {result.total_return:.2%}")
    logger.info("Backtest complete.")

    # Step 3: Execute orders
    logger.info("Step 3: Executing orders...")

    # Get current prices (last close from data)
    prices = {sym: df["close"].iloc[-1] for sym, df in data.items()}

    # Compute trades from empty portfolio
    portfolio = Portfolio(account_value=100_000)
    trades = portfolio.compute_trades(result.target_positions, prices)

    logger.info(f"  Generated {len(trades)} trades:")
    for trade in trades:
        logger.info(f"    {trade['action']} {trade['shares']} {trade['symbol']} @ ${trade['price']:.2f}")

    if dry_run:
        logger.info("[DRY RUN] Orders logged but not submitted.")
    else:
        # Submit orders via IB
        order_mgr = OrderManager(ib_client=None, dry_run=False)
        for trade in trades:
            order = order_mgr.create_order(trade["symbol"], trade["action"], trade["shares"])
            order_mgr.submit(order)

    logger.info("Execution complete.")

    logger.info("=" * 50)
    logger.info(f"Pipeline finished at {datetime.now()}")
    logger.info("=" * 50)

    return True


if __name__ == "__main__":
    # Default universe for testing
    default_universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

    success = run_pipeline(
        universe=default_universe,
        strategy_name="equal_weight",
        dry_run=True,
    )

    sys.exit(0 if success else 1)
