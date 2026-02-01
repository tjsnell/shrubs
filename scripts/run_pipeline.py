#!/usr/bin/env python3
"""Full pipeline runner - fetch, backtest, execute."""
import logging
import sys
from datetime import datetime
from pathlib import Path

from shrubs.config import settings
from shrubs.data.store import DataStore
from shrubs.backtest.engine import BacktestEngine

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
    # TODO: Implement data fetching
    logger.info("Data fetch complete.")

    # Step 2: Run backtest
    logger.info("Step 2: Running backtest...")
    engine = BacktestEngine()
    # TODO: Load data and run backtest
    logger.info("Backtest complete.")

    # Step 3: Execute orders
    logger.info("Step 3: Executing orders...")
    if dry_run:
        logger.info("[DRY RUN] Orders logged but not submitted.")
    # TODO: Execute orders
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
