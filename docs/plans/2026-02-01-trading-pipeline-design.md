# Algorithmic Trading Pipeline Design

**Date**: 2026-02-01
**Status**: Approved

## Overview

End-to-end algorithmic trading pipeline for quantitative portfolio management. Modular architecture with independent stages connected via files.

**Goals**:
- Learning and experimentation first
- Paper trading with path to live trading
- Strategy-agnostic infrastructure supporting multi-factor strategies later

**Constraints**:
- Multi-asset universe: stocks, ETFs, futures, options
- Manual operation during development, scheduled nightly once stable

## Architecture

```
[IB/Polygon] → [Data Store] → [Backtest Engine] → [Signals File]
                                                        ↓
                              [IB Paper/Live] ← [Execution Engine]
```

### Approach: Modular Pipeline

Separate components connected via files. Each stage is its own script:
- `data_fetch` → writes to Parquet
- `backtest` → reads data, writes signals
- `execute` → reads signals, places orders via IB

Easy to debug, test pieces independently, swap components. Future consideration: add workflow orchestration (Prefect/Dagster) once pipeline is stable (see `sh-ug1`).

## Project Structure

```
shrubs/
├── src/
│   └── shrubs/
│       ├── __init__.py
│       ├── config.py          # Settings, API keys, paths
│       ├── data/
│       │   ├── __init__.py
│       │   ├── ib_client.py   # IB TWS/Gateway connection
│       │   ├── polygon.py     # Polygon.io historical data
│       │   └── store.py       # Data persistence (Parquet)
│       ├── backtest/
│       │   ├── __init__.py
│       │   ├── engine.py      # VectorBT wrapper or custom
│       │   └── signals.py     # Signal generation interface
│       ├── execution/
│       │   ├── __init__.py
│       │   ├── orders.py      # Order management
│       │   └── portfolio.py   # Position tracking, reconciliation
│       └── cli.py             # Command-line interface
├── scripts/
│   ├── run_pipeline.py        # Full pipeline runner
│   └── cron_nightly.sh        # Cron wrapper script
├── data/                      # Local data storage (gitignored)
├── tests/
└── pyproject.toml
```

## Data Layer

### IB Client (`ib_client.py`)

Uses `ib_insync` library for TWS API. Handles:
- Connection to TWS or IB Gateway (paper or live)
- Historical data requests (with rate limiting)
- Real-time quotes when needed
- Contract lookups for stocks, ETFs, futures, options

### Polygon Fallback (`polygon.py`)

When IB historical data is insufficient (gaps, depth limits):
- Daily/minute bars for equities
- Options chains
- Corporate actions (splits, dividends)

### Data Store (`store.py`)

Parquet files organized by asset type and timeframe:
```
data/
├── equities/
│   ├── daily/AAPL.parquet
│   └── minute/AAPL.parquet
├── futures/
│   └── daily/ES.parquet
└── options/
    └── chains/AAPL_2024-03.parquet
```

Parquet over SQLite - columnar format is faster for time-series analysis and VectorBT reads it natively.

### Config (`config.py`)

Environment-based settings via `pydantic-settings`:
- IB connection (host, port, client ID, paper vs live)
- Polygon API key
- Data paths
- Universe definitions (watchlists)

## Backtest Layer

### Engine (`engine.py`)

Thin wrapper around VectorBT:
- Load data from Parquet store
- Accept a strategy function (signals in, weights out)
- Run vectorized backtest
- Output performance metrics + target positions DataFrame

Strategy interface - simple function signature:
```python
def strategy(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Returns DataFrame with columns: symbol, weight"""
    ...
```

Strategies are functions, not classes. Easy to test, easy to swap.

### Signals (`signals.py`)

Library of reusable signal generators for multi-factor:
- Momentum (returns over N days)
- Mean reversion (distance from moving average)
- Volatility (rolling std dev)
- Custom signals plug in here

Each signal returns a DataFrame. Combine with simple math:
```python
final_signal = 0.5 * momentum + 0.3 * value + 0.2 * quality
```

### Output Format

Backtest produces a parquet file with:
- Target positions (symbol, target_weight, current_price)
- Performance stats (sharpe, drawdown, etc.)
- Trade log for analysis

This file is the handoff to the execution layer.

## Execution Layer

### Portfolio (`portfolio.py`)

Tracks current state by querying IB:
- Current positions (symbol, quantity, avg cost)
- Account value, buying power, margin
- Reconciles local state with broker state

Compares current vs target positions to compute required trades.

### Orders (`orders.py`)

Handles order creation and submission:
- `order_target_percent(symbol, target_pct)` - key method
- Calculates shares needed to reach target allocation
- Supports order types: market, limit, MOC (market-on-close)
- Tracks order status (submitted, filled, rejected)

### Execution Flow

1. Load target positions from backtest output
2. Fetch current positions from IB
3. Compute diff: what to buy, sell, liquidate
4. Liquidate unwanted positions first (frees capital)
5. Submit new/adjusted positions
6. Log all trades to audit file

### Safety Features

- Dry-run mode (logs orders without submitting)
- Max position size limits
- Max daily turnover limit
- Kill switch - stop all execution if something looks wrong

## CLI & Pipeline Runner

### CLI (`cli.py`)

Using `typer` for clean command structure:

```bash
# Individual stages
shrubs data fetch --universe sp500 --days 30
shrubs data fetch --symbol AAPL --source polygon
shrubs backtest run --strategy momentum --output signals.parquet
shrubs execute --input signals.parquet --dry-run
shrubs execute --input signals.parquet  # real orders

# Utilities
shrubs portfolio show          # current positions
shrubs orders status           # pending orders
shrubs data status             # data freshness report
```

### Pipeline Runner (`run_pipeline.py`)

Chains all stages with error handling:

```python
def run_nightly():
    fetch_result = fetch_data(universe, lookback_days)
    if not fetch_result.success:
        alert("Data fetch failed"); return

    backtest_result = run_backtest(strategy)
    if not backtest_result.success:
        alert("Backtest failed"); return

    execute_result = execute_orders(backtest_result.signals)
    log_run(fetch_result, backtest_result, execute_result)
```

### Cron Setup

```bash
# Run at 5pm ET (after market close)
0 17 * * 1-5 /path/to/scripts/cron_nightly.sh >> /var/log/shrubs.log 2>&1
```

## Testing Strategy

```
tests/
├── unit/
│   ├── test_signals.py      # Signal calculations
│   ├── test_portfolio.py    # Position diff logic
│   └── test_orders.py       # Order size calculations
├── integration/
│   ├── test_ib_client.py    # Against paper account
│   └── test_data_store.py   # Read/write roundtrips
└── conftest.py              # Fixtures, sample data
```

Unit tests mock IB/Polygon. Integration tests hit real paper account (marked slow, run separately).

## Development Workflow

Build order:
1. **Data layer first** - Get IB connection working, fetch some data, store it
2. **Backtest second** - Load stored data, run a dummy strategy, output signals
3. **Execution last** - Read signals, place paper trades
4. **Wire together** - CLI commands, then full pipeline runner

Each stage is usable standalone before wiring up.

## Dependencies

```
ib_insync           # IB API wrapper
polygon-api-client  # Polygon.io
vectorbt            # Backtesting
pandas              # Data manipulation
numpy               # Numerical operations
pyarrow             # Parquet I/O
pydantic-settings   # Config management
typer               # CLI
pytest              # Testing
```

## Future Considerations

- Workflow orchestration (Prefect/Dagster) - tracked in `sh-ug1`
- Multi-factor strategy library
- Performance dashboard / visualization
- Alert system (email/Slack on failures or unusual activity)
