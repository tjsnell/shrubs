# Shrubs Trading System

## Overview

Shrubs is a **portfolio rebalancing system** that maintains target allocations across a universe of stocks. It runs as a nightly batch process after market close.

## Trading Philosophy

This is **not** a timing-based system. The edge comes from systematic factor exposure (currently equal-weight, with multi-factor strategies planned), not from predicting price movements or timing entries.

Key principles:
- **Quantitative portfolio management** - systematic, rules-based allocation
- **Not time-sensitive** - overnight market orders are acceptable
- **Rebalancing, not trading** - we adjust to targets, not chase signals

## How It Works

### The Nightly Pipeline

```
Market Close (4pm ET)
        ↓
[1] Fetch latest prices from IB
        ↓
[2] Run backtest/strategy to compute target weights
        ↓
[3] Compare current portfolio vs targets
        ↓
[4] Generate rebalancing trades
        ↓
[5] Submit orders (execute at next open)
```

### Entry Conditions

A **BUY** order is generated when:
1. A symbol is in the target universe but not in the current portfolio
2. A symbol's current allocation is below its target weight (needs more shares)

There is no "signal" for entry. If a stock is in your universe with a non-zero target weight, you own it.

### Exit Conditions

A **SELL** order is generated when:
1. A symbol is in the portfolio but no longer in the target universe (full liquidation)
2. A symbol's current allocation exceeds its target weight (trim position)

There is no stop-loss or take-profit. Position sizing is purely weight-based.

### Trade Frequency

- **Pipeline runs**: Once daily, after market close
- **Orders execute**: At next market open (market orders)
- **Rebalancing trades**: Only when allocations drift from targets

In practice, with stable prices and no universe changes, most days generate zero trades. Trades occur when:
- Prices move enough to cause allocation drift
- Universe changes (add/remove symbols)
- Strategy weights change (for non-equal-weight strategies)

## Current Strategy: Equal Weight

The default strategy allocates equally across all symbols:

```
weight = 1.0 / number_of_symbols
```

For example:
- 5 symbols → 20% each
- 7 symbols → ~14.3% each
- 3 symbols → ~33.3% each

This is the simplest systematic strategy. It:
- Automatically sells winners (when they grow above target)
- Automatically buys losers (when they fall below target)
- Maintains diversification mechanically

## Order Execution

### Order Types

Currently: **Market orders only**

Placed overnight, executed at open. This is intentional - for a portfolio strategy, the exact fill price matters less than maintaining correct allocations.

### Order Sizing

Uses the `order_target_percent` approach:

```python
target_value = account_value * target_weight
current_value = shares_held * current_price
difference = target_value - current_value
shares_to_trade = difference / current_price
```

Shares are rounded down (no fractional shares).

### Execution Sequence

1. **Liquidate unwanted positions first** - frees capital
2. **Reduce oversized positions** - more capital
3. **Increase undersized positions** - deploy capital
4. **Open new positions** - remaining capital

## Safety Features

- **Dry run mode** (default) - logs orders without submitting
- **Paper trading** - uses IB paper account for testing
- **Max position limits** - (planned) cap individual positions
- **Max turnover limits** - (planned) limit daily trading volume
- **Kill switch** - (planned) halt all execution on anomalies

## Example Scenario

**Day 1: Initial Portfolio**
- Universe: AAPL, NVDA, AMD (equal weight = 33.3% each)
- Account: $100,000
- Action: BUY ~$33,333 of each

**Day 5: Prices Changed**
- NVDA up 10%, now 36% of portfolio
- AAPL flat, now 32% of portfolio
- AMD down 5%, now 32% of portfolio
- Action: SELL some NVDA, BUY some AMD (rebalance to 33.3% each)

**Day 10: Universe Change**
- Remove AMD, add MSFT
- Action: SELL all AMD, BUY MSFT, rebalance others to 33.3%

## Planned Enhancements

1. **Multi-factor strategies** - momentum, value, quality signals
2. **Risk-adjusted weighting** - inverse volatility, risk parity
3. **Sector constraints** - max exposure per sector
4. **Limit orders** - better fills for larger positions
5. **Intraday rebalancing** - for more volatile strategies

## Running the System

```bash
# Dry run (default) - see what trades would execute
python scripts/run_pipeline.py -u mag7

# With custom symbols
python scripts/run_pipeline.py -u AAPL,NVDA,AMD,MSFT

# Live execution (paper account)
python scripts/run_pipeline.py -u faang --live
```

## Key Files

- `scripts/run_pipeline.py` - Main entry point
- `src/shrubs/backtest/engine.py` - Strategy execution
- `src/shrubs/execution/portfolio.py` - Position tracking, trade computation
- `src/shrubs/execution/orders.py` - Order creation and submission
- `src/shrubs/data/ib_client.py` - Interactive Brokers connection
