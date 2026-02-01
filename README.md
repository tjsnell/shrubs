# Shrubs

End-to-end algorithmic trading pipeline in Python.

## Overview

Automated nightly pipeline for quantitative portfolio management:

1. **Data Ingestion** - Download market data after close
2. **Database Staging** - Prepare data for analysis
3. **Data Bundling** - Package for backtest engine
4. **Backtesting** - Run strategy, generate signals
5. **Order Execution** - Stage and execute trades via broker API

## Strategy Philosophy

- Quantitative factor-based approach (momentum, etc.)
- Not time-sensitive - overnight order staging is acceptable
- Edge comes from systematic factor exposure, not intraday timing

## Architecture

```
[Market Data] → [Staging DB] → [Bundles] → [Backtest Engine]
                                                   ↓
[Broker API] ← [Order Execution] ← [Signals DataFrame]
```

## Status

🚧 In development
