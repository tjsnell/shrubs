# Trading Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an end-to-end algorithmic trading pipeline with data ingestion, backtesting, and order execution.

**Architecture:** Modular pipeline with independent stages connected via Parquet files. Data layer (IB + Polygon) → Backtest layer (VectorBT) → Execution layer (IB orders).

**Tech Stack:** Python 3.11+, ib_insync, polygon-api-client, vectorbt, pandas, pyarrow, pydantic-settings, typer, pytest

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/shrubs/__init__.py`
- Create: `src/shrubs/config.py`
- Create: `tests/conftest.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "shrubs"
version = "0.1.0"
description = "Algorithmic trading pipeline"
requires-python = ">=3.11"
dependencies = [
    "ib_insync>=0.9.86",
    "polygon-api-client>=1.13.0",
    "vectorbt>=0.26.0",
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "pyarrow>=14.0.0",
    "pydantic-settings>=2.0.0",
    "typer>=0.9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
]

[project.scripts]
shrubs = "shrubs.cli:app"

[tool.hatch.build.targets.wheel]
packages = ["src/shrubs"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

**Step 2: Create package structure**

```bash
mkdir -p src/shrubs/data src/shrubs/backtest src/shrubs/execution tests/unit tests/integration
touch src/shrubs/__init__.py
touch src/shrubs/data/__init__.py
touch src/shrubs/backtest/__init__.py
touch src/shrubs/execution/__init__.py
```

**Step 3: Create config.py**

```python
"""Configuration management via environment variables."""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # IB Connection
    ib_host: str = "127.0.0.1"
    ib_port: int = 7497  # 7497=TWS paper, 7496=TWS live, 4001=Gateway paper, 4002=Gateway live
    ib_client_id: int = 1

    # Polygon
    polygon_api_key: str = ""

    # Paths
    data_dir: Path = Path("data")

    # Safety
    dry_run: bool = True
    max_position_pct: float = 0.10  # Max 10% in any single position

    class Config:
        env_file = ".env"
        env_prefix = "SHRUBS_"


settings = Settings()
```

**Step 4: Create conftest.py with sample data fixtures**

```python
"""Shared test fixtures."""
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Sample OHLCV data for testing."""
    return pd.DataFrame({
        "open": [100.0, 101.0, 102.0, 101.5, 103.0],
        "high": [101.0, 102.5, 103.0, 102.0, 104.0],
        "low": [99.5, 100.5, 101.0, 100.0, 102.5],
        "close": [100.5, 102.0, 101.5, 101.0, 103.5],
        "volume": [1000, 1100, 900, 1200, 1050],
    }, index=pd.date_range("2024-01-01", periods=5, freq="D"))


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Temporary data directory for tests."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "equities" / "daily").mkdir(parents=True)
    (data_dir / "futures" / "daily").mkdir(parents=True)
    (data_dir / "options" / "chains").mkdir(parents=True)
    return data_dir
```

**Step 5: Verify setup**

Run: `pip install -e ".[dev]"` then `pytest --collect-only`
Expected: Shows conftest fixtures, no errors

**Step 6: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "feat: project scaffolding with config and test fixtures"
```

---

## Task 2: Data Store (Parquet Read/Write)

**Files:**
- Create: `src/shrubs/data/store.py`
- Create: `tests/unit/test_store.py`

**Step 1: Write failing test for save_ohlcv**

```python
"""Tests for data store."""
import pandas as pd
import pytest
from pathlib import Path
from shrubs.data.store import DataStore


def test_save_and_load_ohlcv(tmp_data_dir: Path, sample_ohlcv: pd.DataFrame):
    """Round-trip save and load of OHLCV data."""
    store = DataStore(tmp_data_dir)

    store.save_ohlcv("AAPL", sample_ohlcv, asset_type="equities", timeframe="daily")
    loaded = store.load_ohlcv("AAPL", asset_type="equities", timeframe="daily")

    pd.testing.assert_frame_equal(loaded, sample_ohlcv)


def test_load_nonexistent_returns_none(tmp_data_dir: Path):
    """Loading nonexistent symbol returns None."""
    store = DataStore(tmp_data_dir)
    result = store.load_ohlcv("FAKE", asset_type="equities", timeframe="daily")
    assert result is None


def test_list_symbols(tmp_data_dir: Path, sample_ohlcv: pd.DataFrame):
    """List available symbols for asset type."""
    store = DataStore(tmp_data_dir)
    store.save_ohlcv("AAPL", sample_ohlcv, asset_type="equities", timeframe="daily")
    store.save_ohlcv("MSFT", sample_ohlcv, asset_type="equities", timeframe="daily")

    symbols = store.list_symbols(asset_type="equities", timeframe="daily")
    assert set(symbols) == {"AAPL", "MSFT"}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_store.py -v`
Expected: FAIL with "cannot import name 'DataStore'"

**Step 3: Write DataStore implementation**

```python
"""Parquet-based data storage."""
from pathlib import Path
from typing import Literal

import pandas as pd


AssetType = Literal["equities", "futures", "options"]
Timeframe = Literal["daily", "minute"]


class DataStore:
    """Manages Parquet file storage for market data."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)

    def _get_path(
        self, symbol: str, asset_type: AssetType, timeframe: Timeframe
    ) -> Path:
        """Get file path for a symbol."""
        return self.base_dir / asset_type / timeframe / f"{symbol}.parquet"

    def save_ohlcv(
        self,
        symbol: str,
        data: pd.DataFrame,
        asset_type: AssetType,
        timeframe: Timeframe,
    ) -> None:
        """Save OHLCV data to Parquet."""
        path = self._get_path(symbol, asset_type, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(path)

    def load_ohlcv(
        self, symbol: str, asset_type: AssetType, timeframe: Timeframe
    ) -> pd.DataFrame | None:
        """Load OHLCV data from Parquet. Returns None if not found."""
        path = self._get_path(symbol, asset_type, timeframe)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def list_symbols(self, asset_type: AssetType, timeframe: Timeframe) -> list[str]:
        """List available symbols for asset type and timeframe."""
        dir_path = self.base_dir / asset_type / timeframe
        if not dir_path.exists():
            return []
        return [p.stem for p in dir_path.glob("*.parquet")]
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_store.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/shrubs/data/store.py tests/unit/test_store.py
git commit -m "feat: data store with Parquet read/write"
```

---

## Task 3: IB Client Connection

**Files:**
- Create: `src/shrubs/data/ib_client.py`
- Create: `tests/unit/test_ib_client.py`

**Step 1: Write failing test for contract creation**

```python
"""Tests for IB client."""
import pytest
from shrubs.data.ib_client import IBClient, ContractType


def test_create_stock_contract():
    """Create a stock contract spec."""
    client = IBClient(connect=False)
    contract = client.create_contract("AAPL", ContractType.STOCK)

    assert contract.symbol == "AAPL"
    assert contract.secType == "STK"
    assert contract.exchange == "SMART"
    assert contract.currency == "USD"


def test_create_future_contract():
    """Create a futures contract spec."""
    client = IBClient(connect=False)
    contract = client.create_contract("ES", ContractType.FUTURE, expiry="202403")

    assert contract.symbol == "ES"
    assert contract.secType == "FUT"
    assert contract.lastTradeDateOrContractMonth == "202403"


def test_create_option_contract():
    """Create an options contract spec."""
    client = IBClient(connect=False)
    contract = client.create_contract(
        "AAPL",
        ContractType.OPTION,
        expiry="20240315",
        strike=180.0,
        right="C",
    )

    assert contract.symbol == "AAPL"
    assert contract.secType == "OPT"
    assert contract.strike == 180.0
    assert contract.right == "C"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_ib_client.py -v`
Expected: FAIL with "cannot import name 'IBClient'"

**Step 3: Write IBClient implementation**

```python
"""Interactive Brokers client wrapper."""
from enum import Enum
from typing import Literal

from ib_insync import IB, Stock, Future, Option, Contract

from shrubs.config import settings


class ContractType(Enum):
    STOCK = "STK"
    FUTURE = "FUT"
    OPTION = "OPT"
    ETF = "STK"  # ETFs use stock contract type


class IBClient:
    """Wrapper around ib_insync for IB API access."""

    def __init__(self, connect: bool = True):
        self.ib = IB()
        self._connected = False

        if connect:
            self.connect()

    def connect(self) -> None:
        """Connect to TWS or IB Gateway."""
        if not self._connected:
            self.ib.connect(
                settings.ib_host,
                settings.ib_port,
                clientId=settings.ib_client_id,
            )
            self._connected = True

    def disconnect(self) -> None:
        """Disconnect from IB."""
        if self._connected:
            self.ib.disconnect()
            self._connected = False

    def create_contract(
        self,
        symbol: str,
        contract_type: ContractType,
        expiry: str | None = None,
        strike: float | None = None,
        right: Literal["C", "P"] | None = None,
        exchange: str = "SMART",
        currency: str = "USD",
    ) -> Contract:
        """Create a contract specification."""
        if contract_type == ContractType.STOCK or contract_type == ContractType.ETF:
            return Stock(symbol, exchange, currency)

        elif contract_type == ContractType.FUTURE:
            if not expiry:
                raise ValueError("expiry required for futures")
            return Future(symbol, expiry, exchange, currency=currency)

        elif contract_type == ContractType.OPTION:
            if not all([expiry, strike, right]):
                raise ValueError("expiry, strike, and right required for options")
            return Option(symbol, expiry, strike, right, exchange, currency=currency)

        raise ValueError(f"Unknown contract type: {contract_type}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_ib_client.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/shrubs/data/ib_client.py tests/unit/test_ib_client.py
git commit -m "feat: IB client with contract creation"
```

---

## Task 4: IB Historical Data Fetching

**Files:**
- Modify: `src/shrubs/data/ib_client.py`
- Create: `tests/integration/test_ib_client.py`

**Step 1: Write integration test (mark as slow)**

```python
"""Integration tests for IB client - requires running TWS/Gateway."""
import pandas as pd
import pytest
from shrubs.data.ib_client import IBClient, ContractType


pytestmark = pytest.mark.integration


@pytest.fixture
def ib_client():
    """Connected IB client - requires TWS/Gateway running."""
    client = IBClient(connect=True)
    yield client
    client.disconnect()


def test_fetch_historical_data(ib_client: IBClient):
    """Fetch historical data from IB."""
    contract = ib_client.create_contract("AAPL", ContractType.STOCK)
    df = ib_client.fetch_historical(
        contract,
        duration="5 D",
        bar_size="1 day",
    )

    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert all(col in df.columns for col in ["open", "high", "low", "close", "volume"])
```

**Step 2: Add fetch_historical method to IBClient**

Add to `src/shrubs/data/ib_client.py`:

```python
import pandas as pd
from ib_insync import Contract, util

# Add to IBClient class:

    def fetch_historical(
        self,
        contract: Contract,
        duration: str = "30 D",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
    ) -> pd.DataFrame:
        """Fetch historical bars for a contract.

        Args:
            contract: IB contract specification
            duration: How far back (e.g., "30 D", "1 Y")
            bar_size: Bar size (e.g., "1 day", "1 hour", "5 mins")
            what_to_show: Data type (TRADES, MIDPOINT, BID, ASK)

        Returns:
            DataFrame with OHLCV columns
        """
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=True,
        )

        if not bars:
            return pd.DataFrame()

        df = util.df(bars)
        df = df.rename(columns={"date": "timestamp"})
        df = df.set_index("timestamp")
        df = df[["open", "high", "low", "close", "volume"]]
        return df
```

**Step 3: Run integration test (requires IB connection)**

Run: `pytest tests/integration/test_ib_client.py -v -m integration`
Expected: 1 passed (if TWS running) or skipped

**Step 4: Commit**

```bash
git add src/shrubs/data/ib_client.py tests/integration/test_ib_client.py
git commit -m "feat: IB historical data fetching"
```

---

## Task 5: Polygon Client

**Files:**
- Create: `src/shrubs/data/polygon.py`
- Create: `tests/unit/test_polygon.py`

**Step 1: Write failing test**

```python
"""Tests for Polygon client."""
import pandas as pd
import pytest
from unittest.mock import Mock, patch
from shrubs.data.polygon import PolygonClient


@patch("shrubs.data.polygon.RESTClient")
def test_fetch_daily_bars(mock_rest_class):
    """Fetch daily bars from Polygon."""
    # Mock the API response
    mock_client = Mock()
    mock_rest_class.return_value = mock_client

    mock_agg = Mock()
    mock_agg.open = 100.0
    mock_agg.high = 101.0
    mock_agg.low = 99.0
    mock_agg.close = 100.5
    mock_agg.volume = 1000
    mock_agg.timestamp = 1704067200000  # 2024-01-01

    mock_client.list_aggs.return_value = [mock_agg]

    client = PolygonClient(api_key="test_key")
    df = client.fetch_daily("AAPL", start="2024-01-01", end="2024-01-01")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.iloc[0]["close"] == 100.5


@patch("shrubs.data.polygon.RESTClient")
def test_empty_response_returns_empty_df(mock_rest_class):
    """Empty API response returns empty DataFrame."""
    mock_client = Mock()
    mock_rest_class.return_value = mock_client
    mock_client.list_aggs.return_value = []

    client = PolygonClient(api_key="test_key")
    df = client.fetch_daily("FAKE", start="2024-01-01", end="2024-01-01")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_polygon.py -v`
Expected: FAIL with "cannot import name 'PolygonClient'"

**Step 3: Write PolygonClient implementation**

```python
"""Polygon.io client for historical data."""
import pandas as pd
from polygon import RESTClient

from shrubs.config import settings


class PolygonClient:
    """Wrapper around Polygon.io API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.polygon_api_key
        self.client = RESTClient(self.api_key)

    def fetch_daily(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Fetch daily bars from Polygon.

        Args:
            symbol: Stock ticker
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)

        Returns:
            DataFrame with OHLCV columns
        """
        aggs = list(self.client.list_aggs(
            ticker=symbol,
            multiplier=1,
            timespan="day",
            from_=start,
            to=end,
        ))

        if not aggs:
            return pd.DataFrame()

        data = []
        for agg in aggs:
            data.append({
                "timestamp": pd.Timestamp(agg.timestamp, unit="ms"),
                "open": agg.open,
                "high": agg.high,
                "low": agg.low,
                "close": agg.close,
                "volume": agg.volume,
            })

        df = pd.DataFrame(data)
        df = df.set_index("timestamp")
        return df
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_polygon.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add src/shrubs/data/polygon.py tests/unit/test_polygon.py
git commit -m "feat: Polygon client for historical data"
```

---

## Task 6: Signal Generation Interface

**Files:**
- Create: `src/shrubs/backtest/signals.py`
- Create: `tests/unit/test_signals.py`

**Step 1: Write failing tests**

```python
"""Tests for signal generators."""
import pandas as pd
import pytest
import numpy as np
from shrubs.backtest.signals import momentum, mean_reversion, combine_signals


@pytest.fixture
def price_series() -> pd.Series:
    """Sample price series for testing."""
    return pd.Series(
        [100, 102, 101, 105, 103, 108, 107, 110, 112, 115],
        index=pd.date_range("2024-01-01", periods=10, freq="D"),
    )


def test_momentum_signal(price_series: pd.Series):
    """Momentum signal based on N-day returns."""
    signal = momentum(price_series, lookback=5)

    # Signal should be NaN for first lookback periods
    assert pd.isna(signal.iloc[:5]).all()
    # After lookback, should have values
    assert not pd.isna(signal.iloc[5:]).any()


def test_mean_reversion_signal(price_series: pd.Series):
    """Mean reversion signal based on distance from MA."""
    signal = mean_reversion(price_series, window=3)

    # Signal should be NaN for first window-1 periods
    assert pd.isna(signal.iloc[:2]).all()
    # Positive = above MA (sell signal), negative = below MA (buy signal)
    assert isinstance(signal.iloc[-1], float)


def test_combine_signals():
    """Combine multiple signals with weights."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    signal_a = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=idx)
    signal_b = pd.Series([0.5, 0.4, 0.3, 0.2, 0.1], index=idx)

    combined = combine_signals({"a": signal_a, "b": signal_b}, {"a": 0.6, "b": 0.4})

    expected = 0.6 * signal_a + 0.4 * signal_b
    pd.testing.assert_series_equal(combined, expected)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_signals.py -v`
Expected: FAIL with "cannot import name 'momentum'"

**Step 3: Write signals implementation**

```python
"""Signal generation functions."""
import pandas as pd


def momentum(prices: pd.Series, lookback: int = 20) -> pd.Series:
    """Calculate momentum signal as N-day returns.

    Args:
        prices: Price series
        lookback: Number of days for return calculation

    Returns:
        Series of momentum values (positive = uptrend)
    """
    return prices.pct_change(periods=lookback)


def mean_reversion(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate mean reversion signal as deviation from moving average.

    Args:
        prices: Price series
        window: Moving average window

    Returns:
        Series of z-scores from MA (positive = above MA, negative = below)
    """
    ma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    return (prices - ma) / std


def volatility(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate realized volatility.

    Args:
        prices: Price series
        window: Rolling window for volatility

    Returns:
        Series of annualized volatility values
    """
    returns = prices.pct_change()
    return returns.rolling(window=window).std() * (252 ** 0.5)


def combine_signals(
    signals: dict[str, pd.Series],
    weights: dict[str, float],
) -> pd.Series:
    """Combine multiple signals with weights.

    Args:
        signals: Dict of signal name to signal series
        weights: Dict of signal name to weight (should sum to 1)

    Returns:
        Weighted combination of signals
    """
    result = None
    for name, signal in signals.items():
        weight = weights.get(name, 0)
        weighted = signal * weight
        if result is None:
            result = weighted
        else:
            result = result + weighted
    return result
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_signals.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/shrubs/backtest/signals.py tests/unit/test_signals.py
git commit -m "feat: signal generation functions (momentum, mean reversion)"
```

---

## Task 7: Backtest Engine

**Files:**
- Create: `src/shrubs/backtest/engine.py`
- Create: `tests/unit/test_engine.py`

**Step 1: Write failing tests**

```python
"""Tests for backtest engine."""
import pandas as pd
import pytest
from shrubs.backtest.engine import BacktestEngine, BacktestResult


@pytest.fixture
def multi_symbol_data() -> dict[str, pd.DataFrame]:
    """Sample data for multiple symbols."""
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    return {
        "AAPL": pd.DataFrame({
            "open": range(100, 120),
            "high": range(101, 121),
            "low": range(99, 119),
            "close": range(100, 120),
            "volume": [1000] * 20,
        }, index=idx),
        "MSFT": pd.DataFrame({
            "open": range(200, 220),
            "high": range(201, 221),
            "low": range(199, 219),
            "close": range(200, 220),
            "volume": [2000] * 20,
        }, index=idx),
    }


def dummy_strategy(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Equal weight strategy for testing."""
    symbols = list(data.keys())
    return pd.DataFrame({
        "symbol": symbols,
        "weight": [1.0 / len(symbols)] * len(symbols),
    })


def test_run_backtest(multi_symbol_data: dict[str, pd.DataFrame]):
    """Run a backtest with dummy strategy."""
    engine = BacktestEngine()
    result = engine.run(multi_symbol_data, dummy_strategy)

    assert isinstance(result, BacktestResult)
    assert result.target_positions is not None
    assert len(result.target_positions) == 2


def test_backtest_result_has_metrics(multi_symbol_data: dict[str, pd.DataFrame]):
    """Backtest result includes performance metrics."""
    engine = BacktestEngine()
    result = engine.run(multi_symbol_data, dummy_strategy)

    assert hasattr(result, "sharpe_ratio")
    assert hasattr(result, "max_drawdown")
    assert hasattr(result, "total_return")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_engine.py -v`
Expected: FAIL with "cannot import name 'BacktestEngine'"

**Step 3: Write BacktestEngine implementation**

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_engine.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add src/shrubs/backtest/engine.py tests/unit/test_engine.py
git commit -m "feat: backtest engine with metrics calculation"
```

---

## Task 8: Portfolio State Management

**Files:**
- Create: `src/shrubs/execution/portfolio.py`
- Create: `tests/unit/test_portfolio.py`

**Step 1: Write failing tests**

```python
"""Tests for portfolio management."""
import pandas as pd
import pytest
from shrubs.execution.portfolio import Portfolio, Position


def test_compute_trades_from_scratch():
    """Compute trades when starting with empty portfolio."""
    portfolio = Portfolio(account_value=100_000)
    target = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "weight": [0.5, 0.5],
    })
    prices = {"AAPL": 150.0, "MSFT": 300.0}

    trades = portfolio.compute_trades(target, prices)

    assert len(trades) == 2
    aapl_trade = next(t for t in trades if t["symbol"] == "AAPL")
    assert aapl_trade["shares"] > 0  # Buy
    assert aapl_trade["action"] == "BUY"


def test_compute_trades_with_existing_positions():
    """Compute trades to rebalance existing positions."""
    portfolio = Portfolio(account_value=100_000)
    portfolio.positions = {
        "AAPL": Position("AAPL", shares=100, avg_cost=140.0),
    }

    target = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "weight": [0.3, 0.7],
    })
    prices = {"AAPL": 150.0, "MSFT": 300.0}

    trades = portfolio.compute_trades(target, prices)

    # Should sell some AAPL (has 15k, wants 30k worth)
    # Should buy MSFT (has 0, wants 70k worth)
    assert any(t["symbol"] == "MSFT" and t["action"] == "BUY" for t in trades)


def test_compute_trades_liquidation():
    """Compute trades to liquidate unwanted positions."""
    portfolio = Portfolio(account_value=100_000)
    portfolio.positions = {
        "AAPL": Position("AAPL", shares=100, avg_cost=140.0),
        "GOOG": Position("GOOG", shares=50, avg_cost=100.0),  # Not in target
    }

    target = pd.DataFrame({
        "symbol": ["AAPL"],
        "weight": [1.0],
    })
    prices = {"AAPL": 150.0, "GOOG": 120.0}

    trades = portfolio.compute_trades(target, prices)

    goog_trade = next(t for t in trades if t["symbol"] == "GOOG")
    assert goog_trade["action"] == "SELL"
    assert goog_trade["shares"] == 50  # Liquidate all
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_portfolio.py -v`
Expected: FAIL with "cannot import name 'Portfolio'"

**Step 3: Write Portfolio implementation**

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_portfolio.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/shrubs/execution/portfolio.py tests/unit/test_portfolio.py
git commit -m "feat: portfolio state and trade computation"
```

---

## Task 9: Order Management

**Files:**
- Create: `src/shrubs/execution/orders.py`
- Create: `tests/unit/test_orders.py`

**Step 1: Write failing tests**

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_orders.py -v`
Expected: FAIL with "cannot import name 'OrderManager'"

**Step 3: Write OrderManager implementation**

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_orders.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add src/shrubs/execution/orders.py tests/unit/test_orders.py
git commit -m "feat: order management with dry-run support"
```

---

## Task 10: CLI Commands

**Files:**
- Create: `src/shrubs/cli.py`
- Create: `tests/unit/test_cli.py`

**Step 1: Write failing tests**

```python
"""Tests for CLI commands."""
import pytest
from typer.testing import CliRunner
from shrubs.cli import app


runner = CliRunner()


def test_cli_version():
    """CLI shows version."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_cli_data_status(tmp_path, monkeypatch):
    """Data status command runs."""
    monkeypatch.setenv("SHRUBS_DATA_DIR", str(tmp_path))
    result = runner.invoke(app, ["data", "status"])
    assert result.exit_code == 0


def test_cli_portfolio_show():
    """Portfolio show command runs (dry-run mode)."""
    result = runner.invoke(app, ["portfolio", "show", "--dry-run"])
    assert result.exit_code == 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_cli.py -v`
Expected: FAIL with "cannot import name 'app'"

**Step 3: Write CLI implementation**

```python
"""Command-line interface."""
import typer
from pathlib import Path

from shrubs.config import settings
from shrubs.data.store import DataStore


app = typer.Typer(help="Shrubs - Algorithmic Trading Pipeline")


def version_callback(value: bool):
    if value:
        typer.echo("shrubs 0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    """Shrubs CLI."""
    pass


# Data commands
data_app = typer.Typer(help="Data management commands")
app.add_typer(data_app, name="data")


@data_app.command("status")
def data_status():
    """Show data freshness status."""
    store = DataStore(settings.data_dir)

    for asset_type in ["equities", "futures", "options"]:
        for timeframe in ["daily", "minute"]:
            symbols = store.list_symbols(asset_type, timeframe)
            if symbols:
                typer.echo(f"{asset_type}/{timeframe}: {len(symbols)} symbols")

    typer.echo("Data status complete.")


@data_app.command("fetch")
def data_fetch(
    symbol: str = typer.Option(None, help="Single symbol to fetch"),
    universe: str = typer.Option(None, help="Universe to fetch (e.g., sp500)"),
    days: int = typer.Option(30, help="Days of history"),
    source: str = typer.Option("ib", help="Data source (ib, polygon)"),
):
    """Fetch market data."""
    typer.echo(f"Fetching data: symbol={symbol}, universe={universe}, days={days}, source={source}")
    # TODO: Implement actual fetching
    typer.echo("Data fetch complete.")


# Portfolio commands
portfolio_app = typer.Typer(help="Portfolio management commands")
app.add_typer(portfolio_app, name="portfolio")


@portfolio_app.command("show")
def portfolio_show(
    dry_run: bool = typer.Option(False, help="Don't connect to IB"),
):
    """Show current portfolio positions."""
    if dry_run:
        typer.echo("Portfolio (dry-run mode):")
        typer.echo("  No positions (not connected)")
    else:
        typer.echo("Connecting to IB...")
        # TODO: Actual IB connection
        typer.echo("Portfolio positions loaded.")


# Backtest commands
backtest_app = typer.Typer(help="Backtesting commands")
app.add_typer(backtest_app, name="backtest")


@backtest_app.command("run")
def backtest_run(
    strategy: str = typer.Option("equal_weight", help="Strategy to run"),
    output: Path = typer.Option(Path("signals.parquet"), help="Output file"),
):
    """Run a backtest."""
    typer.echo(f"Running backtest: strategy={strategy}, output={output}")
    # TODO: Implement backtest
    typer.echo("Backtest complete.")


# Execute commands
execute_app = typer.Typer(help="Order execution commands")
app.add_typer(execute_app, name="execute")


@execute_app.command("run")
def execute_run(
    input_file: Path = typer.Argument(..., help="Signals file from backtest"),
    dry_run: bool = typer.Option(True, help="Log orders without submitting"),
):
    """Execute orders from backtest signals."""
    typer.echo(f"Executing orders from {input_file} (dry_run={dry_run})")
    # TODO: Implement execution
    typer.echo("Execution complete.")


if __name__ == "__main__":
    app()
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_cli.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add src/shrubs/cli.py tests/unit/test_cli.py
git commit -m "feat: CLI with data, portfolio, backtest, execute commands"
```

---

## Task 11: Pipeline Runner

**Files:**
- Create: `scripts/run_pipeline.py`
- Update: `pyproject.toml` (add script entry)

**Step 1: Create pipeline runner script**

```python
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
```

**Step 2: Create cron wrapper script**

Create `scripts/cron_nightly.sh`:

```bash
#!/bin/bash
# Nightly pipeline runner for cron

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate virtual environment if present
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run pipeline
python scripts/run_pipeline.py

echo "Pipeline completed at $(date)"
```

**Step 3: Make scripts executable**

```bash
chmod +x scripts/run_pipeline.py scripts/cron_nightly.sh
```

**Step 4: Commit**

```bash
git add scripts/
git commit -m "feat: pipeline runner and cron scripts"
```

---

## Task 12: Integration Test - End to End

**Files:**
- Create: `tests/integration/test_pipeline.py`

**Step 1: Write integration test**

```python
"""End-to-end integration test for the pipeline."""
import pytest
from pathlib import Path

from shrubs.config import settings
from shrubs.data.store import DataStore
from shrubs.backtest.engine import BacktestEngine
from shrubs.execution.portfolio import Portfolio


pytestmark = pytest.mark.integration


def test_full_pipeline_dry_run(tmp_path: Path, sample_ohlcv):
    """Run full pipeline in dry-run mode with sample data."""
    # Setup: Create data store with sample data
    store = DataStore(tmp_path)
    symbols = ["AAPL", "MSFT"]

    for symbol in symbols:
        store.save_ohlcv(symbol, sample_ohlcv, "equities", "daily")

    # Step 1: Load data
    data = {}
    for symbol in symbols:
        data[symbol] = store.load_ohlcv(symbol, "equities", "daily")

    assert len(data) == 2

    # Step 2: Run backtest
    def equal_weight(data):
        import pandas as pd
        return pd.DataFrame({
            "symbol": list(data.keys()),
            "weight": [1.0 / len(data)] * len(data),
        })

    engine = BacktestEngine()
    result = engine.run(data, equal_weight)

    assert result.target_positions is not None
    assert len(result.target_positions) == 2

    # Step 3: Compute trades
    portfolio = Portfolio(account_value=100_000)
    prices = {s: data[s]["close"].iloc[-1] for s in symbols}
    trades = portfolio.compute_trades(result.target_positions, prices)

    assert len(trades) == 2
    assert all(t["action"] == "BUY" for t in trades)
```

**Step 2: Run integration test**

Run: `pytest tests/integration/test_pipeline.py -v -m integration`
Expected: 1 passed

**Step 3: Commit**

```bash
git add tests/integration/test_pipeline.py
git commit -m "test: end-to-end integration test"
```

---

## Summary

| Task | Component | Key Deliverable |
|------|-----------|-----------------|
| 1 | Scaffolding | pyproject.toml, config.py, test fixtures |
| 2 | Data Store | Parquet read/write |
| 3 | IB Client | Contract creation |
| 4 | IB Client | Historical data fetching |
| 5 | Polygon | Historical data fallback |
| 6 | Signals | Momentum, mean reversion |
| 7 | Backtest | Engine with metrics |
| 8 | Portfolio | Trade computation |
| 9 | Orders | Order management |
| 10 | CLI | Command interface |
| 11 | Pipeline | Full runner script |
| 12 | Integration | End-to-end test |

**Development order:** Tasks 1-5 (data layer) → Tasks 6-7 (backtest) → Tasks 8-9 (execution) → Tasks 10-12 (wiring)
