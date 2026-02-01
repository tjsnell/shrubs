"""Interactive Brokers client wrapper."""
from enum import Enum
from typing import Literal

import pandas as pd
from ib_insync import IB, Stock, Future, Option, Contract, util

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

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.disconnect()
