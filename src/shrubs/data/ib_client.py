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
