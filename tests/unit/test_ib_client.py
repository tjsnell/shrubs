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
