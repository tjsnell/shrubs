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
