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
