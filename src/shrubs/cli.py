"""Command-line interface."""
import typer
from pathlib import Path

from shrubs.config import settings
from shrubs.data.store import DataStore
from shrubs.data.ib_client import IBClient, ContractType


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
    # Build symbol list
    symbols = []
    if symbol:
        symbols = [symbol]
    elif universe:
        universes = {
            "test": ["AAPL", "MSFT", "GOOGL"],
            "faang": ["META", "AAPL", "AMZN", "NFLX", "GOOGL"],
            "mag7": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
        }
        symbols = universes.get(universe.lower(), [])
        if not symbols:
            typer.echo(f"Unknown universe: {universe}")
            raise typer.Exit(1)
    else:
        typer.echo("Specify --symbol or --universe")
        raise typer.Exit(1)

    typer.echo(f"Fetching {len(symbols)} symbols from {source} ({days} days)...")

    if source == "ib":
        store = DataStore(settings.data_dir)
        try:
            with IBClient(connect=True) as client:
                for sym in symbols:
                    typer.echo(f"  Fetching {sym}...")
                    contract = client.create_contract(sym, ContractType.STOCK)
                    df = client.fetch_historical(contract, duration=f"{days} D", bar_size="1 day")
                    if len(df) > 0:
                        store.save_ohlcv(sym, df, asset_type="equities", timeframe="daily")
                        typer.echo(f"    Saved {len(df)} bars")
                    else:
                        typer.echo(f"    No data returned")
        except Exception as e:
            typer.echo(f"Error connecting to IB: {e}")
            raise typer.Exit(1)
    else:
        typer.echo(f"Source '{source}' not yet implemented")
        raise typer.Exit(1)

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
