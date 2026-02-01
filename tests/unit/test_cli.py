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
