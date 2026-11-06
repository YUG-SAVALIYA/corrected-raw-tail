"""CLI entry points for the Nifty next-day research engine."""

from __future__ import annotations

import argparse
from pathlib import Path

from .ingest import ingest_from_config
from .phase2 import run_phase2_from_config
from .phase3 import run_phase3_from_config
from .phase4 import run_phase4_from_config
from .backtest import run_backtest_from_config
from .empirical import run_empirical_from_config
from .sequence import run_sequence_models_from_config
from .robustness import run_robustness_from_config


def _ingest_command(config: Path) -> None:
    artifacts = ingest_from_config(config)
    print(f"Ingested {artifacts.rows_written:,} normalized OHLCV rows.")
    print(f"Quality report: {artifacts.quality_report_path}")
    if artifacts.duckdb_path:
        print(f"DuckDB catalog: {artifacts.duckdb_path}")


def _features_command(config: Path) -> None:
    artifacts = run_phase2_from_config(config)
    print(f"Built {artifacts['rows']:,} Phase 2 feature rows.")
    print(f"Features: {artifacts['features']}")


def _baselines_command(config: Path) -> None:
    artifacts = run_phase3_from_config(config)
    print(f"Completed {artifacts['folds']} chronological baseline folds.")
    print(f"Classification metrics: {artifacts['classification_metrics']}")


def _train_command(config: Path) -> None:
    artifacts = run_phase4_from_config(config)
    print(f"Completed {artifacts['folds']} chronological model folds.")
    print(f"Classification metrics: {artifacts['classification_metrics']}")


def _backtest_command(config: Path) -> None:
    artifacts = run_backtest_from_config(config)
    print(f"Completed {artifacts['runs']} strategy/cost simulations.")
    print(f"Summary: {artifacts['summary']}")


def _robustness_command(config: Path) -> None:
    artifacts = run_robustness_from_config(config)
    print(f"Robustness report: {artifacts['report']}")


def _empirical_command(config: Path) -> None:
    artifacts = run_empirical_from_config(config)
    print(f"Completed {artifacts['folds']} empirical model folds.")
    print(f"Classification metrics: {artifacts['classification_metrics']}")


def _sequence_command(config: Path) -> None:
    artifacts = run_sequence_models_from_config(config)
    print(f"Completed {artifacts['folds']} chronological sequence-model folds.")
    print(f"Classification metrics: {artifacts['classification_metrics']}")


def main() -> None:
    """Run Typer when available; retain a minimal compatible fallback for bootstrap."""

    try:
        import typer  # type: ignore[import-not-found]
    except ImportError:
        parser = argparse.ArgumentParser(prog="nifty")
        subcommands = parser.add_subparsers(dest="command", required=True)
        for command in ("ingest", "validate", "features", "baselines", "walk-forward", "train", "backtest", "robustness", "empirical", "sequence"):
            child = subcommands.add_parser(command)
            child.add_argument("--config", required=True, type=Path)
        args = parser.parse_args()
        if args.command in {"ingest", "validate"}:
            _ingest_command(args.config)
        elif args.command == "features":
            _features_command(args.config)
        elif args.command in {"baselines", "walk-forward"}:
            _baselines_command(args.config)
        elif args.command == "train":
            _train_command(args.config)
        elif args.command == "backtest":
            _backtest_command(args.config)
        elif args.command == "robustness":
            _robustness_command(args.config)
        elif args.command == "empirical":
            _empirical_command(args.config)
        else:
            _sequence_command(args.config)
        return

    app = typer.Typer(help="Nifty next-day candle research engine")

    @app.command()
    def ingest(config: Path = typer.Option(..., exists=True, readable=True)) -> None:
        """Normalize daily OHLCV data and write Phase 1 artifacts."""
        _ingest_command(config)

    @app.command()
    def validate(config: Path = typer.Option(..., exists=True, readable=True)) -> None:
        """Run deterministic ingestion validation and write its quality report."""
        _ingest_command(config)

    @app.command()
    def features(config: Path = typer.Option(..., exists=True, readable=True)) -> None:
        """Build separate next-session labels and no-lookahead feature rows."""
        _features_command(config)

    @app.command("baselines")
    def baselines(config: Path = typer.Option(..., exists=True, readable=True)) -> None:
        """Run Phase 3 baseline-only expanding-window evaluation."""
        _baselines_command(config)

    @app.command("walk-forward")
    def walk_forward(config: Path = typer.Option(..., exists=True, readable=True)) -> None:
        """Alias for the current baseline walk-forward evaluation."""
        _baselines_command(config)

    @app.command()
    def train(config: Path = typer.Option(..., exists=True, readable=True)) -> None:
        """Train and calibrate Phase 4 LightGBM/CatBoost models chronologically."""
        _train_command(config)

    @app.command()
    def backtest(config: Path = typer.Option(..., exists=True, readable=True)) -> None:
        """Run Phase 5 next-open-to-close long-only simulations."""
        _backtest_command(config)

    @app.command()
    def robustness(config: Path = typer.Option(..., exists=True, readable=True)) -> None:
        """Run Phase 6 robustness experiments and generate the research report."""
        _robustness_command(config)

    @app.command()
    def empirical(config: Path = typer.Option(..., exists=True, readable=True)) -> None:
        """Compare additional classifiers on chronological held-out periods."""
        _empirical_command(config)

    @app.command("sequence")
    def sequence_models(config: Path = typer.Option(..., exists=True, readable=True)) -> None:
        """Train compact causal LSTM and Transformer sequence models."""
        _sequence_command(config)

    app()
