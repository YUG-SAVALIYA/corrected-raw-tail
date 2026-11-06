"""Phase 1 ingestion, validation, immutable provenance, and Parquet persistence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .adapters import OPTIONAL_COLUMNS, adapt_file, discover_input_files
from .config import DataConfig, load_data_config


@dataclass(frozen=True)
class IngestionArtifacts:
    normalized_path: Path | None
    quality_report_path: Path
    quality_by_symbol_path: Path
    manifest_path: Path
    duckdb_path: Path | None
    rows_written: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def _valid_row_mask(frame: pd.DataFrame) -> pd.Series:
    required = frame[["date", "symbol", "open", "high", "low", "close"]].notna().all(axis=1)
    positive = (frame[["open", "high", "low", "close"]] > 0).all(axis=1)
    nonnegative_volume = frame["volume"].notna() & frame["volume"].ge(0)
    ohlc_consistent = (frame["high"] >= frame[["open", "close", "low"]].max(axis=1)) & (
        frame["low"] <= frame[["open", "close", "high"]].min(axis=1)
    )
    return required & positive & nonnegative_volume & ohlc_consistent


def _apply_membership(frame: pd.DataFrame, config: DataConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame["is_universe_eligible"] = True
    if config.universe_membership_path is None:
        return frame, {"membership_available": False, "survivorship_bias_warning": True}
    membership_path = config.universe_membership_path
    if not membership_path.exists():
        raise FileNotFoundError(f"Universe membership file does not exist: {membership_path}")
    membership = pd.read_parquet(membership_path) if membership_path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(membership_path)
    needed = [config.membership_symbol_column, config.membership_start_column, config.membership_end_column]
    missing = [column for column in needed if column not in membership.columns]
    if missing:
        raise ValueError(f"Membership file missing columns: {', '.join(missing)}")
    membership = membership[needed].copy()
    membership.columns = ["symbol", "start_date", "end_date"]
    membership["symbol"] = membership["symbol"].astype("string").str.upper().str.strip()
    membership["start_date"] = pd.to_datetime(membership["start_date"], errors="coerce").dt.normalize()
    membership["end_date"] = pd.to_datetime(membership["end_date"], errors="coerce").dt.normalize()
    frame["is_universe_eligible"] = False
    for row in membership.dropna(subset=["symbol", "start_date"]).itertuples(index=False):
        end_date = row.end_date if pd.notna(row.end_date) else pd.Timestamp.max
        match = (frame["symbol"] == row.symbol) & frame["date"].between(row.start_date, end_date)
        frame.loc[match, "is_universe_eligible"] = True
    return frame, {"membership_available": True, "survivorship_bias_warning": False, "membership_rows": len(membership)}


def _add_uncertainty_flags(frame: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    frame = frame.sort_values(["symbol", "date", "source_file", "source_row"], kind="stable").copy()
    close_return = frame.groupby("symbol", sort=False)["close"].pct_change().abs()
    extreme = close_return.ge(config.extreme_move_threshold).fillna(False)
    verified = frame["corporate_action_verified"].fillna(False).astype(bool)
    frame["corporate_action_uncertain"] = extreme & ~verified
    uncertainty_is_eligible = (
        ~frame["corporate_action_uncertain"]
        if config.exclude_uncertain_intervals
        else pd.Series(True, index=frame.index)
    )
    frame["is_research_eligible"] = frame["is_universe_eligible"] & uncertainty_is_eligible
    return frame


def _symbol_quality(frame: pd.DataFrame, all_sessions: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for symbol, group in frame.groupby("symbol", sort=True):
        dates = pd.DatetimeIndex(group["date"].unique()).sort_values()
        span_sessions = all_sessions[(all_sessions >= dates.min()) & (all_sessions <= dates.max())]
        rows.append(
            {
                "symbol": symbol,
                "rows": int(len(group)),
                "first_date": dates.min().date().isoformat(),
                "last_date": dates.max().date().isoformat(),
                "active_sessions": int(len(dates)),
                "missing_sessions_within_listing_span": int(len(span_sessions.difference(dates))),
                "uncertain_corporate_action_rows": int(group["corporate_action_uncertain"].sum()),
                "research_eligible_rows": int(group["is_research_eligible"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _write_duckdb(output_dir: Path, normalized_path: Path, quality_path: Path) -> Path | None:
    try:
        import duckdb  # type: ignore[import-not-found]
    except ImportError:
        return None
    db_path = output_dir / "research_catalog.duckdb"
    with duckdb.connect(str(db_path)) as connection:
        normalized_sql = str(normalized_path).replace("'", "''")
        quality_sql = str(quality_path).replace("'", "''")
        connection.execute(f"CREATE OR REPLACE VIEW normalized_ohlcv AS SELECT * FROM read_parquet('{normalized_sql}')")
        connection.execute(f"CREATE OR REPLACE VIEW quality_by_symbol AS SELECT * FROM read_csv_auto('{quality_sql}')")
    return db_path


def ingest(config: DataConfig, *, write_normalized: bool = True) -> IngestionArtifacts:
    """Ingest source data deterministically and persist Phase 1 artifacts."""

    files = discover_input_files(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    schema_errors: list[str] = []
    for path in files:
        frame, errors = adapt_file(path, config)
        schema_errors.extend(errors)
        if not frame.empty:
            frames.append(frame)
    if schema_errors:
        raise ValueError("Input schema validation failed:\n" + "\n".join(schema_errors))
    if not frames:
        raise ValueError("No readable OHLCV rows were found")

    raw = pd.concat(frames, ignore_index=True)
    raw_rows = len(raw)
    valid_mask = _valid_row_mask(raw)
    invalid_rows = int((~valid_mask).sum())
    normalized = raw.loc[valid_mask].copy()
    normalized = normalized.sort_values(["symbol", "date", "source_file", "source_row"], kind="stable")
    duplicate_mask = normalized.duplicated(["symbol", "date"], keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    if duplicate_rows and config.duplicate_policy == "error":
        duplicate_keys = normalized.loc[duplicate_mask, ["symbol", "date"]].head(10).to_dict("records")
        raise ValueError(f"Duplicate symbol/date observations found, examples: {duplicate_keys}")
    if config.duplicate_policy in {"first", "last"}:
        normalized = normalized.drop_duplicates(["symbol", "date"], keep=config.duplicate_policy)
    normalized = normalized.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    normalized, membership_report = _apply_membership(normalized, config)
    normalized = _add_uncertainty_flags(normalized, config)
    all_sessions = pd.DatetimeIndex(normalized["date"].unique()).sort_values()
    symbol_quality = _symbol_quality(normalized, all_sessions)

    normalized_path = config.output_dir / "normalized_ohlcv.parquet" if write_normalized else None
    if normalized_path is not None:
        columns = ["date", "symbol", "open", "high", "low", "close", "volume", *OPTIONAL_COLUMNS,
                   "source_file", "source_row", "is_universe_eligible", "corporate_action_uncertain", "is_research_eligible"]
        normalized[columns].to_parquet(normalized_path, index=False)
    quality_by_symbol_path = config.output_dir / "quality_by_symbol.csv"
    symbol_quality.to_csv(quality_by_symbol_path, index=False)
    first_date, last_date = all_sessions.min(), all_sessions.max()
    limitations = [
        "Raw source files are not modified; hashes are recorded in source_manifest.json.",
        "Extreme close-to-close moves without verified corporate-action data are flagged as uncertain, not adjusted.",
        "No missing OHLC sessions are fabricated.",
    ]
    if first_date > pd.Timestamp("2015-01-01"):
        limitations.append("Available data starts after 2015; it cannot support the requested full 2015-2025 study window.")
    if last_date > pd.Timestamp("2025-12-31"):
        limitations.append("Rows after 2025 are retained as source data but must not enter the frozen 2025 holdout evaluation.")
    report = {
        "phase": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": len(files),
        "raw_rows": raw_rows,
        "normalized_rows": int(len(normalized)),
        "symbols": int(normalized["symbol"].nunique()),
        "date_coverage": {"first": first_date.date().isoformat(), "last": last_date.date().isoformat()},
        "research_window_coverage": {
            "requested_start": "2015-01-01",
            "requested_final_holdout_end": "2025-12-31",
            "has_full_requested_history": bool(first_date <= pd.Timestamp("2015-01-01")),
            "contains_post_holdout_rows": bool(last_date > pd.Timestamp("2025-12-31")),
        },
        "excluded_invalid_rows": invalid_rows,
        "duplicate_rows_detected": duplicate_rows,
        "duplicate_policy": config.duplicate_policy,
        "missing_sessions_within_symbol_spans": int(symbol_quality["missing_sessions_within_listing_span"].sum()),
        "uncertain_corporate_action_rows": int(normalized["corporate_action_uncertain"].sum()),
        "index_coverage": {
            "requested": config.index_symbols,
            "present": sorted(set(config.index_symbols).intersection(set(normalized["symbol"]))),
            "missing": sorted(set(config.index_symbols).difference(set(normalized["symbol"]))),
        },
        "schema_errors": schema_errors,
        **membership_report,
        "limitations": limitations,
    }
    quality_report_path = config.output_dir / "data_quality_report.json"
    quality_report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    manifest_path = config.output_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps(_source_manifest(files), indent=2), encoding="utf-8")
    metadata = {"config": config.model_dump(mode="json"), "artifacts": {key: str(value) for key, value in asdict(IngestionArtifacts(normalized_path, quality_report_path, quality_by_symbol_path, manifest_path, None, len(normalized))).items()}}
    (config.output_dir / "ingestion_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    duckdb_path = _write_duckdb(config.output_dir, normalized_path, quality_by_symbol_path) if normalized_path else None
    return IngestionArtifacts(normalized_path, quality_report_path, quality_by_symbol_path, manifest_path, duckdb_path, len(normalized))


def ingest_from_config(path: str | Path) -> IngestionArtifacts:
    return ingest(load_data_config(path))
