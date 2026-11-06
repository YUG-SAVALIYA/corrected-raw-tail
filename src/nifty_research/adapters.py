"""Configurable readers and canonical-column adapters for daily OHLCV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DataConfig

CANONICAL_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]
OPTIONAL_COLUMNS = ["turnover", "sector", "adjusted_close", "corporate_action_verified"]


def discover_input_files(config: DataConfig) -> list[Path]:
    """Find supported files deterministically; a file input is also supported."""

    root = config.input_path
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {root}")
    files: set[Path] = set()
    for pattern in config.file_globs:
        iterator = root.rglob(pattern) if config.recursive else root.glob(pattern)
        files.update(path for path in iterator if path.is_file())
    result = sorted(files, key=lambda path: str(path).lower())
    if not result:
        patterns = ", ".join(config.file_globs)
        raise FileNotFoundError(f"No input files found in {root} for patterns: {patterns}")
    return result


def _read_frame(path: Path) -> pd.DataFrame:
    """Read CSV/Parquet, preferring Polars when installed for large files."""

    suffix = path.suffix.lower()
    if suffix not in {".csv", ".parquet", ".pq"}:
        raise ValueError(f"Unsupported input format: {path}")
    try:
        import polars as pl  # type: ignore[import-not-found]

        frame = pl.read_parquet(path) if suffix in {".parquet", ".pq"} else pl.read_csv(path)
        return frame.to_pandas()
    except ImportError:
        return pd.read_parquet(path) if suffix in {".parquet", ".pq"} else pd.read_csv(path)


def _normalise_name(name: object) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _find_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    by_normalised = {_normalise_name(column): str(column) for column in frame.columns}
    for candidate in candidates:
        match = by_normalised.get(_normalise_name(candidate))
        if match is not None:
            return match
    return None


def symbol_from_path(path: Path, config: DataConfig) -> str:
    stem = path.stem.strip()
    lower = stem.lower()
    for suffix in config.symbol_from_filename_suffixes:
        if lower.endswith(suffix.lower()):
            stem = stem[: -len(suffix)]
            break
    if not stem:
        raise ValueError(f"Could not derive symbol from filename: {path.name}")
    return stem.upper()


def adapt_file(path: Path, config: DataConfig) -> tuple[pd.DataFrame, list[str]]:
    """Return canonical rows plus source-schema errors (without mutating source data)."""

    raw = _read_frame(path)
    mapping = config.column_mapping
    fields = {
        "date": mapping.date,
        "symbol": mapping.symbol,
        "open": mapping.open,
        "high": mapping.high,
        "low": mapping.low,
        "close": mapping.close,
        "volume": mapping.volume,
        **{field: getattr(mapping, field) for field in OPTIONAL_COLUMNS},
    }
    resolved = {field: _find_column(raw, candidates) for field, candidates in fields.items()}
    missing = [field for field in ("date", "open", "high", "low", "close", "volume") if resolved[field] is None]
    if missing:
        return pd.DataFrame(), [f"{path.name}: missing required columns {', '.join(missing)}"]

    adapted = pd.DataFrame(index=raw.index)
    for field, source_column in resolved.items():
        if source_column is not None:
            adapted[field] = raw[source_column]
    if "symbol" not in adapted:
        adapted["symbol"] = symbol_from_path(path, config)
    for field in OPTIONAL_COLUMNS:
        if field not in adapted:
            adapted[field] = pd.NA
    adapted["source_file"] = str(path.resolve())
    adapted["source_row"] = range(1, len(adapted) + 1)
    adapted["symbol"] = adapted["symbol"].astype("string").str.strip().str.upper()
    timestamp = pd.to_datetime(adapted["date"], errors="coerce", utc=True)
    adapted["date"] = timestamp.dt.tz_convert(None).dt.normalize()
    for field in ("open", "high", "low", "close", "volume", "turnover", "adjusted_close"):
        adapted[field] = pd.to_numeric(adapted[field], errors="coerce")
    adapted["corporate_action_verified"] = adapted["corporate_action_verified"].astype("string").str.lower().isin(
        ["true", "1", "yes", "y"]
    )
    return adapted, []
