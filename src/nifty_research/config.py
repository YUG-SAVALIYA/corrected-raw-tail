"""Validated configuration for the immutable-data ingestion layer."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ColumnMapping(BaseModel):
    """Source-column aliases, matched case-insensitively."""

    date: list[str] = Field(default_factory=lambda: ["date", "datetime", "timestamp", "time"])
    symbol: list[str] = Field(default_factory=lambda: ["symbol", "ticker", "security", "instrument"])
    open: list[str] = Field(default_factory=lambda: ["open", "o"])
    high: list[str] = Field(default_factory=lambda: ["high", "h"])
    low: list[str] = Field(default_factory=lambda: ["low", "l"])
    close: list[str] = Field(default_factory=lambda: ["close", "c", "ltp"])
    volume: list[str] = Field(default_factory=lambda: ["volume", "vol", "qty", "quantity"])
    turnover: list[str] = Field(default_factory=lambda: ["turnover", "value", "traded_value"])
    sector: list[str] = Field(default_factory=lambda: ["sector", "industry"])
    adjusted_close: list[str] = Field(default_factory=lambda: ["adjusted_close", "adj_close"])
    corporate_action_verified: list[str] = Field(
        default_factory=lambda: ["corporate_action_verified", "adjustment_verified"]
    )


class DataConfig(BaseModel):
    """Configuration for a CSV/Parquet file or a directory of symbol files."""

    input_path: Path
    output_dir: Path = Path("artifacts/nifty_phase1")
    file_globs: list[str] = Field(default_factory=lambda: ["*_Day.csv", "*_Day.parquet"])
    recursive: bool = False
    column_mapping: ColumnMapping = Field(default_factory=ColumnMapping)
    symbol_from_filename_suffixes: list[str] = Field(
        default_factory=lambda: ["_daily_candles", "-daily-candles", "_day", "-day", "_daily", "-daily"]
    )
    duplicate_policy: Literal["error", "first", "last"] = "last"
    extreme_move_threshold: float = Field(default=0.30, gt=0.0)
    exclude_uncertain_intervals: bool = True
    index_symbols: list[str] = Field(default_factory=lambda: ["NIFTY50", "NIFTY500"])
    universe_membership_path: Path | None = None
    membership_symbol_column: str = "symbol"
    membership_start_column: str = "start_date"
    membership_end_column: str = "end_date"

    @model_validator(mode="after")
    def validate_file_globs(self) -> "DataConfig":
        if not self.file_globs:
            raise ValueError("file_globs must contain at least one CSV or Parquet pattern")
        return self


def load_data_config(path: str | Path) -> DataConfig:
    """Read YAML and resolve relative paths relative to the config file."""

    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Data config must be a YAML mapping")
    config = DataConfig.model_validate(payload)
    parent = config_path.parent
    for field in ("input_path", "output_dir", "universe_membership_path"):
        value = getattr(config, field)
        if value is not None and not value.is_absolute():
            setattr(config, field, (parent / value).resolve())
    return config
