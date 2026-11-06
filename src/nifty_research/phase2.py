"""Phase 2 orchestration: labels, features, registry, and manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from .features import build_features, write_feature_registry
from .labels import make_next_session_labels


class Phase2Config(BaseModel):
    normalized_path: Path
    output_dir: Path = Path("artifacts/nifty_phase2")
    epsilon: float = 0.0
    doji_policy: str = "exclude"
    enabled_feature_groups: list[str] = Field(default_factory=lambda: ["candle_anatomy", "momentum", "trend", "volatility", "volume", "relative_strength", "calendar", "higher_timeframe"])
    # Optional verified, completed daily index histories.  Supported keys are
    # ``nifty50`` and ``niftybank``; both are required when market_index is an
    # enabled feature group.
    market_index_paths: dict[str, Path] = Field(default_factory=dict)


def load_phase2_config(path: str | Path) -> Phase2Config:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = Phase2Config.model_validate(payload)
    for field in ("normalized_path", "output_dir"):
        value = getattr(config, field)
        if not value.is_absolute():
            setattr(config, field, (config_path.parent / value).resolve())
    config.market_index_paths = {
        name: value if value.is_absolute() else (config_path.parent / value).resolve()
        for name, value in config.market_index_paths.items()
    }
    return config


def _read_market_index(path: Path) -> pd.DataFrame:
    """Read a CSV or Parquet index history with standard OHLC column names."""

    if not path.exists():
        raise FileNotFoundError(f"Market index history not found: {path}")
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    canonical = {str(column).strip().lower(): column for column in frame.columns}
    aliases = {
        "date": ("date", "datetime", "timestamp", "time"),
        "open": ("open", "o"),
        "high": ("high", "h"),
        "low": ("low", "l"),
        "close": ("close", "c", "ltp"),
    }
    resolved: dict[str, str] = {}
    for target, candidates in aliases.items():
        source = next((canonical[name] for name in candidates if name in canonical), None)
        if source is None:
            raise ValueError(f"Market index {path} has no recognized {target!r} column")
        resolved[target] = source
    output = frame[[resolved[name] for name in aliases]].rename(columns={source: target for target, source in resolved.items()})
    for column in ("open", "high", "low", "close"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output = output.dropna(subset=["date", "open", "high", "low", "close"])
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_phase2(config: Phase2Config) -> dict[str, Path | int]:
    if not config.normalized_path.exists():
        raise FileNotFoundError(f"Normalized Phase 1 data not found: {config.normalized_path}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    ohlcv = pd.read_parquet(config.normalized_path)
    labels = make_next_session_labels(ohlcv, epsilon=config.epsilon, doji_policy=config.doji_policy)
    market_indices = {name: _read_market_index(path) for name, path in config.market_index_paths.items()}
    features, registry = build_features(
        ohlcv,
        enabled_groups=set(config.enabled_feature_groups),
        market_indices=market_indices,
    )
    forbidden = {"target_date", "target_green", "target_return", "is_binary_eligible"}.intersection(features.columns)
    if forbidden:
        raise AssertionError(f"Target columns entered feature matrix: {sorted(forbidden)}")
    labels_path = config.output_dir / "labels.parquet"
    features_path = config.output_dir / "features.parquet"
    registry_path = config.output_dir / "feature_registry.json"
    labels.to_parquet(labels_path, index=False)
    features.to_parquet(features_path, index=False)
    write_feature_registry(registry, registry_path)
    report = {
        "phase": 2,
        "source_normalized_path": str(config.normalized_path),
        "source_sha256": _sha256(config.normalized_path),
        "rows": len(features),
        "labels_with_return": int(labels["target_return"].notna().sum()),
        "binary_eligible_labels": int(labels["is_binary_eligible"].sum()),
        "feature_count": len(registry),
        "feature_groups": sorted(set(spec.group for spec in registry)),
        "feature_ready_rows": int(features["feature_ready"].sum()),
        "doji_policy": config.doji_policy,
        "epsilon": config.epsilon,
        "limitations": [
            "Market-index features are emitted only from the verified index histories explicitly supplied in this run; market-regime aggregates are computed only from the supplied stock universe.",
            "Cross-sectional ranks use the currently available source universe; point-in-time membership filtering is required for Nifty 500-specific research.",
        ],
    }
    report_path = config.output_dir / "phase2_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return {"features": features_path, "labels": labels_path, "registry": registry_path, "report": report_path, "rows": len(features)}


def run_phase2_from_config(path: str | Path) -> dict[str, Path | int]:
    return run_phase2(load_phase2_config(path))
