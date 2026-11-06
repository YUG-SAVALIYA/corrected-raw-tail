"""Small causal sequence-model experiment for next-session direction research."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field

from .baselines import classification_metrics, metric_record
from .phase4 import _apply_calibrator, _choose_calibrator
from .validation import DateGroupedExpandingSplitter


SEQUENCE_FEATURES = [
    "close_location", "open_prev_close_gap", "return_1d", "return_2d", "return_5d",
    "return_10d", "rsi_7", "rsi_14", "macd_signal_diff", "consecutive_direction_count",
    "close_sma_20_ratio", "sma_20_slope", "atr_close_14", "volatility_20", "volume_sma_20_ratio",
]


class SequenceConfig(BaseModel):
    features_path: Path
    labels_path: Path
    output_dir: Path = Path("artifacts/nifty_sequence")
    initial_train_end: str = "2022-12-31"
    evaluation_start: str = "2025-01-01"
    evaluation_end: str = "2026-12-31"
    retrain_frequency: str = "yearly"
    calibration_months: int = Field(default=6, ge=2)
    early_stopping_months: int = Field(default=3, ge=1)
    lookback_sessions: int = Field(default=20, ge=5, le=60)
    max_train_sequences: int = Field(default=50_000, ge=5_000)
    max_validation_sequences: int = Field(default=20_000, ge=1_000)
    max_calibration_sequences: int = Field(default=50_000, ge=2_000)
    epochs: int = Field(default=5, ge=1, le=30)
    batch_size: int = Field(default=512, ge=32)
    random_state: int = 42
    n_jobs: int = 4


def load_sequence_config(path: str | Path) -> SequenceConfig:
    config_path = Path(path).resolve()
    config = SequenceConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    for field in ("features_path", "labels_path", "output_dir"):
        value = getattr(config, field)
        if not value.is_absolute():
            setattr(config, field, (config_path.parent / value).resolve())
    return config


def _sample(indexes: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if len(indexes) <= limit:
        return indexes
    return np.sort(rng.choice(indexes, size=limit, replace=False))


def _sequence_model(name: str, inputs: int):
    import torch
    from torch import nn

    class LSTMClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.LSTM(inputs, 32, batch_first=True)
            self.head = nn.Sequential(nn.LayerNorm(32), nn.Dropout(0.15), nn.Linear(32, 1))

        def forward(self, values):
            _, (hidden, _) = self.encoder(values)
            return self.head(hidden[-1]).squeeze(-1)

    class TransformerClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(inputs, 32)
            self.position = nn.Parameter(torch.zeros(1, 60, 32))
            block = nn.TransformerEncoderLayer(d_model=32, nhead=4, dim_feedforward=64, dropout=0.15, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(block, num_layers=2)
            self.head = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 1))

        def forward(self, values):
            encoded = self.projection(values)
            encoded = encoded + self.position[:, : encoded.shape[1]]
            return self.head(self.encoder(encoded)[:, -1]).squeeze(-1)

    return LSTMClassifier() if name == "lstm_classifier" else TransformerClassifier()


def run_sequence_models(config: SequenceConfig) -> dict[str, Path | int]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    torch.set_num_threads(config.n_jobs)
    feature_columns = ["symbol", "date", "feature_ready", "is_research_eligible", *SEQUENCE_FEATURES]
    features = pd.read_parquet(config.features_path, columns=feature_columns).rename(columns={"date": "signal_date"})
    labels = pd.read_parquet(config.labels_path, columns=["symbol", "signal_date", "target_date", "target_green", "target_return", "is_binary_eligible"])
    frame = features.merge(labels, on=["symbol", "signal_date"], how="left", validate="one_to_one").sort_values(["symbol", "signal_date"], kind="stable").reset_index(drop=True)
    frame["sequence_position"] = frame.groupby("symbol", sort=False).cumcount()
    values = frame[SEQUENCE_FEATURES].to_numpy(dtype=np.float32, copy=True)
    labels_array = frame["target_green"].fillna(0).to_numpy(dtype=np.float32)
    eligible = (frame["feature_ready"] & frame["is_research_eligible"] & frame["is_binary_eligible"] & frame["sequence_position"].ge(config.lookback_sessions - 1)).to_numpy(bool)

    class Windows(Dataset):
        def __init__(self, endpoints: np.ndarray, centre: np.ndarray, scale: np.ndarray) -> None:
            self.endpoints, self.centre, self.scale = endpoints, centre, scale

        def __len__(self) -> int:
            return len(self.endpoints)

        def __getitem__(self, item: int):
            end = int(self.endpoints[item])
            window = (values[end - config.lookback_sessions + 1 : end + 1] - self.centre) / self.scale
            return torch.from_numpy(np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)), torch.tensor(labels_array[end])

    def probabilities(model, endpoints: np.ndarray, centre: np.ndarray, scale: np.ndarray) -> np.ndarray:
        loader = DataLoader(Windows(endpoints, centre, scale), batch_size=config.batch_size, shuffle=False)
        model.eval()
        out: list[np.ndarray] = []
        with torch.no_grad():
            for x, _ in loader:
                out.append(torch.sigmoid(model(x)).numpy())
        return np.concatenate(out) if out else np.array([], dtype=float)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = config.output_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    splitter = DateGroupedExpandingSplitter(initial_train_end=config.initial_train_end, evaluation_start=config.evaluation_start, evaluation_end=config.evaluation_end, frequency=config.retrain_frequency)
    metric_rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    rng = np.random.default_rng(config.random_state)
    for fold in splitter.folds(frame):
        cutoff = fold.train_cutoff
        calibration_start = cutoff - pd.DateOffset(months=config.calibration_months)
        validation_start = calibration_start - pd.DateOffset(months=config.early_stopping_months)
        signal = frame["signal_date"]
        target = frame["target_date"]
        train_idx = np.flatnonzero(eligible & signal.lt(validation_start).to_numpy(bool) & target.le(cutoff).to_numpy(bool))
        validation_idx = np.flatnonzero(eligible & signal.between(validation_start, calibration_start - pd.Timedelta(days=1)).to_numpy(bool) & target.le(cutoff).to_numpy(bool))
        calibration_idx = np.flatnonzero(eligible & signal.ge(calibration_start).to_numpy(bool) & signal.le(cutoff).to_numpy(bool) & target.le(cutoff).to_numpy(bool))
        evaluation_idx = np.flatnonzero(eligible & signal.between(fold.evaluation_start, fold.evaluation_end).to_numpy(bool) & target.le(fold.evaluation_end).to_numpy(bool))
        train_idx = _sample(train_idx, config.max_train_sequences, rng)
        validation_idx = _sample(validation_idx, config.max_validation_sequences, rng)
        calibration_idx = _sample(calibration_idx, config.max_calibration_sequences, rng)
        if min(len(train_idx), len(validation_idx), len(calibration_idx), len(evaluation_idx)) == 0:
            continue
        centre = np.nanmedian(values[train_idx], axis=0)
        scale = np.nanstd(values[train_idx], axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
        centre = np.where(np.isfinite(centre), centre, 0.0).astype(np.float32)
        calibration_dates = signal.iloc[calibration_idx].to_numpy()
        split_date = np.sort(pd.unique(calibration_dates))[len(pd.unique(calibration_dates)) // 2]
        calibration_fit = calibration_idx[calibration_dates < split_date]
        calibration_select = calibration_idx[calibration_dates >= split_date]
        for name in ("lstm_classifier", "transformer_classifier"):
            model = _sequence_model(name, len(SEQUENCE_FEATURES))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
            loss_fn = nn.BCEWithLogitsLoss()
            train_loader = DataLoader(Windows(train_idx, centre, scale), batch_size=config.batch_size, shuffle=True)
            best_state, best_loss = None, float("inf")
            for _ in range(config.epochs):
                model.train()
                for x, y in train_loader:
                    optimizer.zero_grad()
                    loss = loss_fn(model(x), y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                validation_probability = probabilities(model, validation_idx, centre, scale)
                validation_loss = float(nn.BCELoss()(torch.tensor(validation_probability), torch.tensor(labels_array[validation_idx])))
                if validation_loss < best_loss:
                    best_loss, best_state = validation_loss, copy.deepcopy(model.state_dict())
            model.load_state_dict(best_state)
            raw_fit = probabilities(model, calibration_fit, centre, scale)
            raw_select = probabilities(model, calibration_select, centre, scale)
            method, calibrator, comparison = _choose_calibrator(raw_fit, pd.Series(labels_array[calibration_fit]), raw_select, pd.Series(labels_array[calibration_select]))
            raw_eval = probabilities(model, evaluation_idx, centre, scale)
            calibrated = _apply_calibrator(calibrator, raw_eval)
            evaluation = frame.iloc[evaluation_idx]
            metric = classification_metrics(evaluation["target_green"], calibrated, evaluation["signal_date"], top_k=5)
            metric_rows.append(metric_record(fold=fold.name, model=name, metric=metric, observations=len(evaluation), train_observations=len(train_idx)))
            output = evaluation[["symbol", "signal_date", "target_date", "target_green", "target_return", "volatility_20"]].copy()
            output["raw_probability"] = raw_eval
            output["calibrated_probability"] = calibrated
            output["predicted_return"] = 0.0
            output["model"] = name
            path = prediction_dir / f"{name}_{fold.name}.parquet"
            output.to_parquet(path, index=False)
            manifest.append({"fold": fold.name, "model": name, "prediction_path": str(path), "training_cutoff": str(cutoff.date()), "lookback_sessions": config.lookback_sessions, "training_sequences": len(train_idx), "calibration_method": method, "calibration_comparison": comparison})
    if not metric_rows:
        raise ValueError("No sequence-model out-of-sample metrics were generated")
    metrics_path = config.output_dir / "classification_metrics.csv"
    manifest_path = config.output_dir / "model_manifest.json"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"classification_metrics": metrics_path, "manifest": manifest_path, "folds": len({row["fold"] for row in metric_rows})}


def run_sequence_models_from_config(path: str | Path) -> dict[str, Path | int]:
    return run_sequence_models(load_sequence_config(path))
