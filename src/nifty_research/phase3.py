"""Phase 3 chronological baseline training and out-of-sample evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator

from .baselines import binary_conditional_score, classification_metrics, logistic_scores, metric_record, regression_metrics
from .validation import DateGroupedExpandingSplitter


class Phase3Config(BaseModel):
    features_path: Path
    labels_path: Path
    output_dir: Path = Path("artifacts/nifty_phase3")
    initial_train_end: str = "2022-12-31"
    evaluation_start: str = "2023-01-01"
    evaluation_end: str = "2024-12-31"
    retrain_frequency: str = "yearly"
    top_k: int = Field(default=5, ge=1)
    logistic_max_train_rows: int = Field(default=500_000, ge=1_000)
    random_state: int = 42

    @model_validator(mode="after")
    def reject_frozen_holdout(self) -> "Phase3Config":
        if pd.Timestamp(self.evaluation_end) > pd.Timestamp("2024-12-31"):
            raise ValueError("Phase 3 cannot evaluate the frozen 2025 holdout; use the gated holdout command in a later phase.")
        return self


def load_phase3_config(path: str | Path) -> Phase3Config:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = Phase3Config.model_validate(payload)
    for field in ("features_path", "labels_path", "output_dir"):
        value = getattr(config, field)
        if not value.is_absolute():
            setattr(config, field, (config_path.parent / value).resolve())
    return config


def _load_dataset(config: Phase3Config) -> tuple[pd.DataFrame, list[str]]:
    features = pd.read_parquet(config.features_path)
    labels = pd.read_parquet(config.labels_path)
    features = features.rename(columns={"date": "signal_date"})
    frame = features.merge(labels, on=["symbol", "signal_date"], how="inner", validate="one_to_one")
    forbidden = {"target_date", "target_green", "target_return", "is_binary_eligible"}
    feature_columns = [column for column in features.columns if column not in {"symbol", "signal_date", "is_universe_eligible", "corporate_action_uncertain", "is_research_eligible", "feature_ready"}]
    if forbidden.intersection(feature_columns):
        raise AssertionError("Target column present in feature list")
    return frame, feature_columns


def run_phase3(config: Phase3Config) -> dict[str, Path | int]:
    frame, feature_columns = _load_dataset(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = config.output_dir / "predictions"
    model_dir = config.output_dir / "models"
    prediction_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    splitter = DateGroupedExpandingSplitter(
        initial_train_end=config.initial_train_end,
        evaluation_start=config.evaluation_start,
        evaluation_end=config.evaluation_end,
        frequency=config.retrain_frequency,
    )
    folds = splitter.folds(frame)
    if not folds:
        raise ValueError("No evaluation folds found for the configured dates")
    classification_rows: list[dict[str, object]] = []
    regression_rows: list[dict[str, object]] = []
    prediction_paths: list[str] = []
    eligible = frame["feature_ready"] & frame["is_research_eligible"]
    for fold in folds:
        train_mask, evaluation_mask = splitter.masks(frame, fold)
        train_class = frame.loc[eligible & train_mask & frame["is_binary_eligible"]].copy()
        evaluation_class = frame.loc[eligible & evaluation_mask & frame["is_binary_eligible"]].copy()
        train_return = frame.loc[eligible & train_mask & frame["target_return"].notna()].copy()
        evaluation_return = frame.loc[eligible & evaluation_mask & frame["target_return"].notna()].copy()
        if train_class.empty or evaluation_class.empty:
            continue
        base_rate = float(train_class["target_green"].mean())
        threshold = float(train_class["cs_return_20_rank"].quantile(0.80))
        conditions = {
            "previous_candle_direction": frame["previous_candle_green"].eq(1),
            "simple_momentum": frame["return_5d"].gt(0),
            "relative_strength_ranking": frame["cs_return_20_rank"].ge(threshold),
        }
        score_map: dict[str, np.ndarray] = {"majority_class": np.full(len(evaluation_class), base_rate)}
        for name, condition in conditions.items():
            score_map[name] = binary_conditional_score(train_class, evaluation_class, condition)
        logistic_probability, model = logistic_scores(train_class, evaluation_class, feature_columns, max_train_rows=config.logistic_max_train_rows, random_state=config.random_state)
        score_map["logistic_regression"] = logistic_probability
        joblib.dump({"model": model, "feature_order": feature_columns, "training_cutoff": str(fold.train_cutoff.date())}, model_dir / f"logistic_{fold.name}.joblib")
        for model_name, scores in score_map.items():
            metric = classification_metrics(evaluation_class["target_green"], scores, evaluation_class["signal_date"], top_k=config.top_k)
            classification_rows.append(metric_record(fold=fold.name, model=model_name, metric=metric, observations=len(evaluation_class), train_observations=len(train_class)))
            prediction_path = prediction_dir / f"baseline_predictions_{fold.name}_{model_name}.parquet"
            pd.DataFrame({"symbol": evaluation_class["symbol"].to_numpy(), "signal_date": evaluation_class["signal_date"].to_numpy(), "target_date": evaluation_class["target_date"].to_numpy(), "target_green": evaluation_class["target_green"].astype("int8").to_numpy(), "target_return": evaluation_class["target_return"].to_numpy(), "model": model_name, "score": scores}).to_parquet(prediction_path, index=False)
            prediction_paths.append(str(prediction_path))
        if not train_return.empty and not evaluation_return.empty:
            for model_name, prediction in {"zero_return": np.zeros(len(evaluation_return)), "historical_mean_return": np.full(len(evaluation_return), float(train_return["target_return"].mean()))}.items():
                regression_rows.append({"fold": fold.name, "model": model_name, "observations": len(evaluation_return), "train_observations": len(train_return), **regression_metrics(evaluation_return["target_return"], prediction)})
                prediction_path = prediction_dir / f"return_predictions_{fold.name}_{model_name}.parquet"
                pd.DataFrame({"symbol": evaluation_return["symbol"].to_numpy(), "signal_date": evaluation_return["signal_date"].to_numpy(), "target_date": evaluation_return["target_date"].to_numpy(), "target_return": evaluation_return["target_return"].to_numpy(), "model": model_name, "prediction": prediction}).to_parquet(prediction_path, index=False)
                prediction_paths.append(str(prediction_path))
    if not classification_rows:
        raise ValueError("No eligible classification observations were available in the configured folds")
    classification_path = config.output_dir / "classification_metrics.csv"
    regression_path = config.output_dir / "regression_metrics.csv"
    pd.DataFrame(classification_rows).to_csv(classification_path, index=False)
    pd.DataFrame(regression_rows).to_csv(regression_path, index=False)
    report_path = config.output_dir / "phase3_report.json"
    report_path.write_text(json.dumps({"phase": 3, "feature_count": len(feature_columns), "folds": [fold.name for fold in folds], "predictions": prediction_paths, "limitations": ["Baseline results are descriptive out-of-sample measurements, not claims of profitability.", "This phase hard-excludes 2025 from evaluation.", "Feature rows lacking a full lookback or marked research-ineligible are excluded." ]}, indent=2), encoding="utf-8")
    return {"classification_metrics": classification_path, "regression_metrics": regression_path, "report": report_path, "folds": len(folds)}


def run_phase3_from_config(path: str | Path) -> dict[str, Path | int]:
    return run_phase3(load_phase3_config(path))
