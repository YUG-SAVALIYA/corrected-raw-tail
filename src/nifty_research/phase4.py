"""Phase 4 pooled LightGBM/CatBoost models with chronological calibration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, model_validator

from .baselines import classification_metrics, deterministic_train_sample, metric_record, regression_metrics
from .validation import DateGroupedExpandingSplitter, WalkForwardFold


class Phase4Config(BaseModel):
    features_path: Path
    labels_path: Path
    output_dir: Path = Path("artifacts/nifty_phase4")
    initial_train_end: str = "2022-12-31"
    evaluation_start: str = "2023-01-01"
    evaluation_end: str = "2024-12-31"
    retrain_frequency: str = "yearly"
    calibration_months: int = Field(default=6, ge=2)
    early_stopping_months: int = Field(default=3, ge=1)
    max_train_rows: int = Field(default=500_000, ge=10_000)
    max_calibration_rows: int = Field(default=250_000, ge=10_000)
    random_state: int = 42
    n_jobs: int = 4
    # The 2025 period is frozen by default.  A separately named configuration
    # must opt in before it can be used as a final holdout evaluation period.
    allow_post_2024_evaluation: bool = False

    @model_validator(mode="after")
    def block_holdout(self) -> "Phase4Config":
        if pd.Timestamp(self.evaluation_end) > pd.Timestamp("2024-12-31") and not self.allow_post_2024_evaluation:
            raise ValueError("Phase 4 cannot access the frozen 2025 holdout without explicit allow_post_2024_evaluation.")
        return self


def load_phase4_config(path: str | Path) -> Phase4Config:
    config_path = Path(path).resolve()
    config = Phase4Config.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    for field in ("features_path", "labels_path", "output_dir"):
        value = getattr(config, field)
        if not value.is_absolute():
            setattr(config, field, (config_path.parent / value).resolve())
    return config


def _load(config: Phase4Config) -> tuple[pd.DataFrame, list[str]]:
    features = pd.read_parquet(config.features_path).rename(columns={"date": "signal_date"})
    labels = pd.read_parquet(config.labels_path)
    frame = features.merge(labels, on=["symbol", "signal_date"], how="left", validate="one_to_one")
    excluded = {"symbol", "signal_date", "is_universe_eligible", "corporate_action_uncertain", "is_research_eligible", "feature_ready"}
    target_names = {"target_date", "target_green", "target_return", "is_binary_eligible"}
    feature_columns = [column for column in features.columns if column not in excluded]
    if target_names.intersection(feature_columns):
        raise AssertionError("Target column entered Phase 4 feature matrix")
    return frame, feature_columns


def _partition_train(frame: pd.DataFrame, fold: WalkForwardFold, config: Phase4Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = frame["feature_ready"] & frame["is_research_eligible"] & frame["is_binary_eligible"]
    matured = valid & frame["signal_date"].le(fold.train_cutoff) & frame["target_date"].le(fold.train_cutoff)
    calibration_start = fold.train_cutoff - pd.DateOffset(months=config.calibration_months)
    early_stop_start = calibration_start - pd.DateOffset(months=config.early_stopping_months)
    train = frame.loc[matured & frame["signal_date"].lt(early_stop_start)].copy()
    early_stop = frame.loc[matured & frame["signal_date"].between(early_stop_start, calibration_start - pd.Timedelta(days=1))].copy()
    calibration = frame.loc[matured & frame["signal_date"].ge(calibration_start)].copy()
    if min(len(train), len(early_stop), len(calibration)) == 0:
        raise ValueError(f"Insufficient chronological partitions for fold {fold.name}")
    return train, early_stop, calibration


def _split_calibration(calibration: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(calibration["signal_date"].unique()).sort_values()
    midpoint = dates[len(dates) // 2]
    fit = calibration.loc[calibration["signal_date"] < midpoint].copy()
    select = calibration.loc[calibration["signal_date"].ge(midpoint)].copy()
    if fit.empty or select.empty:
        raise ValueError("Calibration period cannot be split chronologically")
    return fit, select


def _fit_calibrator(method: str, probabilities: np.ndarray, target: pd.Series):
    if method == "raw":
        return None
    if method == "sigmoid":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(random_state=42).fit(probabilities.reshape(-1, 1), target.astype(int))
        return model
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        return IsotonicRegression(out_of_bounds="clip").fit(probabilities, target.astype(int))
    raise ValueError(f"Unknown calibration method: {method}")


def _apply_calibrator(calibrator, probabilities: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return np.asarray(probabilities, dtype=float)
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
    return calibrator.predict(probabilities)


def _choose_calibrator(raw_fit: np.ndarray, fit_y: pd.Series, raw_select: np.ndarray, select_y: pd.Series) -> tuple[str, object, dict[str, dict[str, float]]]:
    candidates = {method: _fit_calibrator(method, raw_fit, fit_y) for method in ("raw", "sigmoid", "isotonic")}
    comparison: dict[str, dict[str, float]] = {}
    for method, calibrator in candidates.items():
        scores = _apply_calibrator(calibrator, raw_select)
        metric = classification_metrics(select_y, scores, pd.Series(np.arange(len(select_y))), top_k=1)
        comparison[method] = {"log_loss": metric.log_loss, "brier_score": metric.brier_score}
    choice = min(comparison, key=lambda name: (comparison[name]["log_loss"], comparison[name]["brier_score"]))
    full_probability = np.concatenate([raw_fit, raw_select])
    full_target = pd.concat([fit_y, select_y], ignore_index=True)
    return choice, _fit_calibrator(choice, full_probability, full_target), comparison


def _models(config: Phase4Config):
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier, LGBMRegressor

    common = dict(learning_rate=0.03, num_leaves=15, max_depth=5, min_child_samples=200, colsample_bytree=0.8, subsample=0.8, subsample_freq=1, reg_lambda=5.0, n_estimators=1500, random_state=config.random_state, n_jobs=config.n_jobs, verbosity=-1)
    return (
        LGBMClassifier(**common),
        LGBMRegressor(**common, objective="regression_l1"),
        CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=5, l2_leaf_reg=5.0, loss_function="Logloss", eval_metric="Logloss", random_seed=config.random_state, thread_count=config.n_jobs, verbose=False, allow_writing_files=False),
    )


def _hash_dataset(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def run_phase4(config: Phase4Config) -> dict[str, Path | int]:
    from lightgbm import early_stopping

    frame, feature_columns = _load(config)
    dataset_hash = _hash_dataset([config.features_path, config.labels_path])
    config.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir, model_dir = config.output_dir / "predictions", config.output_dir / "models"
    prediction_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    splitter = DateGroupedExpandingSplitter(initial_train_end=config.initial_train_end, evaluation_start=config.evaluation_start, evaluation_end=config.evaluation_end, frequency=config.retrain_frequency)
    metric_rows: list[dict[str, object]] = []
    regression_rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for fold in splitter.folds(frame):
        train, early_stop, calibration = _partition_train(frame, fold, config)
        train = deterministic_train_sample(train, config.max_train_rows)
        early_stop = deterministic_train_sample(early_stop, min(config.max_calibration_rows, len(early_stop)))
        calibration = deterministic_train_sample(calibration, config.max_calibration_rows)
        calibration_fit, calibration_select = _split_calibration(calibration)
        lgb_classifier, lgb_regressor, cat_classifier = _models(config)
        callbacks = [early_stopping(stopping_rounds=100, verbose=False)]
        lgb_classifier.fit(train[feature_columns], train["target_green"].astype(int), eval_set=[(early_stop[feature_columns], early_stop["target_green"].astype(int))], callbacks=callbacks)
        lgb_regressor.fit(train[feature_columns], train["target_return"], eval_set=[(early_stop[feature_columns], early_stop["target_return"])], callbacks=callbacks)
        cat_classifier.fit(train[feature_columns], train["target_green"].astype(int), eval_set=(early_stop[feature_columns], early_stop["target_green"].astype(int)), early_stopping_rounds=100)
        eligible_eval = frame["feature_ready"] & frame["is_research_eligible"]
        eval_mask = frame["signal_date"].between(fold.evaluation_start, fold.evaluation_end) & frame["target_date"].le(fold.evaluation_end)
        evaluation = frame.loc[eligible_eval & eval_mask].copy()
        if evaluation.empty:
            continue
        outputs: dict[str, tuple[np.ndarray, np.ndarray | None, object]] = {}
        for name, classifier in (("lightgbm_classifier", lgb_classifier), ("catboost_classifier", cat_classifier)):
            raw_fit = classifier.predict_proba(calibration_fit[feature_columns])[:, 1]
            raw_select = classifier.predict_proba(calibration_select[feature_columns])[:, 1]
            method, calibrator, comparison = _choose_calibrator(raw_fit, calibration_fit["target_green"], raw_select, calibration_select["target_green"])
            raw_eval = classifier.predict_proba(evaluation[feature_columns])[:, 1]
            calibrated = _apply_calibrator(calibrator, raw_eval)
            outputs[name] = (raw_eval, calibrated, {"model": classifier, "calibrator": calibrator, "calibration_method": method, "comparison": comparison})
        predicted_return = lgb_regressor.predict(evaluation[feature_columns])
        evaluation_target = evaluation.loc[evaluation["is_binary_eligible"]].copy()
        for name, (raw_probability, calibrated_probability, artifact) in outputs.items():
            if not evaluation_target.empty:
                position = evaluation["is_binary_eligible"].to_numpy(bool)
                metric = classification_metrics(evaluation_target["target_green"], calibrated_probability[position], evaluation_target["signal_date"], top_k=5)
                metric_rows.append(metric_record(fold=fold.name, model=name, metric=metric, observations=len(evaluation_target), train_observations=len(train)))
            prediction_path = prediction_dir / f"{name}_{fold.name}.parquet"
            pd.DataFrame({"symbol": evaluation["symbol"].to_numpy(), "signal_date": evaluation["signal_date"].to_numpy(), "target_date": evaluation["target_date"].to_numpy(), "target_green": evaluation["target_green"].astype("Int8").to_numpy(), "target_return": evaluation["target_return"].to_numpy(), "raw_probability": raw_probability, "calibrated_probability": calibrated_probability, "predicted_return": predicted_return, "volatility_20": evaluation["volatility_20"].to_numpy(), "model": name}).to_parquet(prediction_path, index=False)
            artifact_path = model_dir / f"{name}_{fold.name}.joblib"
            joblib.dump({**artifact, "feature_order": feature_columns, "training_cutoff": str(fold.train_cutoff.date()), "dataset_hash": dataset_hash}, artifact_path)
            manifest.append({"fold": fold.name, "model": name, "prediction_path": str(prediction_path), "artifact_path": str(artifact_path), "calibration_method": artifact["calibration_method"], "calibration_comparison": artifact["comparison"]})
        regression_eval = evaluation.loc[evaluation["target_return"].notna()]
        if not regression_eval.empty:
            prediction = predicted_return[evaluation["target_return"].notna().to_numpy(bool)]
            regression_rows.append({"fold": fold.name, "model": "lightgbm_regressor", "observations": len(regression_eval), "train_observations": len(train), **regression_metrics(regression_eval["target_return"], prediction)})
            reg_path = prediction_dir / f"lightgbm_regressor_{fold.name}.parquet"
            pd.DataFrame({"symbol": regression_eval["symbol"].to_numpy(), "signal_date": regression_eval["signal_date"].to_numpy(), "target_date": regression_eval["target_date"].to_numpy(), "target_return": regression_eval["target_return"].to_numpy(), "predicted_return": prediction, "model": "lightgbm_regressor"}).to_parquet(reg_path, index=False)
    if not metric_rows:
        raise ValueError("No out-of-sample model metrics were generated")
    classification_path, regression_path = config.output_dir / "classification_metrics.csv", config.output_dir / "regression_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(classification_path, index=False)
    pd.DataFrame(regression_rows).to_csv(regression_path, index=False)
    manifest_path = config.output_dir / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"classification_metrics": classification_path, "regression_metrics": regression_path, "manifest": manifest_path, "folds": len({row['fold'] for row in metric_rows})}


def run_phase4_from_config(path: str | Path) -> dict[str, Path | int]:
    return run_phase4(load_phase4_config(path))
