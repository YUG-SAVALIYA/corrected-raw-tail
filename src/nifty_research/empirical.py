"""Chronological empirical classifier comparison for held-out market periods."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field

from .baselines import classification_metrics, deterministic_train_sample, metric_record
from .phase4 import _apply_calibrator, _choose_calibrator, _load, _partition_train, _split_calibration
from .validation import DateGroupedExpandingSplitter


class EmpiricalConfig(BaseModel):
    features_path: Path
    labels_path: Path
    output_dir: Path = Path("artifacts/nifty_empirical")
    initial_train_end: str = "2022-12-31"
    evaluation_start: str = "2025-01-01"
    evaluation_end: str = "2026-12-31"
    retrain_frequency: str = "yearly"
    calibration_months: int = Field(default=6, ge=2)
    early_stopping_months: int = Field(default=3, ge=1)
    max_train_rows: int = Field(default=200_000, ge=10_000)
    max_calibration_rows: int = Field(default=150_000, ge=10_000)
    random_state: int = 42
    n_jobs: int = 4
    # HGB controls are explicit so feature experiments can tune complexity on
    # historical folds without editing source code or changing the frozen
    # production recipe.
    hgb_learning_rate: float = Field(default=0.05, gt=0, le=1)
    hgb_max_iter: int = Field(default=200, ge=10)
    hgb_max_leaf_nodes: int = Field(default=31, ge=2)
    hgb_l2_regularization: float = Field(default=5.0, ge=0)
    hgb_min_samples_leaf: int = Field(default=20, ge=1)
    # A prefix removes a feature family after the shared feature artifact has
    # been loaded.  This keeps the experiment's labels and point-in-time
    # eligibility identical while allowing causal ablation studies.
    excluded_feature_prefixes: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=lambda: ["logistic_regression", "hist_gradient_boosting"])
    emit_ensemble: bool = True
    # When enabled, fit a separate next-session open-to-close return regressor
    # alongside each supported probability classifier.  It is used only as an
    # additional ranking/filtering signal; the classifier remains the source of
    # the calibrated green-day probability.
    fit_return_regressor: bool = False
    # Train a separate downside-event model rather than infer short conviction
    # by reversing a green-day classifier.
    classification_target: str = "target_green"
    short_return_threshold: float = Field(default=-0.012, lt=0)
    long_return_threshold: float = Field(default=0.012, gt=0)
    # These are derived only from the preceding calibration block.  They make
    # short entry confidence comparable when yearly calibrated score scales
    # differ.
    short_calibration_quantiles: list[float] = Field(default_factory=list)
    long_calibration_quantiles: list[float] = Field(default_factory=list)


def load_empirical_config(path: str | Path) -> EmpiricalConfig:
    config_path = Path(path).resolve()
    config = EmpiricalConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    for field in ("features_path", "labels_path", "output_dir"):
        value = getattr(config, field)
        if not value.is_absolute():
            setattr(config, field, (config_path.parent / value).resolve())
    return config


def _classifiers(config: EmpiricalConfig) -> dict[str, object]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
    from xgboost import XGBClassifier

    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=0.1, max_iter=300, random_state=config.random_state)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", HistGradientBoostingClassifier(
                learning_rate=config.hgb_learning_rate,
                max_iter=config.hgb_max_iter,
                max_leaf_nodes=config.hgb_max_leaf_nodes,
                l2_regularization=config.hgb_l2_regularization,
                min_samples_leaf=config.hgb_min_samples_leaf,
                early_stopping=False,
                random_state=config.random_state,
            )),
        ]),
        "xgboost_classifier": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_estimators=1_200,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            reg_alpha=0.1,
            early_stopping_rounds=75,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        ),
        "linear_svm": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(C=0.01, random_state=config.random_state, max_iter=5_000)),
        ]),
    }


def _probabilities(classifier: object, values: pd.DataFrame) -> np.ndarray:
    """Use native probabilities, or a sigmoid score for LinearSVC calibration."""

    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(values)[:, 1]
    from scipy.special import expit
    return expit(classifier.decision_function(values))


def _return_regressor(name: str, config: EmpiricalConfig) -> object:
    """Return the return-magnitude model paired with a tree classifier."""

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from xgboost import XGBRegressor

    if name == "hist_gradient_boosting":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("regressor", HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=31,
                l2_regularization=5.0,
                early_stopping=False,
                random_state=config.random_state,
            )),
        ])
    if name == "xgboost_classifier":
        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            n_estimators=1_200,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=100,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            reg_alpha=0.1,
            early_stopping_rounds=75,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )
    raise ValueError(f"Return regression is supported only for tree classifiers, not {name!r}")


def _return_metric_record(evaluation: pd.DataFrame, predicted_return: np.ndarray, *, fold: str, model: str, train_observations: int) -> dict[str, object]:
    """Report magnitude quality separately from the green-day classifier metrics."""

    target = evaluation["target_return"].to_numpy(float)
    error = predicted_return - target
    correlation = float(np.corrcoef(predicted_return, target)[0, 1]) if len(target) > 1 and np.std(predicted_return) and np.std(target) else 0.0
    return {
        "fold": fold,
        "model": model,
        "observations": len(evaluation),
        "train_observations": train_observations,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "pearson_correlation": correlation,
        "directional_accuracy": float(((predicted_return > 0) == (target > 0)).mean()),
        "mean_predicted_return": float(np.mean(predicted_return)),
        "mean_realized_return": float(np.mean(target)),
    }


def _prediction_frame(evaluation: pd.DataFrame, probability: np.ndarray, name: str) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": evaluation["symbol"].to_numpy(),
        "signal_date": evaluation["signal_date"].to_numpy(),
        "target_date": evaluation["target_date"].to_numpy(),
        "target_green": evaluation["target_green"].astype("Int8").to_numpy(),
        "target_return": evaluation["target_return"].to_numpy(),
        "raw_probability": probability,
        "calibrated_probability": probability,
        # This classifier comparison deliberately ranks probability only.  The
        # column remains for compatibility with the shared backtest loader.
        "predicted_return": np.zeros(len(evaluation), dtype=float),
        "volatility_20": evaluation["volatility_20"].to_numpy(),
        "model": name,
    })


def _prepare_target(frame: pd.DataFrame, config: EmpiricalConfig) -> tuple[pd.DataFrame, str]:
    """Create the selected target from the already-mature next-session label."""

    if config.classification_target == "target_green":
        return frame, "target_green"
    if config.classification_target not in {"short_return_threshold", "long_return_threshold"}:
        raise ValueError("classification_target must be 'target_green', 'short_return_threshold', or 'long_return_threshold'")
    prepared = frame.copy()
    available = prepared["target_return"].notna()
    if config.classification_target == "short_return_threshold":
        target_column = "target_short_event"
        event = prepared.loc[available, "target_return"].le(config.short_return_threshold)
    else:
        target_column = "target_long_event"
        event = prepared.loc[available, "target_return"].ge(config.long_return_threshold)
    prepared[target_column] = pd.Series(pd.NA, index=prepared.index, dtype="Int8")
    prepared.loc[available, target_column] = event.astype("Int8")
    prepared["is_binary_eligible"] = available
    return prepared, target_column


def _fit_classifier(classifier: object, name: str, train: pd.DataFrame, feature_columns: list[str], target_column: str, early_stop: pd.DataFrame) -> None:
    """Fit a classifier, balancing the rare direct downside-event target."""

    target = train[target_column].astype(int)
    if target_column in {"target_short_event", "target_long_event"}:
        positives = max(int(target.sum()), 1)
        negatives = max(len(target) - positives, 1)
        weights = np.where(target.to_numpy(bool), len(target) / (2 * positives), len(target) / (2 * negatives))
        if name == "xgboost_classifier":
            classifier.fit(train[feature_columns], target, sample_weight=weights, eval_set=[(early_stop[feature_columns], early_stop[target_column].astype(int))], verbose=False)
        else:
            classifier.fit(train[feature_columns], target, classifier__sample_weight=weights)
        return
    if name == "xgboost_classifier":
        classifier.fit(train[feature_columns], target, eval_set=[(early_stop[feature_columns], early_stop[target_column].astype(int))], verbose=False)
    else:
        classifier.fit(train[feature_columns], target)


def run_empirical(config: EmpiricalConfig) -> dict[str, Path | int]:
    frame, feature_columns = _load(config)
    if config.excluded_feature_prefixes:
        selected = [
            column for column in feature_columns
            if not any(column.startswith(prefix) for prefix in config.excluded_feature_prefixes)
        ]
        if len(selected) == len(feature_columns):
            raise ValueError(f"excluded_feature_prefixes did not match any model feature: {config.excluded_feature_prefixes}")
        if not selected:
            raise ValueError("excluded_feature_prefixes removed every model feature")
        feature_columns = selected
    frame, target_column = _prepare_target(frame, config)
    # Indicators that divide by a prior range/volume can be infinite during an
    # early market-history warm-up.  The estimators already use median
    # imputation, so represent these unavailable values as missing rather than
    # passing invalid numeric values into sklearn.
    frame.loc[:, feature_columns] = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
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
    metric_rows: list[dict[str, object]] = []
    return_metric_rows: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    for fold in splitter.folds(frame):
        train, early_stop, calibration = _partition_train(frame, fold, config)
        train = deterministic_train_sample(train, config.max_train_rows)
        early_stop = deterministic_train_sample(early_stop, min(config.max_calibration_rows, len(early_stop)))
        calibration = deterministic_train_sample(calibration, min(config.max_calibration_rows, len(calibration)))
        calibration_fit, calibration_select = _split_calibration(calibration)
        eligible_eval = frame["feature_ready"] & frame["is_research_eligible"]
        eval_mask = frame["signal_date"].between(fold.evaluation_start, fold.evaluation_end) & frame["target_date"].le(fold.evaluation_end)
        evaluation = frame.loc[eligible_eval & eval_mask].copy()
        if evaluation.empty:
            continue
        evaluation_target = evaluation.loc[evaluation["is_binary_eligible"]].copy()
        usable = evaluation["is_binary_eligible"].to_numpy(bool)
        calibrated_outputs: dict[str, np.ndarray] = {}
        available = _classifiers(config)
        unknown = set(config.models).difference(available)
        if unknown:
            raise ValueError(f"Unknown empirical models: {sorted(unknown)}")
        for name in config.models:
            classifier = available[name]
            _fit_classifier(classifier, name, train, feature_columns, target_column, early_stop)
            raw_fit = _probabilities(classifier, calibration_fit[feature_columns])
            raw_select = _probabilities(classifier, calibration_select[feature_columns])
            method, calibrator, comparison = _choose_calibrator(raw_fit, calibration_fit[target_column], raw_select, calibration_select[target_column])
            raw_eval = _probabilities(classifier, evaluation[feature_columns])
            calibrated = _apply_calibrator(calibrator, raw_eval)
            calibrated_outputs[name] = calibrated
            predicted_return = np.zeros(len(evaluation), dtype=float)
            regressor = None
            if config.fit_return_regressor:
                regressor = _return_regressor(name, config)
                if name == "xgboost_classifier":
                    regressor.fit(
                        train[feature_columns],
                        train["target_return"],
                        eval_set=[(early_stop[feature_columns], early_stop["target_return"])],
                        verbose=False,
                    )
                else:
                    regressor.fit(train[feature_columns], train["target_return"])
                predicted_return = np.asarray(regressor.predict(evaluation[feature_columns]), dtype=float)
                return_metric_rows.append(_return_metric_record(
                    evaluation.loc[usable].copy(), predicted_return[usable], fold=fold.name, model=name, train_observations=len(train),
                ))
            metric = classification_metrics(evaluation_target[target_column], calibrated[usable], evaluation_target["signal_date"], top_k=5)
            metric_rows.append(metric_record(fold=fold.name, model=name, metric=metric, observations=len(evaluation_target), train_observations=len(train)))
            prediction_path = prediction_dir / f"{name}_{fold.name}.parquet"
            prediction = _prediction_frame(evaluation, calibrated, name)
            if target_column in {"target_short_event", "target_long_event"}:
                event_name = "short" if target_column == "target_short_event" else "long"
                threshold = config.short_return_threshold if event_name == "short" else config.long_return_threshold
                prediction["target_short_event"] = evaluation[target_column].astype("Int8").to_numpy()
                prediction = prediction.rename(columns={"target_short_event": target_column})
                prediction[f"raw_{event_name}_probability"] = raw_eval
                prediction[f"calibrated_{event_name}_probability"] = calibrated
                # Retain the usual column solely for shared-file compatibility.
                # A direct-short backtest reads its dedicated column directly;
                # direct-long probabilities already use the usual orientation.
                prediction["calibrated_probability"] = 1 - calibrated if event_name == "short" else calibrated
                operator = "<=" if event_name == "short" else ">="
                prediction["prediction_target"] = f"return{operator}{threshold:.4f}"
                raw_calibration_scores = _probabilities(classifier, calibration[feature_columns])
                calibration_scores = _apply_calibrator(calibrator, raw_calibration_scores)
                quantiles = config.short_calibration_quantiles if event_name == "short" else config.long_calibration_quantiles
                for quantile in quantiles:
                    if not 0 < quantile < 1:
                        raise ValueError(f"{event_name}_calibration_quantiles must contain values strictly between 0 and 1")
                    column = f"{event_name}_probability_calibration_q{int(round(quantile * 10_000)):04d}"
                    prediction[column] = float(np.quantile(calibration_scores, quantile))
                    # Isotonic calibration is intentionally piecewise constant.
                    # It is appropriate for probability reporting, but its
                    # tied plateaus cannot safely define an extreme-tail
                    # selection rule. Persist a continuous pre-calibration
                    # threshold so the backtest cannot admit every tied row.
                    raw_column = f"{event_name}_raw_probability_calibration_q{int(round(quantile * 10_000)):04d}"
                    prediction[raw_column] = float(np.quantile(raw_calibration_scores, quantile))
            prediction["predicted_return"] = predicted_return
            prediction.to_parquet(prediction_path, index=False)
            artifact_path = model_dir / f"{name}_{fold.name}.joblib"
            joblib.dump({"model": classifier, "calibrator": calibrator, "return_regressor": regressor, "feature_order": feature_columns, "training_cutoff": str(fold.train_cutoff.date()), "classification_target": target_column, "short_return_threshold": config.short_return_threshold if target_column == "target_short_event" else None, "long_return_threshold": config.long_return_threshold if target_column == "target_long_event" else None, "calibration_method": method, "calibration_comparison": comparison}, artifact_path)
            manifest.append({"fold": fold.name, "model": name, "prediction_path": str(prediction_path), "artifact_path": str(artifact_path), "classification_target": target_column, "calibration_method": method, "calibration_comparison": comparison})
        # An ensemble is only meaningful when more than one component model was
        # deliberately selected for the experiment.
        if config.emit_ensemble and len(calibrated_outputs) > 1:
            ensemble = np.mean(np.column_stack(list(calibrated_outputs.values())), axis=1)
            name = "empirical_probability_ensemble"
            metric = classification_metrics(evaluation_target["target_green"], ensemble[usable], evaluation_target["signal_date"], top_k=5)
            metric_rows.append(metric_record(fold=fold.name, model=name, metric=metric, observations=len(evaluation_target), train_observations=len(train)))
            prediction_path = prediction_dir / f"{name}_{fold.name}.parquet"
            _prediction_frame(evaluation, ensemble, name).to_parquet(prediction_path, index=False)
            manifest.append({"fold": fold.name, "model": name, "prediction_path": str(prediction_path), "artifact_path": None, "calibration_method": "mean_of_calibrated_components"})
    if not metric_rows:
        raise ValueError("No empirical out-of-sample model metrics were generated")
    metrics_path = config.output_dir / "classification_metrics.csv"
    return_metrics_path = config.output_dir / "return_prediction_metrics.csv"
    manifest_path = config.output_dir / "model_manifest.json"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    if return_metric_rows:
        pd.DataFrame(return_metric_rows).to_csv(return_metrics_path, index=False)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    result: dict[str, Path | int] = {"classification_metrics": metrics_path, "manifest": manifest_path, "folds": len({row["fold"] for row in metric_rows})}
    if return_metric_rows:
        result["return_prediction_metrics"] = return_metrics_path
    return result


def run_empirical_from_config(path: str | Path) -> dict[str, Path | int]:
    return run_empirical(load_empirical_config(path))
