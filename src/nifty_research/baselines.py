"""Baselines and matched chronological classification/regression evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClassificationMetric:
    accuracy: float
    balanced_accuracy: float
    precision_positive: float
    recall_positive: float
    precision_negative: float
    recall_negative: float
    roc_auc: float | None
    pr_auc: float | None
    log_loss: float
    brier_score: float
    daily_top_k_precision: float


def _safe_probability(values: pd.Series | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)


def classification_metrics(y_true: pd.Series, scores: np.ndarray, dates: pd.Series, *, top_k: int) -> ClassificationMetric:
    from sklearn.metrics import (  # type: ignore[import-not-found]
        accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
        log_loss, precision_score, recall_score, roc_auc_score,
    )

    y = np.asarray(y_true, dtype=int)
    p = _safe_probability(scores)
    prediction = (p >= 0.5).astype(int)
    roc_auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    pr_auc = float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else None
    daily = pd.DataFrame({"date": pd.to_datetime(dates).to_numpy(), "target": y, "score": p})
    top = daily.sort_values(["date", "score"], ascending=[True, False], kind="stable").groupby("date", sort=False).head(top_k)
    return ClassificationMetric(
        accuracy=float(accuracy_score(y, prediction)),
        balanced_accuracy=float(balanced_accuracy_score(y, prediction)),
        precision_positive=float(precision_score(y, prediction, pos_label=1, zero_division=0)),
        recall_positive=float(recall_score(y, prediction, pos_label=1, zero_division=0)),
        precision_negative=float(precision_score(y, prediction, pos_label=0, zero_division=0)),
        recall_negative=float(recall_score(y, prediction, pos_label=0, zero_division=0)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        log_loss=float(log_loss(y, p, labels=[0, 1])),
        brier_score=float(brier_score_loss(y, p)),
        daily_top_k_precision=float(top.groupby("date", sort=False)["target"].mean().mean()),
    )


def regression_metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error  # type: ignore[import-not-found]

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prediction, dtype=float)
    return {"mae": float(mean_absolute_error(y, p)), "rmse": float(mean_squared_error(y, p) ** 0.5), "mean_prediction_bias": float((p - y).mean())}


def binary_conditional_score(train: pd.DataFrame, evaluation: pd.DataFrame, condition: pd.Series, *, smoothing: float = 20.0) -> np.ndarray:
    """Smoothed historical P(green) conditional on a signal-date observable."""

    base = float(train["target_green"].mean())
    train_condition = condition.loc[train.index]
    positive = train.loc[train_condition, "target_green"]
    negative = train.loc[~train_condition, "target_green"]
    positive_score = (positive.sum() + smoothing * base) / (len(positive) + smoothing)
    negative_score = (negative.sum() + smoothing * base) / (len(negative) + smoothing)
    return np.where(condition.loc[evaluation.index].to_numpy(bool), positive_score, negative_score)


def deterministic_train_sample(train: pd.DataFrame, maximum_rows: int) -> pd.DataFrame:
    """Bound model cost without random splitting or target-dependent sampling."""

    if len(train) <= maximum_rows:
        return train
    identity = train[["symbol", "signal_date"]].astype(str)
    hashes = pd.util.hash_pandas_object(identity, index=False).to_numpy(dtype="uint64")
    positions = np.argpartition(hashes, maximum_rows - 1)[:maximum_rows]
    return train.iloc[np.sort(positions)]


def logistic_scores(train: pd.DataFrame, evaluation: pd.DataFrame, feature_columns: list[str], *, max_train_rows: int, random_state: int) -> tuple[np.ndarray, object]:
    from sklearn.impute import SimpleImputer  # type: ignore[import-not-found]
    from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
    from sklearn.pipeline import Pipeline  # type: ignore[import-not-found]
    from sklearn.preprocessing import StandardScaler  # type: ignore[import-not-found]

    sampled = deterministic_train_sample(train, max_train_rows)
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=300, random_state=random_state)),
    ])
    model.fit(sampled[feature_columns], sampled["target_green"].astype(int))
    return model.predict_proba(evaluation[feature_columns])[:, 1], model


def metric_record(*, fold: str, model: str, metric: ClassificationMetric, observations: int, train_observations: int) -> dict[str, object]:
    return {"fold": fold, "model": model, "observations": observations, "train_observations": train_observations, **asdict(metric)}
