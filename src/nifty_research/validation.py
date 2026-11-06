"""Date-grouped chronological splits for pooled multi-symbol research."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_cutoff: pd.Timestamp
    evaluation_start: pd.Timestamp
    evaluation_end: pd.Timestamp


class DateGroupedExpandingSplitter:
    """Expanding split where every row from a signal date shares a partition."""

    def __init__(self, *, initial_train_end: str, evaluation_start: str, evaluation_end: str, frequency: str = "yearly") -> None:
        if frequency not in {"yearly", "quarterly"}:
            raise ValueError("frequency must be yearly or quarterly")
        self.initial_train_end = pd.Timestamp(initial_train_end).normalize()
        self.evaluation_start = pd.Timestamp(evaluation_start).normalize()
        self.evaluation_end = pd.Timestamp(evaluation_end).normalize()
        self.frequency = frequency
        if self.evaluation_start > self.evaluation_end:
            raise ValueError("evaluation_start must not be after evaluation_end")

    def folds(self, frame: pd.DataFrame) -> list[WalkForwardFold]:
        dates = pd.DatetimeIndex(pd.to_datetime(frame["signal_date"].unique())).sort_values()
        candidates = dates[(dates >= self.evaluation_start) & (dates <= self.evaluation_end)]
        if candidates.empty:
            return []
        periods = candidates.to_period("Y" if self.frequency == "yearly" else "Q")
        folds: list[WalkForwardFold] = []
        for period in periods.unique():
            period_dates = candidates[periods == period]
            eval_start, eval_end = period_dates.min(), period_dates.max()
            prior_dates = dates[dates < eval_start]
            if prior_dates.empty:
                continue
            cutoff = prior_dates.max()
            # ``initial_train_end`` is a calendar boundary; the preceding actual
            # market session can fall before it (for example, 30 Dec before 31 Dec).
            if eval_start <= self.initial_train_end:
                continue
            folds.append(WalkForwardFold(str(period), cutoff, eval_start, eval_end))
        return folds

    @staticmethod
    def masks(frame: pd.DataFrame, fold: WalkForwardFold) -> tuple[pd.Series, pd.Series]:
        signal_date = pd.to_datetime(frame["signal_date"]).dt.normalize()
        target_date = pd.to_datetime(frame["target_date"]).dt.normalize()
        train = signal_date.le(fold.train_cutoff) & target_date.le(fold.train_cutoff)
        # A signal at the edge of a period must not consume a target label from
        # the following period (especially the frozen 2025 holdout).
        evaluation = signal_date.between(fold.evaluation_start, fold.evaluation_end) & target_date.le(fold.evaluation_end)
        return train, evaluation
