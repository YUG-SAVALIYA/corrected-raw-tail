"""Leakage-safe next-market-session target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_COLUMNS = ["symbol", "signal_date", "target_date", "target_green", "target_return", "is_binary_eligible"]


def make_next_session_labels(
    ohlcv: pd.DataFrame,
    *,
    epsilon: float = 0.0,
    doji_policy: str = "exclude",
) -> pd.DataFrame:
    """Label each signal date using the next *market* session, never a symbol's next observed row.

    A label is unavailable when a symbol lacks a candle on the next actual session.  This
    prevents a data gap from silently becoming a multi-day forecast horizon.
    """

    if doji_policy not in {"exclude", "negative", "positive"}:
        raise ValueError("doji_policy must be one of: exclude, negative, positive")
    required = {"symbol", "date", "open", "close"}
    missing = required.difference(ohlcv.columns)
    if missing:
        raise ValueError(f"OHLCV data missing label inputs: {sorted(missing)}")
    frame = ohlcv[["symbol", "date", "open", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame = frame.sort_values(["symbol", "date"], kind="stable").drop_duplicates(["symbol", "date"], keep="last")
    sessions = pd.DatetimeIndex(frame["date"].unique()).sort_values()
    next_session = dict(zip(sessions[:-1], sessions[1:], strict=True))
    labels = frame.rename(columns={"date": "signal_date"})[["symbol", "signal_date"]].copy()
    labels["target_date"] = labels["signal_date"].map(next_session)
    next_bars = frame.rename(columns={"date": "target_date", "open": "target_open", "close": "target_close"})
    labels = labels.merge(next_bars, how="left", on=["symbol", "target_date"], validate="one_to_one")
    labels["target_return"] = labels["target_close"] / labels["target_open"] - 1.0
    labels["target_green"] = pd.Series(pd.NA, index=labels.index, dtype="Int8")
    available = labels["target_return"].notna()
    labels.loc[available & labels["target_return"].gt(epsilon), "target_green"] = 1
    labels.loc[available & labels["target_return"].lt(-epsilon), "target_green"] = 0
    doji = available & labels["target_return"].sub(epsilon).eq(0)
    if doji_policy == "negative":
        labels.loc[doji, "target_green"] = 0
    elif doji_policy == "positive":
        labels.loc[doji, "target_green"] = 1
    labels["is_binary_eligible"] = labels["target_green"].notna()
    if doji_policy == "exclude":
        labels.loc[doji, "is_binary_eligible"] = False
    labels = labels.drop(columns=["target_open", "target_close"])
    return labels[TARGET_COLUMNS].sort_values(["symbol", "signal_date"], kind="stable").reset_index(drop=True)
