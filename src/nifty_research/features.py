"""Feature registry and close-of-signal-date feature engineering."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    formula: str
    lookback: int
    required_inputs: tuple[str, ...]
    availability_timing: str = "available at signal-date close"
    version: str = "1.0.0"
    null_behavior: str = "null during warmup or when required input is unavailable"


def _specs(
    names: Iterable[str],
    group: str,
    formula: str,
    lookback: int,
    inputs: tuple[str, ...],
    *,
    availability_timing: str = "available at signal-date close",
) -> list[FeatureSpec]:
    return [FeatureSpec(name, group, formula, lookback, inputs, availability_timing=availability_timing) for name in names]


def default_feature_registry() -> list[FeatureSpec]:
    """The initial 53-feature registry. It deliberately excludes target columns."""

    registry: list[FeatureSpec] = []
    registry += _specs(
        ["body_to_range", "upper_wick_to_range", "lower_wick_to_range", "close_location", "open_prev_close_gap"],
        "candle_anatomy", "same-day OHLC relative to high-low range and prior close", 1, ("open", "high", "low", "close"),
    )
    registry += _specs(["green_count_5", "red_count_5", "previous_candle_green"], "candle_anatomy", "rolling signal-date candle direction", 5, ("open", "close"))
    registry += _specs([f"return_{day}d" for day in (1, 2, 3, 5, 10, 20)], "momentum", "close / lagged close - 1", 20, ("close",))
    registry += _specs(["rsi_7", "rsi_14"], "momentum", "SMA RSI computed from signal-date close changes", 14, ("close",))
    registry += _specs(["roc_10", "macd_signal_diff", "consecutive_direction_count"], "momentum", "signal-date momentum indicator", 35, ("open", "close"))
    registry += _specs([f"sma_{day}" for day in (5, 10, 20, 50)], "trend", "rolling close mean", 50, ("close",))
    registry += _specs(["ema_10", "ema_20"], "trend", "close exponential moving average", 20, ("close",))
    registry += _specs([f"close_sma_{day}_ratio" for day in (5, 10, 20, 50)], "trend", "close / same-date SMA - 1", 50, ("close",))
    registry += _specs([f"sma_{day}_slope" for day in (5, 10, 20)], "trend", "SMA / lagged SMA - 1", 40, ("close",))
    registry += _specs(["distance_high_20", "distance_high_50", "distance_low_20", "distance_low_50"], "trend", "close distance from rolling extrema", 50, ("high", "low", "close"))
    registry += _specs(["true_range", "atr_7", "atr_14", "atr_close_7", "atr_close_14", "volatility_10", "volatility_20", "range_vs_max_20", "atr_7_to_14"], "volatility", "signal-date range and rolling close-return volatility", 20, ("high", "low", "close"))
    registry += _specs(["volume_sma_20_ratio", "volume_sma_5_ratio", "volume_change_1d", "volume_price_trend"], "volume", "signal-date volume and volume-price trend", 20, ("volume", "close"))
    registry += _specs(["cs_return_5_rank", "cs_return_20_rank"], "relative_strength", "same-date cross-sectional percentile rank", 20, ("close",))
    registry += _specs(
        [
            "market_median_return_1d", "market_median_return_5d", "market_median_return_20d",
            "market_breadth_positive_20d", "market_breadth_above_sma20",
            "market_median_volatility_20", "market_return_20_dispersion",
            "market_median_volume_sma_20_ratio",
        ],
        "market_regime",
        "same-date cross-sectional aggregates across research-eligible universe members, computed after the signal-date close",
        20,
        ("close", "volume", "is_universe_eligible", "is_research_eligible"),
    )
    registry += _specs(["day_of_week", "month"], "calendar", "calendar value known by signal date", 1, ("date",))
    completed_candle_timing = "only the last fully completed higher-timeframe candle, available after its final market session closes"
    registry += _specs(
        ["weekly_return_1w", "weekly_return_4w", "weekly_return_12w", "weekly_rsi_14", "weekly_close_sma_10_ratio", "weekly_close_sma_20_ratio", "weekly_sma_10_slope_4w", "weekly_atr_close_14", "weekly_volume_sma_4_ratio"],
        "higher_timeframe",
        "indicator computed from completed weekly OHLCV bars and as-of joined to each daily signal date",
        100,
        ("open", "high", "low", "close", "volume", "date"),
        availability_timing=completed_candle_timing,
    )
    registry += _specs(
        ["monthly_return_1m", "monthly_return_3m", "monthly_return_12m", "monthly_rsi_12", "monthly_close_sma_6_ratio", "monthly_close_sma_12_ratio", "monthly_sma_6_slope_3m", "monthly_volume_sma_3_ratio"],
        "higher_timeframe",
        "indicator computed from completed monthly OHLCV bars and as-of joined to each daily signal date",
        60,
        ("open", "high", "low", "close", "volume", "date"),
        availability_timing=completed_candle_timing,
    )
    # These features are intentionally optional.  They are emitted only when
    # Phase 2 is supplied with the corresponding verified index OHLC history.
    # An index candle is used only on its own completed trading date, so a
    # stock signal made after that day's close cannot see a later index bar.
    for prefix, description in (
        ("nifty50", "Nifty 50"),
        ("niftybank", "Nifty Bank"),
    ):
        registry += _specs(
            [
                f"{prefix}_return_1d", f"{prefix}_return_5d", f"{prefix}_return_20d", f"{prefix}_return_60d",
                f"{prefix}_rsi_14", f"{prefix}_close_sma_20_ratio", f"{prefix}_close_sma_50_ratio",
                f"{prefix}_sma_20_slope", f"{prefix}_volatility_20", f"{prefix}_atr_close_14",
                f"{prefix}_range_vs_atr14", f"{prefix}_body_to_range", f"{prefix}_open_prev_close_gap",
                f"{prefix}_close_location",
            ],
            "market_index",
            f"completed same-session {description} OHLC indicators, joined by trading date without forward fill",
            60,
            ("date", "open", "high", "low", "close"),
        )
    return registry


def _rolling(frame: pd.DataFrame, column: str, window: int, *, min_periods: int | None = None) -> pd.Series:
    return frame.groupby("symbol", sort=False)[column].transform(lambda s: s.rolling(window, min_periods=min_periods or window).mean())


def _ema(frame: pd.DataFrame, column: str, span: int) -> pd.Series:
    return frame.groupby("symbol", sort=False)[column].transform(lambda s: s.ewm(span=span, adjust=False, min_periods=span).mean())


def _lag(frame: pd.DataFrame, column: str, periods: int = 1) -> pd.Series:
    return frame.groupby("symbol", sort=False)[column].shift(periods)


def _rsi(frame: pd.DataFrame, period: int) -> pd.Series:
    delta = frame.groupby("symbol", sort=False)["close"].diff()
    gains = delta.clip(lower=0)
    losses = (-delta.clip(upper=0))
    avg_gain = gains.groupby(frame["symbol"], sort=False).transform(lambda s: s.rolling(period, min_periods=period).mean())
    avg_loss = losses.groupby(frame["symbol"], sort=False).transform(lambda s: s.rolling(period, min_periods=period).mean())
    relative_strength = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - 100 / (1 + relative_strength)
    return result.mask(avg_loss.eq(0) & avg_gain.gt(0), 100.0).mask(avg_loss.eq(0) & avg_gain.eq(0), 50.0)


def _completed_higher_timeframe_features(frame: pd.DataFrame, *, frequency: str, prefix: str) -> pd.DataFrame:
    """Create completed weekly/monthly bars and safely as-of join them to daily rows.

    A period becomes available only at the *global final market session* in that
    period.  This is deliberately not a direct period-to-daily join: doing that
    would let a Tuesday row see the Friday close of the same week.
    """

    period_column = "_higher_period"
    availability_column = "_higher_available_date"
    sessions = frame[["date"]].drop_duplicates().sort_values("date", kind="stable").copy()
    sessions[period_column] = sessions["date"].dt.to_period(frequency)
    final_session = sessions.groupby(period_column, sort=False)["date"].max()

    source = frame[["symbol", "date", "open", "high", "low", "close", "volume"]].copy()
    source[period_column] = source["date"].dt.to_period(frequency)
    bars = source.groupby(["symbol", period_column], sort=False, as_index=False).agg(
        _period_open=("open", "first"),
        _period_high=("high", "max"),
        _period_low=("low", "min"),
        _period_close=("close", "last"),
        _period_volume=("volume", "sum"),
    )
    bars[availability_column] = bars[period_column].map(final_session)
    bars = bars.sort_values(["symbol", availability_column], kind="stable").reset_index(drop=True)

    close = bars["_period_close"]
    return_periods = (1, 4, 12) if prefix == "weekly" else (1, 3, 12)
    for periods in return_periods:
        suffix = "w" if prefix == "weekly" else "m"
        bars[f"{prefix}_return_{periods}{suffix}"] = close / bars.groupby("symbol", sort=False)["_period_close"].shift(periods) - 1
    rsi_period = 14 if prefix == "weekly" else 12
    bars[f"{prefix}_rsi_{rsi_period}"] = _rsi(bars.rename(columns={"_period_close": "close"}), rsi_period)

    if prefix == "weekly":
        short_window, long_window, slope_lag, volume_window = 10, 20, 4, 4
    else:
        short_window, long_window, slope_lag, volume_window = 6, 12, 3, 3
    short_sma = bars.groupby("symbol", sort=False)["_period_close"].transform(lambda series: series.rolling(short_window, min_periods=short_window).mean())
    long_sma = bars.groupby("symbol", sort=False)["_period_close"].transform(lambda series: series.rolling(long_window, min_periods=long_window).mean())
    bars[f"{prefix}_close_sma_{short_window}_ratio"] = close / short_sma - 1
    bars[f"{prefix}_close_sma_{long_window}_ratio"] = close / long_sma - 1
    bars[f"{prefix}_sma_{short_window}_slope_{slope_lag}{'w' if prefix == 'weekly' else 'm'}"] = short_sma / short_sma.groupby(bars["symbol"], sort=False).shift(slope_lag) - 1
    volume_sma = bars.groupby("symbol", sort=False)["_period_volume"].transform(lambda series: series.rolling(volume_window, min_periods=volume_window).mean())
    bars[f"{prefix}_volume_sma_{volume_window}_ratio"] = bars["_period_volume"] / volume_sma
    if prefix == "weekly":
        previous_close = bars.groupby("symbol", sort=False)["_period_close"].shift()
        true_range = pd.concat([
            bars["_period_high"] - bars["_period_low"],
            (bars["_period_high"] - previous_close).abs(),
            (bars["_period_low"] - previous_close).abs(),
        ], axis=1).max(axis=1)
        atr_14 = true_range.groupby(bars["symbol"], sort=False).transform(lambda series: series.rolling(14, min_periods=14).mean())
        bars["weekly_atr_close_14"] = atr_14 / close

    higher_columns = [column for column in bars.columns if column.startswith(f"{prefix}_")]
    left = frame[["symbol", "date"]].copy()
    left["_row_order"] = np.arange(len(left))
    left = left.sort_values(["date", "symbol"], kind="stable")
    right = bars[["symbol", availability_column, *higher_columns]].sort_values([availability_column, "symbol"], kind="stable")
    joined = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on=availability_column,
        by="symbol",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_row_order", kind="stable")
    return joined[higher_columns].reset_index(drop=True)


def _market_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return causal, same-close market-state features for every symbol/date row.

    Each aggregate uses only values that are known after the current daily bar
    has closed.  It never shifts a later session backward, so it is safe for a
    next-session prediction made after the signal-date close.
    """

    source = frame.loc[
        frame["is_universe_eligible"] & frame["is_research_eligible"],
        ["date", "return_1d", "return_5d", "return_20d", "close_sma_20_ratio", "volatility_20", "volume_sma_20_ratio"],
    ].copy()
    grouped = source.groupby("date", sort=False)
    regime = grouped.agg(
        market_median_return_1d=("return_1d", "median"),
        market_median_return_5d=("return_5d", "median"),
        market_median_return_20d=("return_20d", "median"),
        market_median_volatility_20=("volatility_20", "median"),
        market_return_20_dispersion=("return_20d", lambda values: values.std(ddof=0)),
        market_median_volume_sma_20_ratio=("volume_sma_20_ratio", "median"),
    )
    regime["market_breadth_positive_20d"] = grouped["return_20d"].apply(lambda values: (values.dropna() > 0).mean())
    regime["market_breadth_above_sma20"] = grouped["close_sma_20_ratio"].apply(lambda values: (values.dropna() > 0).mean())
    regime = regime.reset_index()
    columns = [column for column in regime.columns if column != "date"]
    return frame[["date"]].merge(regime, on="date", how="left", sort=False)[columns]


def _market_index_features(index: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    """Calculate causal daily features for one externally supplied market index."""

    required = {"date", "open", "high", "low", "close"}
    missing = required.difference(index.columns)
    if missing:
        raise ValueError(f"Market index {prefix!r} is missing required columns: {sorted(missing)}")
    source = index[["date", "open", "high", "low", "close"]].copy()
    source["date"] = pd.to_datetime(source["date"], errors="raise").dt.normalize()
    source = source.sort_values("date", kind="stable").drop_duplicates("date", keep="last").reset_index(drop=True)
    source["symbol"] = prefix
    previous_close = source["close"].shift()
    candle_range = (source["high"] - source["low"]).replace(0, np.nan)
    true_range = pd.concat([
        source["high"] - source["low"],
        (source["high"] - previous_close).abs(),
        (source["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    returns = source["close"].pct_change()
    output = source[["date"]].copy()
    for days in (1, 5, 20, 60):
        output[f"{prefix}_return_{days}d"] = source["close"] / source["close"].shift(days) - 1
    output[f"{prefix}_rsi_14"] = _rsi(source, 14)
    sma_20 = source["close"].rolling(20, min_periods=20).mean()
    sma_50 = source["close"].rolling(50, min_periods=50).mean()
    output[f"{prefix}_close_sma_20_ratio"] = source["close"] / sma_20 - 1
    output[f"{prefix}_close_sma_50_ratio"] = source["close"] / sma_50 - 1
    output[f"{prefix}_sma_20_slope"] = sma_20 / sma_20.shift(20) - 1
    output[f"{prefix}_volatility_20"] = returns.rolling(20, min_periods=20).std()
    atr_14 = true_range.rolling(14, min_periods=14).mean()
    output[f"{prefix}_atr_close_14"] = atr_14 / source["close"]
    output[f"{prefix}_range_vs_atr14"] = true_range / atr_14
    output[f"{prefix}_body_to_range"] = (source["close"] - source["open"]) / candle_range
    output[f"{prefix}_open_prev_close_gap"] = source["open"] / previous_close - 1
    output[f"{prefix}_close_location"] = (source["close"] - source["low"]) / candle_range
    return output


def build_features(
    ohlcv: pd.DataFrame,
    *,
    enabled_groups: set[str] | None = None,
    market_indices: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, list[FeatureSpec]]:
    """Compute only close-of-date values. No target or forward OHLCV values are read."""

    registry = default_feature_registry()
    # Preserve the historical no-argument behaviour: optional external index
    # features are never silently required by an existing feature build.
    groups = enabled_groups if enabled_groups is not None else {spec.group for spec in registry if spec.group != "market_index"}
    selected = [spec for spec in registry if spec.group in groups]
    required = {"date", "symbol", "open", "high", "low", "close", "volume", "is_universe_eligible", "is_research_eligible"}
    missing = required.difference(ohlcv.columns)
    if missing:
        raise ValueError(f"Normalized dataset missing required feature columns: {sorted(missing)}")
    frame = ohlcv.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame = frame.sort_values(["symbol", "date"], kind="stable").drop_duplicates(["symbol", "date"], keep="last").reset_index(drop=True)
    previous_close = _lag(frame, "close")
    candle_range = (frame["high"] - frame["low"]).replace(0, np.nan)
    body = frame["close"] - frame["open"]
    direction = np.sign(body).astype("int8")
    frame["body_to_range"] = body / candle_range
    frame["upper_wick_to_range"] = (frame["high"] - frame[["open", "close"]].max(axis=1)) / candle_range
    frame["lower_wick_to_range"] = (frame[["open", "close"]].min(axis=1) - frame["low"]) / candle_range
    frame["close_location"] = (frame["close"] - frame["low"]) / candle_range
    frame["open_prev_close_gap"] = frame["open"] / previous_close - 1
    frame["green_count_5"] = _rolling(frame.assign(_green=(body > 0).astype(float)), "_green", 5)
    frame["red_count_5"] = _rolling(frame.assign(_red=(body < 0).astype(float)), "_red", 5)
    frame["previous_candle_green"] = (body > 0).astype("int8")
    for days in (1, 2, 3, 5, 10, 20):
        frame[f"return_{days}d"] = frame["close"] / _lag(frame, "close", days) - 1
    frame["rsi_7"] = _rsi(frame, 7)
    frame["rsi_14"] = _rsi(frame, 14)
    frame["roc_10"] = frame["return_10d"]
    ema_12 = _ema(frame, "close", 12)
    ema_26 = _ema(frame, "close", 26)
    macd = ema_12 - ema_26
    macd_signal = macd.groupby(frame["symbol"], sort=False).transform(lambda s: s.ewm(span=9, adjust=False, min_periods=9).mean())
    frame["macd_signal_diff"] = macd - macd_signal
    frame["_direction"] = direction
    frame["_direction_run"] = frame.groupby("symbol", sort=False)["_direction"].transform(lambda s: s.ne(s.shift()).cumsum())
    frame["consecutive_direction_count"] = frame.groupby(["symbol", "_direction_run"], sort=False).cumcount() + 1
    for days in (5, 10, 20, 50):
        frame[f"sma_{days}"] = _rolling(frame, "close", days)
        frame[f"close_sma_{days}_ratio"] = frame["close"] / frame[f"sma_{days}"] - 1
    for days in (10, 20):
        frame[f"ema_{days}"] = _ema(frame, "close", days)
    for days in (5, 10, 20):
        frame[f"sma_{days}_slope"] = frame[f"sma_{days}"] / _lag(frame, f"sma_{days}", days) - 1
    for days in (20, 50):
        frame[f"distance_high_{days}"] = frame["close"] / frame.groupby("symbol", sort=False)["high"].transform(lambda s: s.rolling(days, min_periods=days).max()) - 1
        frame[f"distance_low_{days}"] = frame["close"] / frame.groupby("symbol", sort=False)["low"].transform(lambda s: s.rolling(days, min_periods=days).min()) - 1
    true_range = pd.concat([frame["high"] - frame["low"], (frame["high"] - previous_close).abs(), (frame["low"] - previous_close).abs()], axis=1).max(axis=1)
    frame["true_range"] = true_range
    frame["atr_7"] = _rolling(frame, "true_range", 7)
    frame["atr_14"] = _rolling(frame, "true_range", 14)
    frame["atr_close_7"] = frame["atr_7"] / frame["close"]
    frame["atr_close_14"] = frame["atr_14"] / frame["close"]
    frame["volatility_10"] = frame.groupby("symbol", sort=False)["return_1d"].transform(lambda s: s.rolling(10, min_periods=10).std())
    frame["volatility_20"] = frame.groupby("symbol", sort=False)["return_1d"].transform(lambda s: s.rolling(20, min_periods=20).std())
    frame["range_vs_max_20"] = true_range / frame.groupby("symbol", sort=False)["true_range"].transform(lambda s: s.rolling(20, min_periods=20).max())
    frame["atr_7_to_14"] = frame["atr_7"] / frame["atr_14"]
    volume_sma_20 = _rolling(frame, "volume", 20)
    volume_sma_5 = _rolling(frame, "volume", 5)
    frame["volume_sma_20_ratio"] = frame["volume"] / volume_sma_20
    frame["volume_sma_5_ratio"] = frame["volume"] / volume_sma_5
    frame["volume_change_1d"] = frame["volume"] / _lag(frame, "volume") - 1
    frame["volume_price_trend"] = (frame["volume"] * frame["return_1d"].fillna(0)).groupby(frame["symbol"], sort=False).cumsum()
    frame["cs_return_5_rank"] = frame.groupby("date", sort=False)["return_5d"].rank(pct=True)
    frame["cs_return_20_rank"] = frame.groupby("date", sort=False)["return_20d"].rank(pct=True)
    market_regime = _market_regime_features(frame)
    frame["day_of_week"] = frame["date"].dt.dayofweek.astype("int8")
    frame["month"] = frame["date"].dt.month.astype("int8")
    weekly = _completed_higher_timeframe_features(frame, frequency="W-FRI", prefix="weekly")
    monthly = _completed_higher_timeframe_features(frame, frequency="M", prefix="monthly")
    frame = pd.concat([frame, market_regime, weekly, monthly], axis=1)
    if "market_index" in groups:
        supplied_indices = market_indices or {}
        expected = {"nifty50", "niftybank"}
        unknown = set(supplied_indices).difference(expected)
        if unknown:
            raise ValueError(f"Unknown market index keys: {sorted(unknown)}; expected only {sorted(expected)}")
        missing_indices = expected.difference(supplied_indices)
        if missing_indices:
            raise ValueError(f"market_index features require supplied histories for: {sorted(missing_indices)}")
        for prefix in sorted(expected):
            index_features = _market_index_features(supplied_indices[prefix], prefix=prefix)
            frame = frame.merge(index_features, on="date", how="left", validate="many_to_one", sort=False)
    feature_names = [spec.name for spec in selected]
    metadata = ["symbol", "date", "is_universe_eligible", "corporate_action_uncertain", "is_research_eligible"]
    output = frame[[*metadata, *feature_names]].copy()
    output["feature_ready"] = output[feature_names].notna().all(axis=1)
    return output, selected


def write_feature_registry(registry: list[FeatureSpec], path: Path) -> None:
    path.write_text(json.dumps([asdict(spec) for spec in registry], indent=2), encoding="utf-8")
