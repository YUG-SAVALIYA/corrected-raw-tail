"""Long-only, one-day open-to-close execution simulator with explicit cost tables."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field


class FeePeriod(BaseModel):
    effective_from: str
    effective_to: str | None = None
    brokerage_rate_per_side: float = 0.0003
    exchange_transaction_rate_per_side: float
    sebi_rate_per_side: float
    gst_rate: float
    stt_sell_rate: float = 0.00025
    stamp_buy_rate: float
    status: str = "estimated"
    notes: str = ""


class BacktestConfig(BaseModel):
    # A list supports a contiguous walk-forward run assembled from separately
    # trained year blocks without copying or altering the prediction artifacts.
    predictions_glob: str | list[str]
    normalized_path: Path
    fee_schedule_path: Path
    output_dir: Path = Path("artifacts/nifty_phase5")
    strategies: list[str] = Field(default_factory=lambda: ["top5_probability", "top5_expected_return", "hybrid"])
    slippage_bps_per_side: list[float] = Field(default_factory=lambda: [0.0, 5.0, 10.0])
    initial_capital: float = Field(default=1_000_000.0, gt=0)
    # Production/default strategies remain capped at five; Phase 6 uses up to
    # ten only as an explicitly labelled sensitivity experiment.
    max_positions: int = Field(default=5, ge=1, le=10)
    # Optional side-specific caps for the combined strategy.  They let a
    # portfolio reserve (for example) one slot for a rarer short model while
    # keeping ``max_positions`` as the hard total limit.
    max_long_positions: int | None = Field(default=None, ge=1, le=10)
    max_short_positions: int | None = Field(default=None, ge=1, le=10)
    max_allocation_per_stock: float = Field(default=0.20, gt=0, le=1)
    max_turnover_fraction_per_stock: float = Field(default=0.02, gt=0, le=1)
    min_signal_day_turnover: float = Field(default=5_000_000.0, ge=0)
    # A conservative default keeps the strategy out of marginal signals.  The
    # value is a research default, not a promise of live-market performance.
    min_probability: float = Field(default=0.60, ge=0, le=1)
    min_predicted_return: float = -1.0
    # A direct-long event model already predicts a return threshold.  It can
    # select on that calibrated probability alone, just like the direct-short
    # model below.
    require_long_return_prediction: bool = True
    long_probability_threshold_column: str | None = None
    long_volatility_rank_penalty: float = Field(default=1.0, ge=0, le=2)
    # The model is trained against ``target_green``.  For a short decision we
    # expose the equivalent, more intuitive quantity: P(red) = 1 - P(green).
    # ``max_short_probability`` remains only so existing research YAML files
    # reproduce their historical results unchanged.
    min_short_red_probability: float | None = Field(default=None, ge=0, le=1)
    max_short_probability: float = Field(default=0.40, ge=0, le=1)
    max_predicted_short_return: float = Field(default=-0.01, lt=0)
    # Dedicated downside-event models already predict a return threshold, so
    # they may select on that direct probability alone.
    require_short_return_prediction: bool = True
    # Optional per-row threshold computed only from the historical calibration
    # block that preceded the prediction year.
    short_probability_threshold_column: str | None = None
    # Score used for strict direct-short thresholding and within-day ranking.
    # Leave unset to preserve historical calibrated-probability behaviour.
    # A continuous raw model score avoids expanding an isotonic-calibration tie
    # into many apparent top-percentile signals.
    short_signal_score_column: str | None = None
    # Optional causal signal-day filter.  It avoids applying an intraday short
    # rule to unusually quiet stocks, where its historical trade quality was
    # weak.  This is stock 20-session volatility, not future target-day data.
    min_short_volatility_20: float | None = Field(default=None, gt=0)
    # How strongly to penalize volatile names after they have passed the
    # minimum-volatility filter.  ``1.0`` preserves the original ranking;
    # ``0.0`` ranks the dedicated downside model purely by its calibrated
    # probability.  This changes ranking only, never uses target-day data.
    short_volatility_rank_penalty: float = Field(default=1.0, ge=0, le=2)
    # Median of the already-completed 20-session stock volatilities on the
    # signal day.  When set, the simulator does not open new positions above
    # this market-regime ceiling.  It is a causal, cross-sectional proxy rather
    # than a future or intraday volatility measurement.
    market_volatility_ceiling: float | None = Field(default=None, gt=0)
    # Optional deleveraging rule.  Above this causal signal-day volatility
    # trigger, retain the selected symbols but use this smaller allocation per
    # stock.  This can be used instead of, or alongside, a hard regime skip.
    market_volatility_allocation_trigger: float | None = Field(default=None, gt=0)
    high_volatility_allocation_per_stock: float | None = Field(default=None, gt=0, le=1)
    # Daily bars can only establish that the intraday low crossed this level;
    # they cannot reconstruct queue position, a gap-through fill, or order
    # sequencing.  Leave unset for the close-exit baseline.
    hard_stop_loss_pct: float | None = Field(default=None, gt=0, lt=1)
    # Optional behavioural cooldown: after this many consecutive signal days
    # with at least one *executed* position, skip new positions for the next
    # configured number of market signal days.  Disabled by default.
    max_consecutive_trade_days: int | None = Field(default=None, ge=1)
    cooldown_days_after_streak: int = Field(default=0, ge=0)
    # Optional causal external-index gate.  The selected index candle is
    # joined strictly on the signal date after that session has completed.
    market_index_filter: str | None = None
    market_index_paths: dict[str, Path] = Field(default_factory=dict)


def load_backtest_config(path: str | Path) -> BacktestConfig:
    config_path = Path(path).resolve()
    config = BacktestConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    for field in ("normalized_path", "fee_schedule_path", "output_dir"):
        value = getattr(config, field)
        if not value.is_absolute():
            setattr(config, field, (config_path.parent / value).resolve())
    globs = config.predictions_glob if isinstance(config.predictions_glob, list) else [config.predictions_glob]
    resolved_globs = [str(Path(item).resolve()) if Path(item).is_absolute() else str((config_path.parent / item).resolve()) for item in globs]
    config.predictions_glob = resolved_globs if isinstance(config.predictions_glob, list) else resolved_globs[0]
    config.market_index_paths = {
        name: value if value.is_absolute() else (config_path.parent / value).resolve()
        for name, value in config.market_index_paths.items()
    }
    return config


def load_fee_schedule(path: Path) -> list[FeePeriod]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    periods = [FeePeriod.model_validate(item) for item in payload.get("periods", [])]
    if not periods:
        raise ValueError("Fee schedule has no periods")
    return sorted(periods, key=lambda item: item.effective_from)


def fee_for_date(periods: list[FeePeriod], date: pd.Timestamp) -> FeePeriod:
    day = pd.Timestamp(date).normalize()
    for period in periods:
        start = pd.Timestamp(period.effective_from)
        end = pd.Timestamp(period.effective_to) if period.effective_to else pd.Timestamp.max
        if start <= day <= end:
            return period
    raise ValueError(f"No fee schedule covers {day.date()}")


def _rates(fee: FeePeriod) -> tuple[float, float]:
    taxable_buy = fee.brokerage_rate_per_side + fee.exchange_transaction_rate_per_side + fee.sebi_rate_per_side
    buy_rate = taxable_buy * (1 + fee.gst_rate) + fee.stamp_buy_rate
    taxable_sell = fee.brokerage_rate_per_side + fee.exchange_transaction_rate_per_side + fee.sebi_rate_per_side
    sell_rate = taxable_sell * (1 + fee.gst_rate) + fee.stt_sell_rate
    return buy_rate, sell_rate


def transaction_costs(buy_value: float, sell_value: float, fee: FeePeriod) -> dict[str, float]:
    buy_taxable = buy_value * (fee.brokerage_rate_per_side + fee.exchange_transaction_rate_per_side + fee.sebi_rate_per_side)
    sell_taxable = sell_value * (fee.brokerage_rate_per_side + fee.exchange_transaction_rate_per_side + fee.sebi_rate_per_side)
    brokerage = (buy_value + sell_value) * fee.brokerage_rate_per_side
    exchange = (buy_value + sell_value) * fee.exchange_transaction_rate_per_side
    sebi = (buy_value + sell_value) * fee.sebi_rate_per_side
    gst = (buy_taxable + sell_taxable) * fee.gst_rate
    stt = sell_value * fee.stt_sell_rate
    stamp = buy_value * fee.stamp_buy_rate
    return {"brokerage": brokerage, "exchange_transaction_charge": exchange, "sebi_fee": sebi, "gst": gst, "stt": stt, "stamp_duty": stamp, "total_cost": brokerage + exchange + sebi + gst + stt + stamp}


def _rank_candidates(candidates: pd.DataFrame, strategy: str, config: BacktestConfig) -> pd.DataFrame:
    candidates = candidates.loc[candidates["signal_day_turnover"].ge(config.min_signal_day_turnover)].copy()

    def long_hybrid() -> pd.DataFrame:
        mask = candidates["calibrated_probability"].ge(config.min_probability)
        if config.long_probability_threshold_column is not None:
            column = config.long_probability_threshold_column
            if column not in candidates:
                raise ValueError(f"Long probability threshold column is missing: {column}")
            mask &= candidates["calibrated_probability"].ge(candidates[column])
        if config.require_long_return_prediction:
            mask &= candidates["predicted_return"].ge(config.min_predicted_return)
        selected = candidates.loc[mask].copy()
        return_score = selected["predicted_return"].rank(pct=True) if config.require_long_return_prediction else 0.0
        selected["selection_score"] = (
            selected["calibrated_probability"].rank(pct=True)
            + return_score
            - config.long_volatility_rank_penalty * selected["volatility_20"].rank(pct=True)
        )
        selected["side"] = "long"
        return selected

    def short_hybrid() -> pd.DataFrame:
        short_candidates = candidates.copy()
        if "calibrated_short_probability" in short_candidates:
            short_candidates["calibrated_red_probability"] = short_candidates["calibrated_short_probability"]
        else:
            short_candidates["calibrated_red_probability"] = 1 - short_candidates["calibrated_probability"]
        min_red_probability = (
            config.min_short_red_probability
            if config.min_short_red_probability is not None
            else 1 - config.max_short_probability
        )
        mask = short_candidates["calibrated_red_probability"].ge(min_red_probability)
        signal_score = short_candidates["calibrated_red_probability"]
        if config.short_signal_score_column is not None:
            score_column = config.short_signal_score_column
            if score_column not in short_candidates:
                raise ValueError(f"Short signal-score column is missing: {score_column}")
            signal_score = short_candidates[score_column]
        if config.short_probability_threshold_column is not None:
            column = config.short_probability_threshold_column
            if column not in short_candidates:
                raise ValueError(f"Short probability threshold column is missing: {column}")
            mask &= signal_score.ge(short_candidates[column])
        if config.require_short_return_prediction:
            mask &= short_candidates["predicted_return"].le(config.max_predicted_short_return)
        selected = short_candidates.loc[mask].copy()
        if config.min_short_volatility_20 is not None:
            selected = selected.loc[selected["volatility_20"].ge(config.min_short_volatility_20)].copy()
        return_score = (-selected["predicted_return"]).rank(pct=True) if config.require_short_return_prediction else 0.0
        selected["selection_score"] = (
            signal_score.loc[selected.index].rank(pct=True)
            + return_score
            - config.short_volatility_rank_penalty * selected["volatility_20"].rank(pct=True)
        )
        selected["side"] = "short"
        return selected

    if strategy == "top5_probability":
        candidates = candidates.loc[candidates["calibrated_probability"].ge(config.min_probability)]
        candidates["selection_score"] = candidates["calibrated_probability"]
        candidates["side"] = "long"
    elif strategy == "top5_expected_return":
        candidates = candidates.loc[candidates["predicted_return"].ge(config.min_predicted_return)]
        candidates["selection_score"] = candidates["predicted_return"]
        candidates["side"] = "long"
    elif strategy == "hybrid":
        candidates = long_hybrid()
    elif strategy == "hybrid_short":
        candidates = short_hybrid()
    elif strategy == "hybrid_long_short":
        long_candidates = long_hybrid().sort_values(["selection_score", "symbol"], ascending=[False, True], kind="stable")
        short_candidates = short_hybrid().sort_values(["selection_score", "symbol"], ascending=[False, True], kind="stable")
        # A long-model score and a direct-downside-model score are not the same
        # physical quantity, so choose the permitted candidates within each
        # side before applying the portfolio's hard total cap.
        long_limit = config.max_long_positions or config.max_positions
        short_limit = config.max_short_positions or config.max_positions
        candidates = pd.concat([long_candidates.head(long_limit), short_candidates.head(short_limit)], ignore_index=True)
    elif strategy == "priority_hybrid_short":
        required = {"short_preselected", "signal_source_priority", "source_score", "signal_source"}
        missing = required.difference(candidates.columns)
        if missing:
            raise ValueError(f"priority_hybrid_short requires columns: {sorted(missing)}")
        # Candidate eligibility was calculated separately for each model using
        # its own past-calibration threshold.  Here we only apply the requested
        # portfolio rule: baseline source first, then fill unused slots from
        # the secondary source.  Deduplication prevents two models shorting the
        # same stock on the same signal date.
        candidates = candidates.loc[candidates["short_preselected"]].copy()
        candidates["selection_score"] = candidates["source_score"]
        candidates["side"] = "short"
        candidates = candidates.sort_values(
            ["signal_source_priority", "selection_score", "symbol"],
            ascending=[True, False, True],
            kind="stable",
        ).drop_duplicates("symbol", keep="first")
        return candidates.head(config.max_positions).copy()
    elif strategy == "priority_hybrid_long_short":
        required = {"preselected", "signal_source_priority", "source_score", "signal_source", "side"}
        missing = required.difference(candidates.columns)
        if missing:
            raise ValueError(f"priority_hybrid_long_short requires columns: {sorted(missing)}")
        # Each source applies its own causal model threshold before this point.
        # Source scores are not comparable across the long and short models, so
        # the declared source priority is the portfolio tie-breaker; a source's
        # score ranks candidates only within that source.  Keeping the first
        # duplicate prevents simultaneously opening opposite sides of a symbol.
        candidates = candidates.loc[candidates["preselected"]].copy()
        candidates["selection_score"] = candidates["source_score"]
        candidates = candidates.sort_values(
            ["signal_source_priority", "selection_score", "symbol"],
            ascending=[True, False, True],
            kind="stable",
        ).drop_duplicates("symbol", keep="first")
        return candidates.head(config.max_positions).copy()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    return candidates.sort_values(["selection_score", "symbol"], ascending=[False, True], kind="stable").head(config.max_positions)


def simulate(candidates: pd.DataFrame, periods: list[FeePeriod], config: BacktestConfig, *, strategy: str, slippage_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    capital = config.initial_capital
    trades: list[dict] = []
    daily: list[dict] = []
    slippage = slippage_bps / 10_000
    consecutive_trade_days = 0
    cooldown_days_remaining = 0
    for signal_date, day in candidates.groupby("signal_date", sort=True):
        market_volatility = float(day["volatility_20"].median())
        regime_blocked = bool(
            config.market_volatility_ceiling is not None
            and market_volatility > config.market_volatility_ceiling
        )
        regime_deleveraged = bool(
            config.market_volatility_allocation_trigger is not None
            and config.high_volatility_allocation_per_stock is not None
            and market_volatility > config.market_volatility_allocation_trigger
        )
        allocation_per_stock = (
            config.high_volatility_allocation_per_stock
            if regime_deleveraged else config.max_allocation_per_stock
        )
        cooldown_active = cooldown_days_remaining > 0
        if cooldown_active:
            selected = day.iloc[0:0].copy()
            cooldown_days_remaining -= 1
        else:
            selected = day.iloc[0:0].copy() if regime_blocked else _rank_candidates(day, strategy, config)
        start_capital = capital
        daily_pnl = 0.0
        trade_count_before = len(trades)
        for row in selected.itertuples(index=False):
            fee = fee_for_date(periods, row.target_date)
            buy_rate, _ = _rates(fee)
            # The trade day's realised turnover is unavailable at the opening
            # order.  Cap the order using only the prior, fully completed
            # signal day's turnover so sizing remains executable without
            # forward information.
            liquidity_cap = row.signal_day_turnover * config.max_turnover_fraction_per_stock
            cash_budget = min(start_capital * allocation_per_stock, liquidity_cap)
            if cash_budget <= 0:
                continue
            side = getattr(row, "side", "long")
            if side == "long":
                entry_price = row.target_open * (1 + slippage)
                stop_price = entry_price * (1 - config.hard_stop_loss_pct) if config.hard_stop_loss_pct is not None else np.nan
                stop_triggered = bool(config.hard_stop_loss_pct is not None and getattr(row, "target_low", np.inf) <= stop_price)
                raw_exit_price = stop_price if stop_triggered else row.target_close
                exit_price = raw_exit_price * (1 - slippage)
                gross_return = raw_exit_price / row.target_open - 1
            else:
                entry_price = row.target_open * (1 - slippage)
                stop_price = entry_price * (1 + config.hard_stop_loss_pct) if config.hard_stop_loss_pct is not None else np.nan
                stop_triggered = bool(config.hard_stop_loss_pct is not None and getattr(row, "target_high", -np.inf) >= stop_price)
                raw_exit_price = stop_price if stop_triggered else row.target_close
                exit_price = raw_exit_price * (1 + slippage)
                gross_return = row.target_open / raw_exit_price - 1
            quantity = cash_budget / (entry_price * (1 + buy_rate))
            buy_value, sell_value = (quantity * entry_price, quantity * exit_price) if side == "long" else (quantity * exit_price, quantity * entry_price)
            costs = transaction_costs(buy_value, sell_value, fee)
            pnl = sell_value - buy_value - costs["total_cost"]
            daily_pnl += pnl
            trades.append({"strategy": strategy, "side": side, "slippage_bps_per_side": slippage_bps, "signal_date": signal_date, "target_date": row.target_date, "symbol": row.symbol, "signal_source": getattr(row, "signal_source", None), "signal_source_priority": getattr(row, "signal_source_priority", None), "selection_score": row.selection_score, "calibrated_probability": row.calibrated_probability, "calibrated_red_probability": getattr(row, "calibrated_red_probability", 1 - row.calibrated_probability), "predicted_return": row.predicted_return, "target_open": row.target_open, "target_close": row.target_close, "target_high": getattr(row, "target_high", np.nan), "target_low": getattr(row, "target_low", np.nan), "stop_price": stop_price, "stop_triggered": stop_triggered, "exit_reason": "hard_stop" if stop_triggered else "close", "quantity": quantity, "cash_budget": cash_budget, "buy_value": buy_value, "sell_value": sell_value, "gross_return": gross_return, "net_return_on_budget": pnl / cash_budget, "fee_status": fee.status, **costs})
        executed_trade_count = len(trades) - trade_count_before
        cooldown_triggered = False
        displayed_streak = 0
        if cooldown_active:
            # A cooldown deliberately breaks a run of consecutive trade days.
            consecutive_trade_days = 0
        elif executed_trade_count:
            consecutive_trade_days += 1
            displayed_streak = consecutive_trade_days
            if (
                config.max_consecutive_trade_days is not None
                and config.cooldown_days_after_streak > 0
                and consecutive_trade_days >= config.max_consecutive_trade_days
            ):
                cooldown_days_remaining = config.cooldown_days_after_streak
                cooldown_triggered = True
                consecutive_trade_days = 0
        else:
            consecutive_trade_days = 0
        capital += daily_pnl
        daily.append({"strategy": strategy, "slippage_bps_per_side": slippage_bps, "signal_date": signal_date, "capital_start": start_capital, "pnl": daily_pnl, "capital_end": capital, "portfolio_return": daily_pnl / start_capital if start_capital else 0.0, "trade_count": executed_trade_count, "zero_trade_day": executed_trade_count == 0, "market_volatility_20": market_volatility, "regime_blocked": regime_blocked, "regime_deleveraged": regime_deleveraged, "allocation_per_stock": allocation_per_stock, "cooldown_active": cooldown_active, "consecutive_trade_days": displayed_streak, "cooldown_triggered": cooldown_triggered, "cooldown_days_remaining_after": cooldown_days_remaining})
    return pd.DataFrame(trades), pd.DataFrame(daily)


def performance_summary(trades: pd.DataFrame, daily: pd.DataFrame, initial_capital: float) -> dict[str, float | int]:
    final_capital = float(daily["capital_end"].iloc[-1]) if not daily.empty else initial_capital
    total_return = final_capital / initial_capital - 1
    years = max((pd.Timestamp(daily["signal_date"].iloc[-1]) - pd.Timestamp(daily["signal_date"].iloc[0])).days / 365.25, 1 / 365.25) if not daily.empty else 1 / 365.25
    cagr = (final_capital / initial_capital) ** (1 / years) - 1
    returns = daily["portfolio_return"] if not daily.empty else pd.Series(dtype=float)
    annual_volatility = float(returns.std(ddof=0) * np.sqrt(252)) if len(returns) else 0.0
    downside = returns[returns < 0]
    sharpe = float(returns.mean() / returns.std(ddof=0) * np.sqrt(252)) if len(returns) and returns.std(ddof=0) else 0.0
    sortino = float(returns.mean() / downside.std(ddof=0) * np.sqrt(252)) if len(downside) and downside.std(ddof=0) else 0.0
    equity = daily["capital_end"] if not daily.empty else pd.Series([initial_capital])
    drawdown = equity / equity.cummax() - 1
    max_drawdown = float(drawdown.min())
    gross_profit = float(trades.loc[trades["net_return_on_budget"] > 0, "net_return_on_budget"].sum()) if not trades.empty else 0.0
    gross_loss = float(-trades.loc[trades["net_return_on_budget"] < 0, "net_return_on_budget"].sum()) if not trades.empty else 0.0
    return {"total_return": total_return, "cagr": cagr, "annualized_volatility": annual_volatility, "sharpe": sharpe, "sortino": sortino, "max_drawdown": max_drawdown, "calmar": cagr / abs(max_drawdown) if max_drawdown else 0.0, "win_rate": float((trades["net_return_on_budget"] > 0).mean()) if not trades.empty else 0.0, "profit_factor": gross_profit / gross_loss if gross_loss else 0.0, "average_gross_return_per_trade": float(trades["gross_return"].mean()) if not trades.empty else 0.0, "average_net_return_per_trade": float(trades["net_return_on_budget"].mean()) if not trades.empty else 0.0, "number_of_trading_days": len(daily), "number_of_trades": len(trades), "turnover": float((trades["buy_value"] + trades["sell_value"]).sum()) if not trades.empty else 0.0, "zero_trade_day_pct": float(daily["zero_trade_day"].mean()) if not daily.empty else 1.0}


def load_backtest_candidates(predictions_glob: str | list[str], normalized_path: Path) -> pd.DataFrame:
    """Join OOS predictions to signal-day liquidity and executable next-session OHLCV."""

    patterns = predictions_glob if isinstance(predictions_glob, list) else [predictions_glob]
    files = []
    for pattern_text in patterns:
        pattern = Path(pattern_text)
        files.extend(sorted(pattern.parent.glob(pattern.name)) if pattern.is_absolute() else sorted(Path().glob(pattern_text)))
    if not files:
        raise FileNotFoundError(f"No prediction files match {predictions_glob}")
    predictions = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
    ohlcv = pd.read_parquet(normalized_path)
    signal_liquidity = ohlcv[["symbol", "date", "close", "volume"]].rename(columns={"date": "signal_date", "close": "signal_close", "volume": "signal_volume"})
    target_bars = ohlcv[["symbol", "date", "open", "high", "low", "close", "volume"]].rename(columns={"date": "target_date", "open": "target_open", "high": "target_high", "low": "target_low", "close": "target_close", "volume": "target_volume"})
    candidates = predictions.merge(signal_liquidity, on=["symbol", "signal_date"], how="inner", validate="one_to_one").merge(target_bars, on=["symbol", "target_date"], how="inner", validate="one_to_one")
    candidates["signal_day_turnover"] = candidates["signal_close"] * candidates["signal_volume"]
    candidates["target_day_turnover"] = candidates["target_close"] * candidates["target_volume"]
    return candidates


def apply_market_index_filter(candidates: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
    """Apply an optional completed-candle market gate without forward filling."""

    if config.market_index_filter is None:
        return candidates
    if config.market_index_filter != "niftybank_below_sma20":
        raise ValueError("market_index_filter must be 'niftybank_below_sma20' when supplied")
    path = config.market_index_paths.get("niftybank")
    if path is None:
        raise ValueError("market_index_filter='niftybank_below_sma20' requires market_index_paths.niftybank")
    # Imports stay local to avoid making the general simulator depend on index
    # research files when no optional market filter is requested.
    from .features import _market_index_features
    from .phase2 import _read_market_index

    features = _market_index_features(_read_market_index(path), prefix="niftybank")
    joined = candidates.merge(features[["date", "niftybank_close_sma_20_ratio"]], left_on="signal_date", right_on="date", how="left", validate="many_to_one")
    # A missing same-date index candle is deliberately not filled from a later
    # date, so it cannot become a look-ahead signal.
    return joined.loc[joined["niftybank_close_sma_20_ratio"].lt(0)].drop(columns=["date", "niftybank_close_sma_20_ratio"])


def run_backtest(config: BacktestConfig) -> dict[str, Path | int]:
    candidates = load_backtest_candidates(config.predictions_glob, config.normalized_path)
    candidates = apply_market_index_filter(candidates, config)
    periods = load_fee_schedule(config.fee_schedule_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    all_trades: list[pd.DataFrame] = []
    all_daily: list[pd.DataFrame] = []
    for strategy in config.strategies:
        for slippage in config.slippage_bps_per_side:
            trades, daily = simulate(candidates, periods, config, strategy=strategy, slippage_bps=slippage)
            summaries.append({"strategy": strategy, "slippage_bps_per_side": slippage, **performance_summary(trades, daily, config.initial_capital)})
            all_trades.append(trades)
            all_daily.append(daily)
    trade_path, daily_path, summary_path = config.output_dir / "trades.csv", config.output_dir / "daily_equity.csv", config.output_dir / "strategy_summary.csv"
    pd.concat(all_trades, ignore_index=True).to_csv(trade_path, index=False)
    pd.concat(all_daily, ignore_index=True).to_csv(daily_path, index=False)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    metadata_path = config.output_dir / "backtest_metadata.json"
    metadata_path.write_text(json.dumps({"phase": 5, "fee_schedule": [period.model_dump() for period in periods], "limitations": ["Costs are explicitly marked estimated where broker, venue-tier, state stamp-duty, or historical source evidence is incomplete.", "Daily OHLCV cannot establish opening-auction fills, spreads, market impact, or intraday order sequencing.", "Short simulations are intraday-only, reserve the configured notional allocation, use the same estimated cash-equity cost schedule, and exclude borrow, margin financing, auction, and short-sale availability effects." ]}, indent=2), encoding="utf-8")
    return {"trades": trade_path, "daily_equity": daily_path, "summary": summary_path, "metadata": metadata_path, "runs": len(summaries)}


def run_backtest_from_config(path: str | Path) -> dict[str, Path | int]:
    return run_backtest(load_backtest_config(path))
