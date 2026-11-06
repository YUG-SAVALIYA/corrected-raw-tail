"""API for the completed-candle direct-short HGB dashboard (research only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


MIN_DATE, MAX_DATE = pd.Timestamp("2020-01-01"), pd.Timestamp("2026-10-16")
THRESHOLDS = {
    "99.00%": "short_probability_calibration_q9900", "99.50%": "short_probability_calibration_q9950",
    "99.70%": "short_probability_calibration_q9970", "99.90%": "short_probability_calibration_q9990",
    "99.95%": "short_probability_calibration_q9995", "99.97%": "short_probability_calibration_q9997",
    "99.99%": "short_probability_calibration_q9999",
}


class HGBDashboardRequest(BaseModel):
    start_date: str = "2020-01-01"; end_date: str = "2026-10-16"; initial_capital: float = 100000
    max_positions: int = 3; allocation_pct: float = 33; probability_quantile: str = "99.97%"
    min_volatility_20_pct: float = 3.6; min_signal_day_turnover: float = 5_000_000
    slippage_bps_per_side: float = 5; stop_loss_pct: float = 10
    max_consecutive_trade_days: int = 0; cooldown_days_after_streak: int = 0


def _rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def register_completed_hgb_dashboard(app, *, base_dir: Path) -> None:
    router = APIRouter(tags=["Completed-candle HGB"])
    artifacts = base_dir / "artifacts"
    result_dir = artifacts / "nifty_hgb_direct_short_max3_alloc33_capital100k_2020_2026" / "backtests"
    prediction_dir = artifacts / "nifty_hgb_direct_short_event_calibrated_strict_thresholds_2020_2026" / "predictions"
    normalized_path = artifacts / "nifty_phase1" / "normalized_ohlcv.parquet"
    fee_path = base_dir / "configs" / "costs.yaml"

    def reference() -> dict[str, Any]:
        needed = [result_dir / name for name in ("strategy_summary.csv", "trades.csv", "daily_equity.csv")]
        if not all(path.exists() for path in needed):
            raise HTTPException(500, "Completed-candle HGB reference artifacts are missing.")
        return {
            "kind": "audited completed-candle reference", "model": "Direct-short HGB: P(next open-to-close return <= -1.2%)",
            "summary": _rows(pd.read_csv(result_dir / "strategy_summary.csv"))[0],
            "trades": _rows(pd.read_csv(result_dir / "trades.csv")), "daily_equity": _rows(pd.read_csv(result_dir / "daily_equity.csv")),
            "limits": {"min_date": str(MIN_DATE.date()), "max_date": str(MAX_DATE.date())},
        }

    @router.get("/api/nifty-hgb/reference")
    def get_reference() -> dict[str, Any]:
        return reference()

    @router.post("/api/nifty-hgb/backtest")
    def run_backtest(request: HGBDashboardRequest) -> dict[str, Any]:
        try:
            start, end = pd.Timestamp(request.start_date).normalize(), pd.Timestamp(request.end_date).normalize()
        except ValueError as error:
            raise HTTPException(422, "Dates must be YYYY-MM-DD.") from error
        if pd.isna(start) or pd.isna(end) or start > end or start < MIN_DATE or end > MAX_DATE:
            raise HTTPException(422, f"Choose dates from {MIN_DATE.date()} through {MAX_DATE.date()}.")
        if not 1 <= request.max_positions <= 3 or request.allocation_pct <= 0 or request.allocation_pct * request.max_positions > 100:
            raise HTTPException(422, "Use 1–3 positions and allocation × positions no greater than 100%.")
        if request.probability_quantile not in THRESHOLDS:
            raise HTTPException(422, "Invalid probability threshold.")
        if request.max_consecutive_trade_days < 0 or request.cooldown_days_after_streak < 0:
            raise HTTPException(422, "Cooldown values cannot be negative.")
        if bool(request.max_consecutive_trade_days) != bool(request.cooldown_days_after_streak):
            raise HTTPException(422, "Set both cooldown values, or leave both at zero to disable the rule.")
        try:
            from nifty_research.backtest import BacktestConfig, load_backtest_candidates, load_fee_schedule, performance_summary, simulate
            config = BacktestConfig(
                predictions_glob=str(prediction_dir / "hist_gradient_boosting_*.parquet"), normalized_path=normalized_path,
                fee_schedule_path=fee_path, strategies=["hybrid_short"], slippage_bps_per_side=[request.slippage_bps_per_side],
                initial_capital=request.initial_capital, max_positions=request.max_positions, max_allocation_per_stock=request.allocation_pct / 100,
                max_turnover_fraction_per_stock=.02, min_signal_day_turnover=request.min_signal_day_turnover, min_short_red_probability=0,
                short_probability_threshold_column=THRESHOLDS[request.probability_quantile], require_short_return_prediction=False,
                min_short_volatility_20=request.min_volatility_20_pct / 100, short_volatility_rank_penalty=0,
                hard_stop_loss_pct=request.stop_loss_pct / 100,
                max_consecutive_trade_days=request.max_consecutive_trade_days or None,
                cooldown_days_after_streak=request.cooldown_days_after_streak,
            )
            candidates = load_backtest_candidates(config.predictions_glob, normalized_path)
            candidates["signal_date"] = pd.to_datetime(candidates["signal_date"]).dt.normalize()
            trades, daily = simulate(candidates.loc[candidates["signal_date"].between(start, end)].copy(), load_fee_schedule(fee_path), config, strategy="hybrid_short", slippage_bps=request.slippage_bps_per_side)
            summary = performance_summary(trades, daily, config.initial_capital)
        except Exception as error:
            raise HTTPException(500, f"HGB backtest failed: {error}") from error
        return {
            "kind": "custom completed-candle frozen-prediction backtest", "model": "Frozen yearly direct-short HGB models; no model was trained or refitted.",
            "settings": request.model_dump(), "summary": {"strategy":"hybrid_short", "slippage_bps_per_side":request.slippage_bps_per_side, **summary},
            "trades": _rows(trades), "daily_equity": _rows(daily), "limits": {"min_date":str(MIN_DATE.date()), "max_date":str(MAX_DATE.date())},
        }

    app.include_router(router)
