"""Merge baseline q99.97 and corrected raw-tail short models by slot priority."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import pandas as pd

from nifty_research.backtest import BacktestConfig, load_backtest_candidates, load_fee_schedule, performance_summary, simulate


ROOT = Path(__file__).resolve().parents[1]
NORMALIZED = ROOT / "artifacts/nifty_phase1/normalized_ohlcv.parquet"
FEES = ROOT / "configs/costs.yaml"
DEFAULT_OUTPUT = ROOT / "artifacts/nifty_hgb_priority_baseline_raw_tail_merge_2020_2026/backtests"
BASELINE = ROOT / "deliverables/nifty_direct_short_hgb_frozen_2026_10_28/frozen_predictions/hist_gradient_boosting_*.parquet"
RAW_TAIL = ROOT / "artifacts/nifty_hgb_direct_short_market_index_raw_tail_walkforward_2020_2026/predictions/hist_gradient_boosting_*.parquet"


def config(hard_stop_loss_pct: float, max_allocation_per_stock: float) -> BacktestConfig:
    return BacktestConfig(
        predictions_glob="unused",
        normalized_path=NORMALIZED,
        fee_schedule_path=FEES,
        initial_capital=100_000,
        max_positions=3,
        max_allocation_per_stock=max_allocation_per_stock,
        max_turnover_fraction_per_stock=0.02,
        min_signal_day_turnover=5_000_000,
        strategies=["priority_hybrid_short"],
        slippage_bps_per_side=[5.0],
        min_short_red_probability=0.0,
        require_short_return_prediction=False,
        min_short_volatility_20=0.036,
        short_volatility_rank_penalty=0.0,
        hard_stop_loss_pct=hard_stop_loss_pct,
    )


def baseline_candidates() -> pd.DataFrame:
    frame = load_backtest_candidates(str(BASELINE), NORMALIZED)
    score = frame["calibrated_short_probability"]
    eligible = score.ge(frame["short_probability_calibration_q9997"]) & frame["volatility_20"].ge(0.036)
    frame["short_preselected"] = eligible
    frame["source_score"] = score
    frame["signal_source"] = "baseline_q9997"
    frame["signal_source_priority"] = 0
    return frame


def raw_tail_candidates() -> pd.DataFrame:
    frame = load_backtest_candidates(str(RAW_TAIL), NORMALIZED)
    score = frame["raw_short_probability"]
    eligible = score.ge(frame["short_raw_probability_calibration_q9997"]) & frame["volatility_20"].ge(0.036)
    frame["short_preselected"] = eligible
    frame["source_score"] = score
    frame["signal_source"] = "raw_tail_q9997"
    frame["signal_source_priority"] = 1
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-stop-loss-pct", type=float, default=0.10)
    parser.add_argument("--max-allocation-per-stock", type=float, default=0.33)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 0 < args.hard_stop_loss_pct < 1:
        raise ValueError("--hard-stop-loss-pct must be between 0 and 1.")
    if not 0 < args.max_allocation_per_stock <= 1:
        raise ValueError("--max-allocation-per-stock must be between 0 and 1.")

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    merged = pd.concat([baseline_candidates(), raw_tail_candidates()], ignore_index=True, sort=False)
    cfg = config(args.hard_stop_loss_pct, args.max_allocation_per_stock)
    trades, daily = simulate(merged, load_fee_schedule(FEES), cfg, strategy="priority_hybrid_short", slippage_bps=5.0)
    summary = pd.DataFrame([{"strategy": "priority_hybrid_short", "slippage_bps_per_side": 5.0, **performance_summary(trades, daily, cfg.initial_capital)}])
    trades.to_csv(output / "trades.csv", index=False)
    daily.to_csv(output / "daily_equity.csv", index=False)
    summary.to_csv(output / "strategy_summary.csv", index=False)
    (output / "merge_metadata.json").write_text(json.dumps({
        "priority": "baseline q99.97 first; corrected raw-tail fills only unused daily slots",
        "max_positions": cfg.max_positions,
        "max_allocation_per_stock": cfg.max_allocation_per_stock,
        "hard_stop_loss_pct": cfg.hard_stop_loss_pct,
        "baseline_predictions": str(BASELINE),
        "raw_tail_predictions": str(RAW_TAIL),
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(trades["signal_source"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
