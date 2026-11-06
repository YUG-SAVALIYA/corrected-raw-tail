"""Create a production-readiness review for all dashboard configurations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts/short_model_gap_down3_comparison_dashboard"
OUTPUT = INPUT / "production_review"


def longest_true_streak(values: pd.Series) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(INPUT / "comparison_summary.csv")
    yearly = pd.read_csv(INPUT / "comparison_yearwise.csv")
    trades = pd.read_parquet(INPUT / "comparison_trade_ledger.parquet")
    daily = pd.read_parquet(INPUT / "comparison_daily_equity.parquet")
    trades["target_date"] = pd.to_datetime(trades["target_date"])
    daily["target_date"] = pd.to_datetime(daily["target_date"])
    rows: list[dict[str, object]] = []
    for key, item in summary.set_index("key").iterrows():
        item_trades = trades.loc[trades["key"].eq(key)].sort_values(["target_date", "symbol"], kind="stable")
        item_daily = daily.loc[daily["key"].eq(key)].sort_values("target_date", kind="stable")
        monthly = item_daily.set_index("target_date")["portfolio_return"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
        annual = yearly.loc[yearly["key"].eq(key)]
        calmar = item["CAGR (%)"] / abs(item["Max drawdown (%)"])
        rows.append({
            "key": key,
            **item.to_dict(),
            "calmar": round(float(calmar), 3),
            "positive_years": int(annual["return_pct"].gt(0).sum()),
            "worst_year_return_pct": round(float(annual["return_pct"].min()), 2),
            "worst_year_drawdown_pct": round(float(annual["max_drawdown_pct"].min()), 2),
            "worst_month_return_pct": round(float(monthly.min() * 100), 2),
            "positive_month_pct": round(float(monthly.gt(0).mean() * 100), 2),
            "daily_return_p05_pct": round(float(item_daily["portfolio_return"].quantile(0.05) * 100), 3),
            "max_consecutive_loss_days": longest_true_streak(item_daily["portfolio_return"].lt(0)),
            "max_consecutive_losing_trades": longest_true_streak(item_trades["net_pnl"].lt(0)),
        })
    review = pd.DataFrame(rows)
    columns = [
        "key", "model_portfolio", "allocation_per_position", "exit_rule", "Trades",
        "Total return (%)", "CAGR (%)", "Max drawdown (%)", "Sharpe", "calmar",
        "Trade profit factor", "Win rate (%)", "Avg. net/trade (%)", "positive_years",
        "worst_year_return_pct", "worst_month_return_pct", "positive_month_pct",
        "daily_return_p05_pct", "max_consecutive_loss_days", "max_consecutive_losing_trades",
        "Same-day TP/SL collisions", "TP/SL collision rate (%)",
    ]
    review = review[columns].sort_values(["allocation_per_position", "Sharpe"], ascending=[True, False])
    review.to_csv(OUTPUT / "all_configuration_production_review.csv", index=False)

    shortlist = review.loc[
        (review["allocation_per_position"].eq(0.20))
        & review["exit_rule"].eq("SL 4% / TP 8% (conservative)")
    ].copy()
    order = {
        "Priority merge: Baseline then raw-tail": 1,
        "Baseline q99.97": 2,
        "Corrected raw-tail q99.97": 3,
    }
    shortlist["shortlist_rank"] = shortlist["model_portfolio"].map(order)
    shortlist["recommended_role"] = shortlist["shortlist_rank"].map({
        1: "Primary candidate (only one to consider for live pilot)",
        2: "Shadow benchmark / fallback; do not run alongside the merge",
        3: "Shadow capacity monitor; already a component of the merge",
    })
    shortlist = shortlist.sort_values("shortlist_rank")
    shortlist.to_csv(OUTPUT / "top_three_shortlist.csv", index=False)

    report = [
        "# Production-readiness review: short-model configurations",
        "",
        "## Decision",
        "",
        "The historical evidence supports one candidate for a controlled live pilot: the 20% allocation priority merge with 4% stop / 8% target. The other two shortlist entries are not independent portfolios to run at the same time: the priority merge already contains Baseline first and raw-tail as capacity filler.",
        "",
        "## Top-three shortlist",
        "",
        "| Rank | Portfolio | Allocation | Exit | Return | CAGR | Max DD | Sharpe | Trade PF | Trades | Role |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in shortlist.iterrows():
        report.append(
            f"| {row['shortlist_rank']} | {row['model_portfolio']} | {row['allocation_per_position'] * 100:.0f}% | {row['exit_rule']} | "
            f"{row['Total return (%)']:.2f}% | {row['CAGR (%)']:.2f}% | {row['Max drawdown (%)']:.2f}% | "
            f"{row['Sharpe']:.2f} | {row['Trade profit factor']:.2f} | {row['Trades']:,} | {row['recommended_role']} |"
        )
    report += [
        "",
        "## Why the 20% 4%/8% shortlist",
        "",
        "- It keeps maximum historical drawdown near 5%, unlike the 33% allocation configurations (8% to 16% drawdown) and the 10% stop configurations (roughly 6% to 16% drawdown).",
        "- Baseline and the merge have positive returns in all seven tested calendar years. Raw-tail has no 2020 trades and is positive in each of its six active years. The merge provides the highest return and Sharpe among the low-drawdown 20% target configurations.",
        "- The 4% / 8% target was evaluated conservatively: when daily OHLC touches both stop and target, the stop is assumed first. Intraday data is still required to validate this ordering.",
        "",
        "## Reasons this is not production-approved yet",
        "",
        "- The gap filter requires observing the opening price; the current backtest still assumes a fill at that official open after the rule passes. A post-open intraday execution model is required.",
        "- Daily OHLC cannot validate target/stop sequencing, partial fills, borrow availability/cost, market impact, or queue position. The strategy is short-only and needs an executable SLB/shorting workflow.",
        "- Historical data/model checks were strong for the corrected raw-tail pipeline, but the research universe may have survivorship bias and does not prove live performance.",
        "- Before a pilot: use 5- or 15-minute data, add realistic post-open entries and borrow costs, run a paper-trading period, and enforce daily loss, exposure, kill-switch, audit-log, and broker-compliance controls.",
        "",
        "## Configuration review",
        "",
        "See `all_configuration_production_review.csv` for every configuration's return, drawdown, Sharpe, Calmar, year/month consistency, 5th-percentile daily result, losing streaks, and target/stop collision counts.",
        "",
    ]
    (OUTPUT / "production_readiness_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
