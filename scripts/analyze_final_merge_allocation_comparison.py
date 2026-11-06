"""Compare 20% and 33% sizing for the final merged 4% SL / 8% TP setup."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/short_model_gap_down3_comparison_dashboard"
OUTPUT = SOURCE / "production_review"
RULE = "SL 4% / TP 8% (conservative)"
MODEL = "Priority merge: Baseline then raw-tail"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SOURCE / "comparison_summary.csv")
    years = pd.read_csv(SOURCE / "comparison_yearwise.csv")
    trades = pd.read_parquet(SOURCE / "comparison_trade_ledger.parquet")
    daily = pd.read_parquet(SOURCE / "comparison_daily_equity.parquet")
    base = summary.loc[
        summary["model_portfolio"].eq(MODEL) & summary["exit_rule"].eq(RULE)
    ].copy()
    columns = [
        "allocation_per_position", "Trades", "Positive trades", "Negative trades", "Win rate (%)",
        "Sum gross trade returns (%)", "Sum net trade returns (%)", "Total return (%)", "CAGR (%)",
        "Max drawdown (%)", "Sharpe", "Trade profit factor", "Avg. net/trade (%)",
        "Same-day TP/SL collisions", "TP/SL collision rate (%)",
    ]
    base[columns].to_csv(OUTPUT / "final_4_8_allocation_summary.csv", index=False)
    yearly = years.loc[
        years["model_portfolio"].eq(MODEL) & years["exit_rule"].eq(RULE)
    ].copy()
    yearly = yearly[[
        "allocation_per_position", "year", "trades", "positive", "negative", "win_rate_pct",
        "profit_factor", "avg_net_trade_pct", "return_pct", "max_drawdown_pct",
    ]]
    yearly.to_csv(OUTPUT / "final_4_8_allocation_yearwise.csv", index=False)

    rows: list[dict[str, float | int]] = []
    for allocation in (0.20, 0.33):
        key = base.loc[base["allocation_per_position"].eq(allocation), "key"].iloc[0]
        item_trades = trades.loc[trades["key"].eq(key)].copy()
        item_daily = daily.loc[daily["key"].eq(key), ["signal_date", "target_date", "capital_start", "portfolio_return"]].copy()
        item_trades = item_trades.merge(item_daily[["signal_date", "capital_start"]], on="signal_date", how="left")
        item_trades["desired_budget"] = item_trades["capital_start"] * allocation
        item_trades["effective_allocation"] = item_trades["cash_budget"] / item_trades["capital_start"]
        item_trades["liquidity_capped"] = item_trades["cash_budget"] < (item_trades["desired_budget"] - 0.01)
        item_trades["year"] = pd.to_datetime(item_trades["target_date"]).dt.year
        by_day = item_trades.groupby("signal_date", sort=False).agg(
            cash_budget=("cash_budget", "sum"), capital_start=("capital_start", "first"),
        )
        by_day["gross_exposure"] = by_day["cash_budget"] / by_day["capital_start"]
        rows.append({
            "allocation_per_position": allocation,
            "liquidity_capped_trade_pct": 100 * item_trades["liquidity_capped"].mean(),
            "mean_effective_allocation_per_trade_pct": 100 * item_trades["effective_allocation"].mean(),
            "median_effective_allocation_per_trade_pct": 100 * item_trades["effective_allocation"].median(),
            "mean_gross_exposure_on_trade_days_pct": 100 * by_day["gross_exposure"].mean(),
            "median_gross_exposure_on_trade_days_pct": 100 * by_day["gross_exposure"].median(),
            "daily_return_p05_pct": 100 * item_daily["portfolio_return"].quantile(0.05),
            "worst_daily_return_pct": 100 * item_daily["portfolio_return"].min(),
        })
    pd.DataFrame(rows).to_csv(OUTPUT / "final_4_8_allocation_exposure.csv", index=False)
    print(f"Wrote allocation comparison tables under {OUTPUT}")


if __name__ == "__main__":
    main()
