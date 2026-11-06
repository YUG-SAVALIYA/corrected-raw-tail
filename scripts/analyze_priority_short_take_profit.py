"""Assess fixed intraday profit targets for the priority short merge.

Daily OHLC does not reveal whether a target or stop was touched first when
both occur in the same candle.  Each target is therefore evaluated under
conservative (stop-first) and optimistic (target-first) assumptions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nifty_research.backtest import (
    fee_for_date,
    load_backtest_candidates,
    load_fee_schedule,
    performance_summary,
    transaction_costs,
    _rates,
)


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "deliverables/nifty_direct_short_hgb_frozen_2026_10_28/frozen_predictions/hist_gradient_boosting_*.parquet"
RAW_TAIL = ROOT / "artifacts/nifty_hgb_direct_short_market_index_raw_tail_walkforward_2020_2026/predictions/hist_gradient_boosting_*.parquet"
NORMALIZED = ROOT / "artifacts/nifty_phase1/normalized_ohlcv.parquet"
FEES = ROOT / "configs/costs.yaml"
OUTPUT = ROOT / "artifacts/nifty_priority_short_take_profit_sensitivity_2020_2026"
# Fixed-target experiments. ``None`` is the existing exit-at-close control for
# the same stop size. Most are 1:2 reward/risk; 4% / 10% is included as the
# requested 2.5:1 variation.
TARGET_CONFIGURATIONS = [
    (0.02, 0.04),
    (0.03, 0.06),
    (0.04, 0.08),
    (0.04, 0.10),
    (0.05, 0.10),
]


def candidates() -> pd.DataFrame:
    baseline = load_backtest_candidates(str(BASELINE), NORMALIZED)
    baseline["preselected"] = baseline["calibrated_short_probability"].ge(baseline["short_probability_calibration_q9997"]) & baseline["volatility_20"].ge(0.036)
    baseline["source_score"] = baseline["calibrated_short_probability"]
    baseline["signal_source"] = "baseline_q9997"
    baseline["signal_source_priority"] = 0
    raw = load_backtest_candidates(str(RAW_TAIL), NORMALIZED)
    raw["preselected"] = raw["raw_short_probability"].ge(raw["short_raw_probability_calibration_q9997"]) & raw["volatility_20"].ge(0.036)
    raw["source_score"] = raw["raw_short_probability"]
    raw["signal_source"] = "raw_tail_q9997"
    raw["signal_source_priority"] = 1
    return pd.concat([baseline, raw], ignore_index=True, sort=False)


def select_candidates(frame: pd.DataFrame, max_negative_gap: float | None = None) -> pd.DataFrame:
    eligible = frame.loc[frame["preselected"] & frame["signal_day_turnover"].ge(5_000_000)].copy()
    # The daily signal is known at the completed signal-day close.  The next
    # session's observed opening price is used to reject an excessive gap-down.
    # It is deliberately applied *before* source-priority ranking and the daily
    # three-position cap; excluded names therefore do not consume a slot.
    # A true live implementation must observe the opening quote and then submit
    # the order, so an official-open fill remains an optimistic bar-level
    # approximation until validated with intraday execution data.
    eligible["opening_gap_return"] = eligible["target_open"].div(eligible["signal_close"]).sub(1.0)
    if max_negative_gap is not None:
        eligible = eligible.loc[eligible["opening_gap_return"].gt(max_negative_gap)].copy()
    return (
        eligible.sort_values(["signal_date", "signal_source_priority", "source_score", "symbol"], ascending=[True, True, False, True], kind="stable")
        .drop_duplicates(["signal_date", "symbol"], keep="first")
        .groupby("signal_date", sort=False)
        .head(3)
    )


def simulate(
    signal_dates: pd.DatetimeIndex,
    selected: pd.DataFrame,
    periods,
    *,
    hard_stop_loss_pct: float,
    take_profit_pct: float | None,
    both_hit_policy: str,
    allocation_per_position: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    if both_hit_policy not in {"stop_first", "target_first"}:
        raise ValueError(both_hit_policy)
    selected_by_day = {day: frame for day, frame in selected.groupby("signal_date", sort=False)}
    capital = 100_000.0
    trades: list[dict[str, object]] = []
    daily: list[dict[str, object]] = []
    ambiguity = {"target_hits": 0, "stop_hits": 0, "both_hits": 0}
    slippage = 0.0005
    for signal_date in signal_dates:
        start_capital = capital
        pnl = 0.0
        selected_day = selected_by_day.get(signal_date)
        if selected_day is None:
            selected_day = selected.iloc[0:0]
        for row in selected_day.itertuples(index=False):
            fee = fee_for_date(periods, row.target_date)
            buy_rate, _ = _rates(fee)
            cash_budget = min(start_capital * allocation_per_position, row.signal_day_turnover * 0.02)
            entry = row.target_open * (1 - slippage)
            stop_price = entry * (1 + hard_stop_loss_pct)
            target_price = entry * (1 - take_profit_pct) if take_profit_pct is not None else np.nan
            stop_hit = row.target_high >= stop_price
            target_hit = take_profit_pct is not None and row.target_low <= target_price
            ambiguity["stop_hits"] += int(stop_hit)
            ambiguity["target_hits"] += int(target_hit)
            ambiguity["both_hits"] += int(stop_hit and target_hit)
            if target_hit and stop_hit:
                target_wins = both_hit_policy == "target_first"
            else:
                target_wins = bool(target_hit)
            if target_wins:
                raw_exit, reason = target_price, "profit_target"
            elif stop_hit:
                raw_exit, reason = stop_price, "hard_stop"
            else:
                raw_exit, reason = row.target_close, "close"
            exit_price = raw_exit * (1 + slippage)
            quantity = cash_budget / (entry * (1 + buy_rate))
            buy_value, sell_value = quantity * exit_price, quantity * entry
            costs = transaction_costs(buy_value, sell_value, fee)
            net_pnl = sell_value - buy_value - costs["total_cost"]
            pnl += net_pnl
            trades.append({
                "signal_date": signal_date,
                "target_date": row.target_date,
                "symbol": row.symbol,
                "signal_source": row.signal_source,
                "exit_reason": reason,
                "gross_return": row.target_open / raw_exit - 1,
                "net_return_on_budget": net_pnl / cash_budget,
                "cash_budget": cash_budget,
                "buy_value": buy_value,
                "sell_value": sell_value,
                "net_pnl": net_pnl,
            })
        capital += pnl
        trade_count = len(selected_day)
        target_date = selected_day["target_date"].iloc[0] if trade_count else signal_date
        daily.append({"signal_date": signal_date, "target_date": target_date, "capital_start": start_capital, "pnl": pnl, "capital_end": capital, "portfolio_return": pnl / start_capital if start_capital else 0.0, "trade_count": trade_count, "zero_trade_day": trade_count == 0})
    return pd.DataFrame(trades), pd.DataFrame(daily), ambiguity


def pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def strategy_name(stop: float, target: float | None, policy: str) -> str:
    """Stable, readable identifier used in the tabular exports."""
    stop_label = f"SL {stop * 100:g}%"
    if target is None:
        return f"{stop_label} / close exit"
    policy_label = "conservative (stop first)" if policy == "stop_first" else "optimistic (target first)"
    return f"{stop_label} / TP {target * 100:g}% / {policy_label}"


def tabular_statistics(trades: pd.DataFrame, daily: pd.DataFrame) -> dict[str, float | int]:
    """Return trade and portfolio measures suitable for all-time or yearly tables."""
    net = trades["net_pnl"]
    positive = net.gt(0)
    negative = net.lt(0)
    gross_profit = float(net.loc[positive].sum())
    gross_loss = float(-net.loc[negative].sum())
    daily_return = daily["portfolio_return"].astype(float)
    equity = (1.0 + daily_return).cumprod()
    drawdown = equity.div(equity.cummax()).sub(1.0)
    daily_std = float(daily_return.std(ddof=1))
    downside_std = float(daily_return.loc[daily_return.lt(0)].std(ddof=1))
    annualized_volatility = daily_std * np.sqrt(252) if np.isfinite(daily_std) else np.nan
    sharpe = float(daily_return.mean() / daily_std * np.sqrt(252)) if daily_std > 0 else np.nan
    sortino = float(daily_return.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else np.nan
    return {
        "total_trades": int(len(trades)),
        "positive_trades": int(positive.sum()),
        "negative_trades": int(negative.sum()),
        "flat_trades": int((~positive & ~negative).sum()),
        "win_rate": float(positive.mean()) if len(trades) else np.nan,
        "gross_profit_rs": gross_profit,
        "gross_loss_rs": gross_loss,
        "net_profit_rs": float(net.sum()),
        "profit_factor": gross_profit / gross_loss if gross_loss else np.nan,
        "sum_gross_return": float(trades["gross_return"].sum()),
        "sum_net_return": float(trades["net_return_on_budget"].sum()),
        "average_gross_return_per_trade": float(trades["gross_return"].mean()),
        "average_net_return_per_trade": float(trades["net_return_on_budget"].mean()),
        "profit_target_exits": int(trades["exit_reason"].eq("profit_target").sum()),
        "hard_stop_exits": int(trades["exit_reason"].eq("hard_stop").sum()),
        "close_exits": int(trades["exit_reason"].eq("close").sum()),
        "portfolio_return": float(equity.iloc[-1] - 1.0) if len(equity) else np.nan,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else np.nan,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
    }


def write_markdown_report(
    report: pd.DataFrame,
    output: Path,
    *,
    all_candidates: pd.DataFrame,
    eligible_before_gap: pd.DataFrame,
    selected: pd.DataFrame,
    unfiltered_selected: pd.DataFrame,
    max_negative_gap: float | None,
    allocation_per_position: float,
) -> None:
    """Write a human-readable companion to the complete CSV/JSON results."""
    window_start = all_candidates["signal_date"].min().date()
    window_end = all_candidates["signal_date"].max().date()
    lines = [
        "# Priority short merge: gap-filtered fixed-target analysis",
        "",
        "## Scope and portfolio construction",
        "",
        f"- Signal window: {window_start} to {window_end}; {report.iloc[0]['number_of_trading_days']:.0f} trading days.",
        "- Strategy: priority short merge - frozen Baseline q99.97 is ranked first, then Corrected raw-tail q99.97 fills any remaining daily capacity.",
        f"- Capacity and sizing: maximum three positions per day, {pct(allocation_per_position)} of start-of-day capital per position (maximum {pct(3 * allocation_per_position)} gross exposure).",
        "- Candidate eligibility: model preselection plus signal-day turnover of at least Rs. 50 lakh.",
        "- Gap definition: next-session open / completed signal-day close - 1.",
        (
            f"- Gap rule: exclude candidates with a gap at or below {pct(max_negative_gap)} "
            "before source priority, de-duplication, and the three-slot cap."
            if max_negative_gap is not None
            else "- No opening-gap rule was applied."
        ),
        "- Costs: project fee schedule plus 5 bps price slippage per side. Position sizes are additionally capped at 2% of the signal-day turnover.",
        "",
        "## Filter and selection impact",
        "",
        "| Measure | Count |",
        "|---|---:|",
        f"| Eligible model candidates before gap filter | {len(eligible_before_gap):,} |",
        f"| Rejected for gap <= {pct(max_negative_gap)} | {len(eligible_before_gap) - len(eligible_before_gap.loc[eligible_before_gap['opening_gap_return'].gt(max_negative_gap)]):,} |" if max_negative_gap is not None else "| Rejected for gap rule | 0 |",
        f"| Selected transactions without gap filter | {len(unfiltered_selected):,} |",
        f"| Selected transactions with gap filter | {len(selected):,} |",
        f"| Selected signal days with gap filter | {selected['signal_date'].nunique():,} |",
        "",
        "The gap filter is applied while all eligible signals are still available. A rejected Baseline signal therefore does not consume a slot; the next eligible Baseline or raw-tail signal may replace it.",
        "",
        "## Same-stop close-exit controls",
        "",
        "| Stop | Total return | CAGR | Max drawdown | Sharpe | Win rate | Profit factor | Avg. net/trade | Trades |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    controls = report.loc[report["experiment"].eq("close_control")].sort_values("hard_stop_loss_pct")
    for row in controls.itertuples(index=False):
        lines.append(
            f"| {pct(row.hard_stop_loss_pct)} | {pct(row.total_return)} | {pct(row.cagr)} | {pct(row.max_drawdown)} | "
            f"{row.sharpe:.2f} | {pct(row.win_rate)} | {row.profit_factor:.2f} | {pct(row.average_net_return_per_trade)} | {row.number_of_trades:,} |"
        )
    lines += [
        "",
        "## Fixed target results",
        "",
        "For an intraday daily candle that touches both target and stop, daily OHLC cannot determine which came first. Both bounds are shown below. Conservative = stop first; optimistic = target first.",
        "",
        "| Stop / target | Assumption | Total return | CAGR | Max DD | Sharpe | Win rate | Profit factor | Avg. net/trade | Target hits | Stop hits | Both hit |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    targets = report.loc[report["experiment"].eq("fixed_target")].sort_values(["hard_stop_loss_pct", "take_profit_pct", "both_hit_policy"])
    for row in targets.itertuples(index=False):
        assumption = "Conservative: stop first" if row.both_hit_policy == "stop_first" else "Optimistic: target first"
        lines.append(
            f"| {pct(row.hard_stop_loss_pct)} / {pct(row.take_profit_pct)} | {assumption} | {pct(row.total_return)} | "
            f"{pct(row.cagr)} | {pct(row.max_drawdown)} | {row.sharpe:.2f} | {pct(row.win_rate)} | "
            f"{row.profit_factor:.2f} | {pct(row.average_net_return_per_trade)} | {row.target_hits:,} | {row.stop_hits:,} | {row.both_hits:,} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Use the conservative row for a daily-bar decision because same-day target/stop collisions cannot be ordered from this data.",
        "- Compare reward/risk ratios using the `reward_risk_ratio` column in the accompanying tables. This is historical backtest evidence, not a guarantee of future performance.",
        "- `strategy_summary_table.csv` provides the full all-period trade table; `strategy_yearwise_table.csv` provides the same fields for every calendar year and strategy. `strategy_trade_ledger.parquet` contains the underlying simulated trades.",
        "",
        "## Execution limitations",
        "",
        "- The opening gap becomes known at the market open. The simulation continues to value a passing trade at the official open (with 5 bps slippage), which is optimistic if the live order can only be submitted after observing that open.",
        "- Daily OHLC cannot prove intraday stop/target order, partial fills, liquidity, short availability, borrowing costs, market impact, or exchange execution rules.",
        "- Validate the chosen rule with 5- or 15-minute data and a realistic post-open entry model before deployment.",
        "",
    ]
    (output / "gap_filter_1to2_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-negative-gap",
        type=float,
        default=None,
        help="Exclude a signal before slot selection when next-open / signal-close - 1 is at or below this value (e.g. -0.03).",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument(
        "--allocation-per-position",
        type=float,
        default=0.20,
        help="Fraction of start-of-day capital allocated to each selected position (default: 0.20).",
    )
    args = parser.parse_args()
    if not 0 < args.allocation_per_position <= 1 / 3:
        parser.error("--allocation-per-position must be greater than 0 and no more than 1/3 when three slots are used.")
    args.output.mkdir(parents=True, exist_ok=True)
    all_candidates = candidates()
    eligible_before_gap = all_candidates.loc[
        all_candidates["preselected"] & all_candidates["signal_day_turnover"].ge(5_000_000)
    ].copy()
    eligible_before_gap["opening_gap_return"] = eligible_before_gap["target_open"].div(eligible_before_gap["signal_close"]).sub(1.0)
    unfiltered_selected = select_candidates(all_candidates)
    selected = select_candidates(all_candidates, args.max_negative_gap)
    signal_dates = pd.DatetimeIndex(all_candidates["signal_date"].unique()).sort_values()
    periods = load_fee_schedule(FEES)
    rows: list[dict[str, object]] = []
    table_rows: list[dict[str, object]] = []
    year_rows: list[dict[str, object]] = []
    trade_ledger: list[pd.DataFrame] = []
    experiments = (
        [(stop, None, "close_control") for stop in sorted({stop for stop, _ in TARGET_CONFIGURATIONS})]
        + [(stop, target, "fixed_target") for stop, target in TARGET_CONFIGURATIONS]
    )
    for stop, selected_target, label in experiments:
        policies = ["close_only"] if selected_target is None else ["stop_first", "target_first"]
        for policy in policies:
                trades, daily, ambiguity = simulate(
                    signal_dates,
                    selected,
                    periods,
                    hard_stop_loss_pct=stop,
                    take_profit_pct=selected_target,
                    both_hit_policy="stop_first" if policy == "close_only" else policy,
                    allocation_per_position=args.allocation_per_position,
                )
                summary = performance_summary(trades, daily, 100_000)
                name = strategy_name(stop, selected_target, policy)
                rows.append({
                    "strategy": name,
                    "experiment": label,
                    "reward_risk_ratio": None if selected_target is None else selected_target / stop,
                    "max_negative_gap": args.max_negative_gap,
                    "allocation_per_position": args.allocation_per_position,
                    "max_positions": 3,
                    "hard_stop_loss_pct": stop,
                    "take_profit_pct": selected_target,
                    "both_hit_policy": policy,
                    "allocation_per_position": args.allocation_per_position,
                    **summary,
                    **ambiguity,
                    "profit_target_exits": int((trades["exit_reason"] == "profit_target").sum()),
                    "hard_stop_exits": int((trades["exit_reason"] == "hard_stop").sum()),
                })
                table_rows.append({
                    "strategy": name,
                    "period": "All",
                    "hard_stop_loss_pct": stop,
                    "take_profit_pct": selected_target,
                    "both_hit_policy": policy,
                    **tabular_statistics(trades, daily),
                })
                trades_with_year = trades.assign(year=pd.to_datetime(trades["target_date"]).dt.year)
                daily_with_year = daily.assign(year=pd.to_datetime(daily["target_date"]).dt.year)
                for year in sorted(trades_with_year["year"].unique()):
                    year_trades = trades_with_year.loc[trades_with_year["year"].eq(year)]
                    year_daily = daily_with_year.loc[daily_with_year["year"].eq(year)]
                    year_rows.append({
                        "strategy": name,
                        "year": int(year),
                        "hard_stop_loss_pct": stop,
                        "take_profit_pct": selected_target,
                        "both_hit_policy": policy,
                        "allocation_per_position": args.allocation_per_position,
                        **tabular_statistics(year_trades, year_daily),
                    })
                trade_ledger.append(trades.assign(strategy=name, hard_stop_loss_pct=stop, take_profit_pct=selected_target, both_hit_policy=policy, allocation_per_position=args.allocation_per_position))
    report = pd.DataFrame(rows)
    strategy_table = pd.DataFrame(table_rows)
    yearwise_table = pd.DataFrame(year_rows)
    report.to_csv(args.output / "take_profit_sensitivity.csv", index=False)
    report.to_json(args.output / "take_profit_sensitivity.json", orient="records", indent=2)
    strategy_table.to_csv(args.output / "strategy_summary_table.csv", index=False)
    yearwise_table.to_csv(args.output / "strategy_yearwise_table.csv", index=False)
    pd.concat(trade_ledger, ignore_index=True).to_parquet(args.output / "strategy_trade_ledger.parquet", index=False)
    selected.to_parquet(args.output / "selected_signals.parquet", index=False)
    write_markdown_report(
        report,
        args.output,
        all_candidates=all_candidates,
        eligible_before_gap=eligible_before_gap,
        selected=selected,
        unfiltered_selected=unfiltered_selected,
        max_negative_gap=args.max_negative_gap,
        allocation_per_position=args.allocation_per_position,
    )
    print(
        f"Selected {len(selected)} signals after gap filter {args.max_negative_gap}; "
        f"{selected['signal_date'].nunique()} signal days."
    )
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
