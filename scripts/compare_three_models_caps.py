"""Compare Baseline q99.97, Corrected Raw-tail, and Priority Merge with and without 2% turnover cap."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from build_short_model_comparison_dashboard import select_for_model, MAX_NEGATIVE_GAP, MAX_POSITIONS
from analyze_priority_short_take_profit import candidates, load_fee_schedule, performance_summary, tabular_statistics
from nifty_research.backtest import fee_for_date, transaction_costs, _rates

ROOT = Path(__file__).resolve().parents[1]
FEES = ROOT / "configs/costs.yaml"
OUTPUT = ROOT / "results/cap_comparison"
OUTPUT.mkdir(parents=True, exist_ok=True)

def simulate_custom(
    signal_dates: pd.DatetimeIndex,
    selected: pd.DataFrame,
    periods,
    *,
    hard_stop_loss_pct: float,
    take_profit_pct: float | None,
    both_hit_policy: str,
    allocation_per_position: float,
    apply_turnover_cap: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
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
            
            # Sizing logic: with or without 2% turnover cap
            if apply_turnover_cap:
                cash_budget = min(start_capital * allocation_per_position, row.signal_day_turnover * 0.02)
            else:
                cash_budget = start_capital * allocation_per_position
                
            if cash_budget <= 0:
                continue
                
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
        daily.append({
            "signal_date": signal_date,
            "target_date": target_date,
            "capital_start": start_capital,
            "pnl": pnl,
            "capital_end": capital,
            "portfolio_return": pnl / start_capital if start_capital else 0.0,
            "trade_count": trade_count,
            "zero_trade_day": trade_count == 0
        })
    return pd.DataFrame(trades), pd.DataFrame(daily), ambiguity


def main():
    print("Loading all backtest candidates...", flush=True)
    all_candidates = candidates()
    periods = load_fee_schedule(FEES)
    signal_dates = pd.DatetimeIndex(sorted(all_candidates["signal_date"].unique()))
    
    models = [
        ("baseline", "Baseline q99.97"),
        ("raw_tail", "Corrected raw-tail q99.97"),
        ("merged", "Priority merge: Baseline then raw-tail"),
    ]
    
    exit_rules = [
        ("SL 4% / TP 8% (conservative)", 0.04, 0.08, "stop_first"),
        ("SL 10% / close exit", 0.10, None, "close_only"),
        ("SL 5% / close exit", 0.05, None, "close_only"),
    ]
    
    allocations = [0.20, 0.33]
    cap_options = [
        (True, "With 2% Turnover Cap"),
        (False, "Without Cap (Uncapped)"),
    ]
    
    rows = []
    yearly_rows = []
    
    for model_key, model_label in models:
        print(f"Selecting candidates for {model_label}...", flush=True)
        selected = select_for_model(all_candidates, model_key)
        for alloc in allocations:
            for exit_label, sl, tp, policy in exit_rules:
                for apply_cap, cap_label in cap_options:
                    trades, daily, ambiguity = simulate_custom(
                        signal_dates,
                        selected,
                        periods,
                        hard_stop_loss_pct=sl,
                        take_profit_pct=tp,
                        both_hit_policy=policy,
                        allocation_per_position=alloc,
                        apply_turnover_cap=apply_cap,
                    )
                    perf = performance_summary(trades, daily, 100_000.0)
                    stats = tabular_statistics(trades, daily)
                    
                    rows.append({
                        "Model": model_label,
                        "Cap Variation": cap_label,
                        "Allocation": f"{int(alloc*100)}%",
                        "Exit Rule": exit_label,
                        "Trades": int(stats["total_trades"]),
                        "Positive Trades": int(stats["positive_trades"]),
                        "Negative Trades": int(stats["negative_trades"]),
                        "Win Rate (%)": round(float(stats["win_rate"]) * 100, 2),
                        "Total Return (%)": round(float(perf["total_return"]) * 100, 2),
                        "CAGR (%)": round(float(perf["cagr"]) * 100, 2),
                        "Max Drawdown (%)": round(float(perf["max_drawdown"]) * 100, 2),
                        "Sharpe": round(float(perf["sharpe"]), 3),
                        "Profit Factor": round(float(stats["profit_factor"]), 3),
                        "Avg Net/Trade (%)": round(float(stats["average_net_return_per_trade"]) * 100, 3),
                    })

                    # Yearly breakdown
                    if not daily.empty:
                        daily_yr = daily.copy()
                        daily_yr["year"] = pd.to_datetime(daily_yr["signal_date"]).dt.year
                        trades_yr = trades.copy()
                        if not trades_yr.empty:
                            trades_yr["year"] = pd.to_datetime(trades_yr["signal_date"]).dt.year
                        for yr in range(2020, 2027):
                            dyr = daily_yr[daily_yr["year"] == yr]
                            tyr = trades_yr[trades_yr["year"] == yr] if not trades_yr.empty else trades_yr
                            num_trades = len(tyr)
                            pos = int((tyr["net_pnl"] > 0).sum()) if num_trades else 0
                            neg = int((tyr["net_pnl"] < 0).sum()) if num_trades else 0
                            wr = round(pos / num_trades * 100, 2) if num_trades else 0.0
                            if len(dyr):
                                c_start = dyr.iloc[0]["capital_start"]
                                c_end = dyr.iloc[-1]["capital_end"]
                                ret = round((c_end / c_start - 1) * 100, 2)
                                eq = dyr["capital_end"]
                                peak = eq.cummax()
                                intra_dd = round(((eq - peak) / peak).min() * 100, 2)
                            else:
                                ret = 0.0
                                intra_dd = 0.0
                            yearly_rows.append({
                                "Model": model_label,
                                "Cap Variation": cap_label,
                                "Allocation": f"{int(alloc*100)}%",
                                "Exit Rule": exit_label,
                                "Year": yr,
                                "Trades": num_trades,
                                "Positive": pos,
                                "Negative": neg,
                                "Win Rate (%)": wr,
                                "Return (%)": ret,
                                "Max Drawdown (%)": intra_dd,
                            })
    
    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT / "cap_comparison_summary.csv", index=False)
    yearly_df = pd.DataFrame(yearly_rows)
    yearly_df.to_csv(OUTPUT / "cap_comparison_yearwise.csv", index=False)
    print("Done! Summary table:")
    print(result_df.to_string(index=False))

if __name__ == "__main__":
    main()
