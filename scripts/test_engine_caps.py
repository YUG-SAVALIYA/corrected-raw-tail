import pandas as pd
from nifty_research.backtest import BacktestConfig, simulate, load_fee_schedule, performance_summary, load_backtest_candidates
from pathlib import Path

ROOT = Path('.')
NORMALIZED = ROOT / 'artifacts/nifty_phase1/normalized_ohlcv.parquet'
FEES = ROOT / 'configs/costs.yaml'
BASELINE = ROOT / 'deliverables/nifty_direct_short_hgb_frozen_2026_10_28/frozen_predictions/hist_gradient_boosting_*.parquet'
RAW_TAIL = ROOT / 'artifacts/nifty_hgb_direct_short_market_index_raw_tail_walkforward_2020_2026/predictions/hist_gradient_boosting_*.parquet'

base = load_backtest_candidates(str(BASELINE), NORMALIZED)
base['short_preselected'] = base['calibrated_short_probability'].ge(base['short_probability_calibration_q9997']) & base['volatility_20'].ge(0.036)
base['source_score'] = base['calibrated_short_probability']
base['signal_source'] = 'baseline_q9997'
base['signal_source_priority'] = 0

raw = load_backtest_candidates(str(RAW_TAIL), NORMALIZED)
raw['short_preselected'] = raw['raw_short_probability'].ge(raw['short_raw_probability_calibration_q9997']) & raw['volatility_20'].ge(0.036)
raw['source_score'] = raw['raw_short_probability']
raw['signal_source'] = 'raw_tail_q9997'
raw['signal_source_priority'] = 1

merged = pd.concat([base, raw], ignore_index=True, sort=False)
periods = load_fee_schedule(FEES)

print("--- STANDARD ENGINE RUN (CLOSE-EXIT, SL 10%, ALLOC 33%) ---")
for name, candidates_df, strat in [
    ('Baseline q99.97', base, 'priority_hybrid_short'),
    ('Corrected raw-tail q99.97', raw, 'priority_hybrid_short'),
    ('Priority merge', merged, 'priority_hybrid_short')
]:
    for cap_val, cap_name in [(0.02, '2% turnover cap'), (1.0, 'No cap')]:
        cfg = BacktestConfig(
            predictions_glob='unused',
            normalized_path=NORMALIZED,
            fee_schedule_path=FEES,
            initial_capital=100000,
            max_positions=3,
            max_allocation_per_stock=0.33,
            max_turnover_fraction_per_stock=cap_val,
            min_signal_day_turnover=5000000,
            strategies=[strat],
            slippage_bps_per_side=[5.0],
            min_short_red_probability=0.0,
            require_short_return_prediction=False,
            min_short_volatility_20=0.036,
            short_volatility_rank_penalty=0.0,
            hard_stop_loss_pct=0.10
        )
        trades, daily = simulate(candidates_df, periods, cfg, strategy=strat, slippage_bps=5.0)
        perf = performance_summary(trades, daily, 100000)
        wins = int((trades['net_return_on_budget'] > 0).sum())
        losses = int((trades['net_return_on_budget'] < 0).sum())
        ret = perf['total_return'] * 100
        cagr = perf['cagr'] * 100
        mdd = perf['max_drawdown'] * 100
        sharpe = perf['sharpe']
        wr = wins / len(trades) * 100 if len(trades) else 0.0
        print(f"{name} [{cap_name}]: Trades={len(trades)}, Pos={wins}, Neg={losses}, WR={wr:.2f}%, Ret={ret:.2f}%, CAGR={cagr:.2f}%, MaxDD={mdd:.2f}%, Sharpe={sharpe:.3f}")

