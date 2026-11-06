# Corrected raw-tail HGB short model: reproduction bundle

## Contents

- `src/nifty_research/`: the feature, walk-forward HGB, scoring and backtest implementation.
- `configs/`: exact 98-feature raw-tail training/backtest settings and cost schedule.
- `artifacts/nifty_hgb_direct_short_market_index_raw_tail_walkforward_2020_2026/`: seven fitted walk-forward HGB model files, model manifest, out-of-sample predictions, metrics and the 360-degree audit.
- `deliverables/nifty_direct_short_hgb_frozen_2026_10_28/frozen_predictions/`: frozen Baseline q99.97 prediction files required for priority merging.
- `scripts/`: standalone audit, raw-tail/priority merge runners and final comparison scripts.
- `results/`: standalone raw-tail and merged results, including year-wise tables under the final shared execution settings.

The 1.86 GB phase-2 feature matrix and source OHLCV dataset are intentionally excluded from the ZIP. Recreate them locally from the data sources, or copy the existing files into the paths below.

## Environment

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Required source artifacts (not bundled):

```text
artifacts/nifty_phase1/normalized_ohlcv.parquet
artifacts/nifty_phase2_higher_timeframe_market_index/features.parquet
artifacts/nifty_phase2_higher_timeframe_market_index/labels.parquet
```

If the phase-2 files are absent, create them first from normalized OHLCV:

```powershell
nifty features --config configs/features_higher_timeframe_market_index.yaml
```

## Reproduce the corrected raw-tail model

The model is a yearly expanding walk-forward `HistGradientBoostingClassifier`, trained on 98 completed-candle features (70 stock features plus 28 Nifty 50/Nifty Bank features). The direct label is next-session open-to-close return <= -1.2%. Each year is scored only by a model trained before that year.

```powershell
nifty empirical --config configs/hgb_direct_short_market_index_raw_tail_walkforward_2020_2026.yaml
python scripts/audit_corrected_raw_tail_model.py
```

The correction is deliberate: selection uses `raw_short_probability >= short_raw_probability_calibration_q9997`, not calibrated-probability threshold ties. The audit should report 71/71 checks passing when the source data and package versions reproduce exactly.

Standalone raw-tail backtest (the original raw q99.97 path):

```powershell
nifty backtest --config configs/backtest_hgb_direct_short_market_index_raw_tail_q9997_2020_2026.yaml
```

## How the Baseline and corrected raw-tail models are merged

The priority merge is **not an average of model scores**:

1. Baseline candidate: `calibrated_short_probability >= short_probability_calibration_q9997` and `volatility_20 >= 0.036`.
2. Corrected raw-tail candidate: `raw_short_probability >= short_raw_probability_calibration_q9997` and `volatility_20 >= 0.036`.
3. For the final comparison, candidates with `target_open / signal_close - 1 <= -3%` are removed **before** ranking and before any of the three slots is filled.
4. For each signal date, sort Baseline first, then raw-tail; within a source sort score descending, then symbol ascending. Remove duplicate symbol/date rows and retain the first three candidates.
5. Therefore Baseline consumes capacity first; raw-tail is only a capacity filler. Do not run the standalone models beside the merge, as that duplicates exposure.

Original close-exit merge runner:

```powershell
python scripts/backtest_priority_baseline_raw_tail_merge.py --hard-stop-loss-pct 0.05 --max-allocation-per-stock 0.20
```

Final shared comparison (gap rule, 20%/33%, 5%/10% stops, and conservative 4% SL / 8% TP):

```powershell
python scripts/build_short_model_comparison_dashboard.py
```

Open `artifacts/short_model_gap_down3_comparison_dashboard/short_model_comparison_dashboard.html` after that command.

## Final 4% SL / 8% TP result files

`results/standalone_raw_tail_final_4sl_8tp_gapdown3_alloc20_yearwise.csv` and the two `merged_final_4sl_8tp_gapdown3_*_yearwise.csv` files contain the requested year-wise results. They use the gap filter before slot filling and the conservative stop-first rule whenever daily OHLC reaches both stop and target.

## Important limitations

This is a research/reproduction package, not a trading recommendation. Daily OHLC cannot prove intraday stop/target order; the gap rule is observable only at the open while the backtest uses an official-open fill; borrowing availability/cost, market impact and partial fills are not fully represented. Validate using intraday data and a paper-trading process before deployment.
