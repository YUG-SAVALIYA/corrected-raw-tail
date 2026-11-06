# Corrected raw-tail model: 360-degree integrity audit

**Result:** 71/71 checks passed; 0 failed.

## Fold evidence

| Fold | Prediction rows | Raw q99.97 threshold | Raw ties at threshold | Calibrated ties at threshold | Raw selections before portfolio filters |
|---:|---:|---:|---:|---:|---:|
| 2020 | 310,971 | 0.8244717072 | 0 | 0 | 1 |
| 2021 | 332,325 | 0.7914425944 | 0 | 16,646 | 17 |
| 2022 | 359,257 | 0.7887159832 | 0 | 16 | 63 |
| 2023 | 390,713 | 0.7762539548 | 0 | 0 | 306 |
| 2024 | 423,478 | 0.7744031898 | 0 | 0 | 109 |
| 2025 | 471,800 | 0.7771354679 | 0 | 740 | 153 |
| 2026 | 376,782 | 0.7799767851 | 0 | 535 | 576 |

## Checks

| Status | Check | Evidence |
|---|---|---|
| PASS | configuration_target | Direct-downside target is next-session open-to-close return <= -1.2%. |
| PASS | configuration_raw_tail | Backtest selects on the continuous raw score, not calibrated isotonic plateaus. |
| PASS | feature_registry | Feature registry has 98 unique features. |
| PASS | phase2_feature_count | Phase 2 report agrees with registry and has usable rows. |
| PASS | normalized_data_hash | The Phase 2 feature set was built from the current normalized OHLCV file. |
| PASS | index_nifty50_available | External index history exists and has sufficient rows. |
| PASS | index_nifty50_duplicates | No duplicate index dates after input sanitation. |
| PASS | index_nifty50_ohlc | Index OHLC ordering is internally valid. |
| PASS | index_niftybank_available | External index history exists and has sufficient rows. |
| PASS | index_niftybank_duplicates | No duplicate index dates after input sanitation. |
| PASS | index_niftybank_ohlc | Index OHLC ordering is internally valid. |
| PASS | label_keys_unique | There is exactly one label per symbol and signal date. |
| PASS | label_next_session_alignment | Labels use the next actual market session, not a symbol's next observed row. |
| PASS | fold_completeness | One prediction file, model artifact, and manifest entry exists for every yearly walk-forward fold. |
| PASS | fold_2020_keys | Prediction keys are unique and remain within the assigned evaluation fold. |
| PASS | fold_2020_probability_range | Raw and calibrated probabilities are finite values in [0, 1]. |
| PASS | fold_2020_raw_threshold | Raw q99.97 threshold is constant within the fold and does not expand into a material tie set. |
| PASS | fold_2020_short_orientation | Shared calibrated probability is exactly 1 - direct short probability. |
| PASS | fold_2020_event_target | Stored direct-short target equals target return <= -1.2% wherever the next-session bar exists. |
| PASS | fold_2020_label_lineage | Prediction labels exactly match the Phase 2 label artifact. |
| PASS | fold_2020_artifact | Model artifact uses the 98-feature order, direct-short target, and a strictly earlier training cutoff. |
| PASS | fold_2021_keys | Prediction keys are unique and remain within the assigned evaluation fold. |
| PASS | fold_2021_probability_range | Raw and calibrated probabilities are finite values in [0, 1]. |
| PASS | fold_2021_raw_threshold | Raw q99.97 threshold is constant within the fold and does not expand into a material tie set. |
| PASS | fold_2021_short_orientation | Shared calibrated probability is exactly 1 - direct short probability. |
| PASS | fold_2021_event_target | Stored direct-short target equals target return <= -1.2% wherever the next-session bar exists. |
| PASS | fold_2021_label_lineage | Prediction labels exactly match the Phase 2 label artifact. |
| PASS | fold_2021_artifact | Model artifact uses the 98-feature order, direct-short target, and a strictly earlier training cutoff. |
| PASS | fold_2022_keys | Prediction keys are unique and remain within the assigned evaluation fold. |
| PASS | fold_2022_probability_range | Raw and calibrated probabilities are finite values in [0, 1]. |
| PASS | fold_2022_raw_threshold | Raw q99.97 threshold is constant within the fold and does not expand into a material tie set. |
| PASS | fold_2022_short_orientation | Shared calibrated probability is exactly 1 - direct short probability. |
| PASS | fold_2022_event_target | Stored direct-short target equals target return <= -1.2% wherever the next-session bar exists. |
| PASS | fold_2022_label_lineage | Prediction labels exactly match the Phase 2 label artifact. |
| PASS | fold_2022_artifact | Model artifact uses the 98-feature order, direct-short target, and a strictly earlier training cutoff. |
| PASS | fold_2023_keys | Prediction keys are unique and remain within the assigned evaluation fold. |
| PASS | fold_2023_probability_range | Raw and calibrated probabilities are finite values in [0, 1]. |
| PASS | fold_2023_raw_threshold | Raw q99.97 threshold is constant within the fold and does not expand into a material tie set. |
| PASS | fold_2023_short_orientation | Shared calibrated probability is exactly 1 - direct short probability. |
| PASS | fold_2023_event_target | Stored direct-short target equals target return <= -1.2% wherever the next-session bar exists. |
| PASS | fold_2023_label_lineage | Prediction labels exactly match the Phase 2 label artifact. |
| PASS | fold_2023_artifact | Model artifact uses the 98-feature order, direct-short target, and a strictly earlier training cutoff. |
| PASS | fold_2024_keys | Prediction keys are unique and remain within the assigned evaluation fold. |
| PASS | fold_2024_probability_range | Raw and calibrated probabilities are finite values in [0, 1]. |
| PASS | fold_2024_raw_threshold | Raw q99.97 threshold is constant within the fold and does not expand into a material tie set. |
| PASS | fold_2024_short_orientation | Shared calibrated probability is exactly 1 - direct short probability. |
| PASS | fold_2024_event_target | Stored direct-short target equals target return <= -1.2% wherever the next-session bar exists. |
| PASS | fold_2024_label_lineage | Prediction labels exactly match the Phase 2 label artifact. |
| PASS | fold_2024_artifact | Model artifact uses the 98-feature order, direct-short target, and a strictly earlier training cutoff. |
| PASS | fold_2025_keys | Prediction keys are unique and remain within the assigned evaluation fold. |
| PASS | fold_2025_probability_range | Raw and calibrated probabilities are finite values in [0, 1]. |
| PASS | fold_2025_raw_threshold | Raw q99.97 threshold is constant within the fold and does not expand into a material tie set. |
| PASS | fold_2025_short_orientation | Shared calibrated probability is exactly 1 - direct short probability. |
| PASS | fold_2025_event_target | Stored direct-short target equals target return <= -1.2% wherever the next-session bar exists. |
| PASS | fold_2025_label_lineage | Prediction labels exactly match the Phase 2 label artifact. |
| PASS | fold_2025_artifact | Model artifact uses the 98-feature order, direct-short target, and a strictly earlier training cutoff. |
| PASS | fold_2026_keys | Prediction keys are unique and remain within the assigned evaluation fold. |
| PASS | fold_2026_probability_range | Raw and calibrated probabilities are finite values in [0, 1]. |
| PASS | fold_2026_raw_threshold | Raw q99.97 threshold is constant within the fold and does not expand into a material tie set. |
| PASS | fold_2026_short_orientation | Shared calibrated probability is exactly 1 - direct short probability. |
| PASS | fold_2026_event_target | Stored direct-short target equals target return <= -1.2% wherever the next-session bar exists. |
| PASS | fold_2026_label_lineage | Prediction labels exactly match the Phase 2 label artifact. |
| PASS | fold_2026_artifact | Model artifact uses the 98-feature order, direct-short target, and a strictly earlier training cutoff. |
| PASS | artifact_score_sample_coverage | All deterministic prediction samples were found in the feature artifact. |
| PASS | market_index_feature_reproduction | All 28 Nifty 50/Nifty Bank feature values for deterministic samples reproduce directly from the supplied index files. |
| PASS | artifact_score_reproduction | Re-scoring 21 deterministic saved feature rows reproduces raw and calibrated probabilities exactly. |
| PASS | backtest_candidate_join | Every prediction with an available next-session target bar joined to signal-day liquidity and executable target-day OHLC; non-executable rows are excluded. |
| PASS | selection_rule_raw_tail | Eligibility independently reconstructed from raw score, historical q99.97 threshold, volatility, and prior-day liquidity. |
| PASS | backtest_trade_reproduction | Independent simulation reproduces every stored trade identity, exit, sizing, return, and cost. |
| PASS | backtest_equity_reproduction | Independent simulation reproduces the daily equity curve. |
| PASS | backtest_summary_reproduction | Independent performance summary matches the stored result. |

## Residual limitations

- This audit verifies artifact consistency, causal date boundaries, and deterministic reproduction. It cannot prove that the original vendor OHLCV/index files are economically correct or point-in-time constituent-clean.
- Daily OHLC cannot establish intraday order sequence, stop fill quality, spread, market impact, stock borrow, or short-sale availability.
- The 2020-2026 walk-forward sequence is historical model-selection evidence, not an untouched final validation set after all later strategy and risk-rule decisions.
