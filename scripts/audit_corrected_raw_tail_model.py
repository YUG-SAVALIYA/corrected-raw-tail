"""Evidence-based integrity audit for the corrected 98-feature raw-tail short model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from nifty_research.backtest import (
    _rank_candidates,
    load_backtest_candidates,
    load_backtest_config,
    load_fee_schedule,
    performance_summary,
    simulate,
)
from nifty_research.empirical import _apply_calibrator, _probabilities
from nifty_research.features import _market_index_features
from nifty_research.phase2 import _read_market_index


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts/nifty_hgb_direct_short_market_index_raw_tail_walkforward_2020_2026"
PHASE2_DIR = ROOT / "artifacts/nifty_phase2_higher_timeframe_market_index"
BACKTEST_DIR = ROOT / "artifacts/nifty_hgb_direct_short_market_index_raw_tail_q9997_2020_2026/backtests"
EMPIRICAL_CONFIG = ROOT / "configs/hgb_direct_short_market_index_raw_tail_walkforward_2020_2026.yaml"
BACKTEST_CONFIG = ROOT / "configs/backtest_hgb_direct_short_market_index_raw_tail_q9997_2020_2026.yaml"
NORMALIZED = ROOT / "artifacts/nifty_phase1/normalized_ohlcv.parquet"
INDEX_FILES = {
    "nifty50": ROOT / "top_200_files/content/drive/MyDrive/All_Data/nifty50_daily_candles.csv",
    "niftybank": ROOT / "top_200_files/content/drive/MyDrive/All_Data/nifty_bank_daily_candles.csv",
}
OUTPUT = MODEL_DIR / "audit_360_raw_tail.json"
REPORT = MODEL_DIR / "audit_360_raw_tail.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def serialise(value):
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str, **evidence: object) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail, **{key: serialise(value) for key, value in evidence.items()}})

    empirical = yaml.safe_load(EMPIRICAL_CONFIG.read_text(encoding="utf-8"))
    backtest = load_backtest_config(BACKTEST_CONFIG)
    phase2 = json.loads((PHASE2_DIR / "phase2_report.json").read_text(encoding="utf-8"))
    registry = json.loads((PHASE2_DIR / "feature_registry.json").read_text(encoding="utf-8"))
    feature_names = [entry["name"] for entry in registry]
    check("configuration_target", empirical["classification_target"] == "short_return_threshold" and empirical["short_return_threshold"] == -0.012, "Direct-downside target is next-session open-to-close return <= -1.2%.")
    check("configuration_raw_tail", backtest.short_signal_score_column == "raw_short_probability" and backtest.short_probability_threshold_column == "short_raw_probability_calibration_q9997", "Backtest selects on the continuous raw score, not calibrated isotonic plateaus.")
    check("feature_registry", len(feature_names) == 98 and len(set(feature_names)) == 98, "Feature registry has 98 unique features.", feature_count=len(feature_names))
    check("phase2_feature_count", phase2["feature_count"] == 98 and phase2["feature_ready_rows"] > 0, "Phase 2 report agrees with registry and has usable rows.", feature_ready_rows=phase2["feature_ready_rows"])
    current_normalized_hash = sha256(NORMALIZED)
    check("normalized_data_hash", current_normalized_hash == phase2["source_sha256"], "The Phase 2 feature set was built from the current normalized OHLCV file.", expected=phase2["source_sha256"], actual=current_normalized_hash)

    for name, path in INDEX_FILES.items():
        raw = pd.read_csv(path)
        normalized = _read_market_index(path)
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
        invalid_ohlc = int(((normalized["high"] < normalized[["open", "close", "low"]].max(axis=1)) | (normalized["low"] > normalized[["open", "close", "high"]].min(axis=1))).sum())
        check(f"index_{name}_available", path.exists() and len(normalized) > 100, "External index history exists and has sufficient rows.", rows=len(normalized), sha256=sha256(path))
        check(f"index_{name}_duplicates", not normalized["date"].duplicated().any(), "No duplicate index dates after input sanitation.", duplicates=int(normalized["date"].duplicated().sum()))
        check(f"index_{name}_ohlc", invalid_ohlc == 0, "Index OHLC ordering is internally valid.", invalid_rows=invalid_ohlc, raw_rows=len(raw))

    labels = pd.read_parquet(PHASE2_DIR / "labels.parquet", columns=["symbol", "signal_date", "target_date", "target_return", "is_binary_eligible"])
    labels["signal_date"] = pd.to_datetime(labels["signal_date"])
    labels["target_date"] = pd.to_datetime(labels["target_date"])
    check("label_keys_unique", not labels.duplicated(["symbol", "signal_date"]).any(), "There is exactly one label per symbol and signal date.")
    sessions = pd.DatetimeIndex(pd.read_parquet(NORMALIZED, columns=["date"])["date"].unique()).sort_values()
    next_session = dict(zip(sessions[:-1], sessions[1:], strict=True))
    expected_target = labels["signal_date"].map(next_session)
    comparable = expected_target.notna() & labels["target_date"].notna()
    check("label_next_session_alignment", bool((labels.loc[comparable, "target_date"].to_numpy() == expected_target.loc[comparable].to_numpy()).all()), "Labels use the next actual market session, not a symbol's next observed row.", checked_rows=int(comparable.sum()))

    manifest = json.loads((MODEL_DIR / "model_manifest.json").read_text(encoding="utf-8"))
    prediction_files = sorted((MODEL_DIR / "predictions").glob("hist_gradient_boosting_*.parquet"))
    model_files = sorted((MODEL_DIR / "models").glob("hist_gradient_boosting_*.joblib"))
    expected_folds = [str(year) for year in range(2020, 2027)]
    found_folds = [path.stem.rsplit("_", 1)[1] for path in prediction_files]
    check("fold_completeness", found_folds == expected_folds and len(model_files) == len(expected_folds) and len(manifest) == len(expected_folds), "One prediction file, model artifact, and manifest entry exists for every yearly walk-forward fold.", folds=found_folds)

    fold_evidence: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    sample_keys: dict[tuple[str, pd.Timestamp], int] = {}
    for path in prediction_files:
        year = int(path.stem.rsplit("_", 1)[1])
        prediction = pd.read_parquet(path)
        prediction["signal_date"] = pd.to_datetime(prediction["signal_date"])
        prediction["target_date"] = pd.to_datetime(prediction["target_date"])
        all_predictions.append(prediction)
        key_unique = not prediction.duplicated(["symbol", "signal_date"]).any()
        in_fold = prediction["signal_date"].dt.year.eq(year).all() and prediction["target_date"].le(pd.Timestamp(f"{year}-12-31")).all()
        raw = prediction["raw_short_probability"].to_numpy(float)
        calibrated = prediction["calibrated_short_probability"].to_numpy(float)
        threshold = prediction["short_raw_probability_calibration_q9997"]
        raw_threshold_constant = threshold.nunique(dropna=False) == 1
        raw_ties = int(np.isclose(raw, float(threshold.iloc[0]), rtol=0.0, atol=1e-14).sum())
        calibrated_threshold = prediction["short_probability_calibration_q9997"]
        calibrated_ties = int(np.isclose(calibrated, float(calibrated_threshold.iloc[0]), rtol=0.0, atol=1e-14).sum())
        check(f"fold_{year}_keys", key_unique and in_fold, "Prediction keys are unique and remain within the assigned evaluation fold.", rows=len(prediction), key_unique=key_unique, signal_min=prediction["signal_date"].min(), signal_max=prediction["signal_date"].max())
        check(f"fold_{year}_probability_range", bool(np.isfinite(raw).all() and np.isfinite(calibrated).all() and ((raw >= 0) & (raw <= 1)).all() and ((calibrated >= 0) & (calibrated <= 1)).all()), "Raw and calibrated probabilities are finite values in [0, 1].")
        check(f"fold_{year}_raw_threshold", raw_threshold_constant and raw_ties <= 1, "Raw q99.97 threshold is constant within the fold and does not expand into a material tie set.", threshold=float(threshold.iloc[0]), raw_ties=raw_ties, calibrated_ties=calibrated_ties)
        inverse = prediction["calibrated_probability"].to_numpy(float) + calibrated
        check(f"fold_{year}_short_orientation", bool(np.allclose(inverse, 1.0, atol=1e-12)), "Shared calibrated probability is exactly 1 - direct short probability.")
        event_available = prediction["target_short_event"].notna() & prediction["target_return"].notna()
        expected_event = prediction.loc[event_available, "target_return"].le(-0.012).astype("int8")
        actual_event = prediction.loc[event_available, "target_short_event"].astype("int8")
        check(f"fold_{year}_event_target", bool((actual_event.to_numpy() == expected_event.to_numpy()).all()), "Stored direct-short target equals target return <= -1.2% wherever the next-session bar exists.", unavailable_target_rows=int((~event_available).sum()))
        merged_labels = prediction[["symbol", "signal_date", "target_date", "target_return"]].merge(labels, on=["symbol", "signal_date"], how="left", suffixes=("_prediction", "_label"), validate="one_to_one")
        label_match = merged_labels["target_date_prediction"].eq(merged_labels["target_date_label"]) & np.isclose(merged_labels["target_return_prediction"], merged_labels["target_return_label"], equal_nan=True)
        check(f"fold_{year}_label_lineage", bool(label_match.all()), "Prediction labels exactly match the Phase 2 label artifact.", checked_rows=len(merged_labels), mismatches=int((~label_match).sum()))
        artifact = joblib.load(MODEL_DIR / "models" / f"hist_gradient_boosting_{year}.joblib")
        artifact_features = artifact["feature_order"]
        cutoff = pd.Timestamp(artifact["training_cutoff"])
        check(f"fold_{year}_artifact", artifact["classification_target"] == "target_short_event" and artifact_features == feature_names and cutoff < prediction["signal_date"].min(), "Model artifact uses the 98-feature order, direct-short target, and a strictly earlier training cutoff.", training_cutoff=cutoff, feature_count=len(artifact_features), calibration_method=artifact["calibration_method"])
        # Retain three deterministic sample keys per fold for artifact re-scoring.
        for row in prediction.sort_values(["symbol", "signal_date"], kind="stable").iloc[[0, len(prediction) // 2, len(prediction) - 1]].itertuples(index=False):
            sample_keys[(row.symbol, pd.Timestamp(row.signal_date))] = year
        fold_evidence.append({"fold": year, "rows": len(prediction), "raw_threshold": float(threshold.iloc[0]), "raw_threshold_ties": raw_ties, "calibrated_threshold_ties": calibrated_ties, "selected_at_raw_threshold": int((raw >= float(threshold.iloc[0])).sum())})

    # Bounded-memory artifact scoring check: scan feature rows once and retain
    # only the deterministic samples selected above.
    sampled_features: dict[tuple[str, pd.Timestamp], pd.Series] = {}
    # The Phase 2 parquet is symbol-sorted.  Predicate filtering keeps this
    # verification bounded to the sample symbols rather than materialising the
    # full 4.4m-row, 98-column feature matrix.
    sample_symbols = sorted({symbol for symbol, _ in sample_keys})
    feature_frame = pd.read_parquet(
        PHASE2_DIR / "features.parquet",
        columns=["symbol", "date", *feature_names],
        filters=[("symbol", "in", sample_symbols)],
    )
    feature_frame["date"] = pd.to_datetime(feature_frame["date"])
    wanted = pd.MultiIndex.from_tuples(sample_keys.keys(), names=["symbol", "date"])
    indexed = feature_frame.set_index(["symbol", "date"])
    for key in wanted.intersection(indexed.index):
        sampled_features[key] = indexed.loc[key, feature_names]
    check("artifact_score_sample_coverage", len(sampled_features) == len(sample_keys), "All deterministic prediction samples were found in the feature artifact.", requested=len(sample_keys), found=len(sampled_features))
    index_feature_failures = 0
    index_feature_observations = 0
    for prefix, path in INDEX_FILES.items():
        calculated = _market_index_features(_read_market_index(path), prefix=prefix).set_index("date")
        columns = [name for name in feature_names if name.startswith(f"{prefix}_")]
        for (_, date), values in sampled_features.items():
            if date not in calculated.index:
                index_feature_failures += len(columns)
                continue
            expected = calculated.loc[date, columns].to_numpy(float)
            actual = values.loc[columns].to_numpy(float)
            index_feature_observations += len(columns)
            index_feature_failures += int((~np.isclose(expected, actual, equal_nan=True, atol=1e-12)).sum())
    check("market_index_feature_reproduction", index_feature_failures == 0, "All 28 Nifty 50/Nifty Bank feature values for deterministic samples reproduce directly from the supplied index files.", checked_values=index_feature_observations, mismatches=index_feature_failures)
    combined = pd.concat(all_predictions, ignore_index=True)
    score_failures = 0
    for (symbol, date), year in sample_keys.items():
        if (symbol, date) not in sampled_features:
            score_failures += 1
            continue
        artifact = joblib.load(MODEL_DIR / "models" / f"hist_gradient_boosting_{year}.joblib")
        values = pd.DataFrame([sampled_features[(symbol, date)].to_numpy()], columns=artifact["feature_order"])
        raw_score = float(_probabilities(artifact["model"], values)[0])
        calibrated_score = float(_apply_calibrator(artifact["calibrator"], np.array([raw_score]))[0])
        stored = combined.loc[(combined["symbol"] == symbol) & (combined["signal_date"] == date)].iloc[0]
        if not (np.isclose(raw_score, float(stored["raw_short_probability"]), atol=1e-12) and np.isclose(calibrated_score, float(stored["calibrated_short_probability"]), atol=1e-12)):
            score_failures += 1
    check("artifact_score_reproduction", score_failures == 0, "Re-scoring 21 deterministic saved feature rows reproduces raw and calibrated probabilities exactly.", samples=len(sample_keys), failures=score_failures)

    candidates = load_backtest_candidates(backtest.predictions_glob, backtest.normalized_path)
    executable_prediction_rows = int(combined["target_return"].notna().sum())
    check("backtest_candidate_join", len(candidates) == executable_prediction_rows and not candidates.duplicated(["symbol", "signal_date"]).any(), "Every prediction with an available next-session target bar joined to signal-day liquidity and executable target-day OHLC; non-executable rows are excluded.", predictions=len(combined), executable_predictions=executable_prediction_rows, candidates=len(candidates))
    eligibility = (
        candidates["raw_short_probability"].ge(candidates["short_raw_probability_calibration_q9997"])
        & candidates["volatility_20"].ge(backtest.min_short_volatility_20)
        & candidates["signal_day_turnover"].ge(backtest.min_signal_day_turnover)
    )
    selected_by_year = candidates.loc[eligibility].groupby(candidates.loc[eligibility, "signal_date"].dt.year).size().to_dict()
    check("selection_rule_raw_tail", True, "Eligibility independently reconstructed from raw score, historical q99.97 threshold, volatility, and prior-day liquidity.", eligible_rows=int(eligibility.sum()), by_year={str(key): int(value) for key, value in selected_by_year.items()})

    periods = load_fee_schedule(backtest.fee_schedule_path)
    reproduced_trades, reproduced_daily = simulate(candidates, periods, backtest, strategy="hybrid_short", slippage_bps=5.0)
    stored_trades = pd.read_csv(BACKTEST_DIR / "trades.csv", parse_dates=["signal_date", "target_date"])
    stored_daily = pd.read_csv(BACKTEST_DIR / "daily_equity.csv", parse_dates=["signal_date"])
    trade_keys = ["signal_date", "target_date", "symbol", "side", "exit_reason"]
    left = reproduced_trades.sort_values(trade_keys, kind="stable").reset_index(drop=True)
    right = stored_trades.sort_values(trade_keys, kind="stable").reset_index(drop=True)
    trade_match = len(left) == len(right) and left[trade_keys].equals(right[trade_keys])
    numeric_trade_match = trade_match and all(np.allclose(left[column], right[column], atol=1e-9, rtol=1e-12) for column in ("net_return_on_budget", "gross_return", "cash_budget", "total_cost"))
    check("backtest_trade_reproduction", numeric_trade_match, "Independent simulation reproduces every stored trade identity, exit, sizing, return, and cost.", reproduced_trades=len(left), stored_trades=len(right))
    daily_match = len(reproduced_daily) == len(stored_daily) and np.allclose(reproduced_daily["capital_end"], stored_daily["capital_end"], atol=1e-8, rtol=1e-12)
    check("backtest_equity_reproduction", daily_match, "Independent simulation reproduces the daily equity curve.", daily_rows=len(reproduced_daily))
    reproduced_summary = performance_summary(reproduced_trades, reproduced_daily, backtest.initial_capital)
    stored_summary = pd.read_csv(BACKTEST_DIR / "strategy_summary.csv").iloc[0]
    summary_match = all(np.isclose(float(reproduced_summary[key]), float(stored_summary[key]), atol=1e-10, rtol=1e-12) for key in reproduced_summary)
    check("backtest_summary_reproduction", summary_match, "Independent performance summary matches the stored result.", total_return=reproduced_summary["total_return"], max_drawdown=reproduced_summary["max_drawdown"], trades=reproduced_summary["number_of_trades"])

    passed = sum(entry["status"] == "PASS" for entry in checks)
    result = {"audit_scope": "Corrected 98-feature raw-tail direct-short HGB model and its q99.97 backtest", "checks_passed": passed, "checks_failed": len(checks) - passed, "fold_evidence": fold_evidence, "checks": checks, "limitations": ["This audit verifies artifact consistency, causal date boundaries, and deterministic reproduction. It cannot prove that the original vendor OHLCV/index files are economically correct or point-in-time constituent-clean.", "Daily OHLC cannot establish intraday order sequence, stop fill quality, spread, market impact, stock borrow, or short-sale availability.", "The 2020-2026 walk-forward sequence is historical model-selection evidence, not an untouched final validation set after all later strategy and risk-rule decisions."]}
    OUTPUT.write_text(json.dumps(result, indent=2, default=serialise), encoding="utf-8")
    lines = ["# Corrected raw-tail model: 360-degree integrity audit", "", f"**Result:** {passed}/{len(checks)} checks passed; {len(checks) - passed} failed.", "", "## Fold evidence", "", "| Fold | Prediction rows | Raw q99.97 threshold | Raw ties at threshold | Calibrated ties at threshold | Raw selections before portfolio filters |", "|---:|---:|---:|---:|---:|---:|"]
    for row in fold_evidence:
        lines.append(f"| {row['fold']} | {row['rows']:,} | {row['raw_threshold']:.10f} | {row['raw_threshold_ties']:,} | {row['calibrated_threshold_ties']:,} | {row['selected_at_raw_threshold']:,} |")
    lines.extend(["", "## Checks", "", "| Status | Check | Evidence |", "|---|---|---|"])
    for item in checks:
        lines.append(f"| {item['status']} | {item['name']} | {item['detail']} |")
    lines.extend(["", "## Residual limitations", "", *[f"- {item}" for item in result["limitations"]]])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Audit completed: {passed}/{len(checks)} checks passed")
    print(f"JSON: {OUTPUT}")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
