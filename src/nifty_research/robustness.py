"""Phase 6 repeatable robustness experiments and conservative research reporting."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field

from .backtest import BacktestConfig, load_backtest_candidates, load_fee_schedule, performance_summary, simulate
from .baselines import classification_metrics, deterministic_train_sample
from .features import default_feature_registry


class RobustnessConfig(BaseModel):
    features_path: Path
    phase4_dir: Path
    normalized_path: Path
    fee_schedule_path: Path
    output_dir: Path = Path("artifacts/nifty_phase6")
    ablation_max_rows_per_fold: int = Field(default=50_000, ge=1_000)
    bootstrap_repetitions: int = Field(default=500, ge=100)
    bootstrap_block_days: int = Field(default=20, ge=2)
    random_state: int = 42


def load_robustness_config(path: str | Path) -> RobustnessConfig:
    config_path = Path(path).resolve()
    config = RobustnessConfig.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    for field in ("features_path", "phase4_dir", "normalized_path", "fee_schedule_path", "output_dir"):
        value = getattr(config, field)
        if not value.is_absolute():
            setattr(config, field, (config_path.parent / value).resolve())
    return config


def block_bootstrap_mean(values: pd.Series, *, repetitions: int, block_size: int, random_state: int) -> tuple[float, float, float]:
    """Chronological moving-block bootstrap interval for a daily mean."""

    sample = np.asarray(values.dropna(), dtype=float)
    if not len(sample):
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(random_state)
    draws = np.empty(repetitions)
    for draw in range(repetitions):
        pieces: list[np.ndarray] = []
        while sum(map(len, pieces)) < len(sample):
            start = rng.integers(0, len(sample))
            indexes = (start + np.arange(block_size)) % len(sample)
            pieces.append(sample[indexes])
        draws[draw] = np.concatenate(pieces)[: len(sample)].mean()
    return (float(sample.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def _apply_calibrator(artifact: dict, probability: np.ndarray) -> np.ndarray:
    calibrator = artifact["calibrator"]
    if calibrator is None:
        return probability
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(probability.reshape(-1, 1))[:, 1]
    return calibrator.predict(probability)


def _reliability(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (fold, model), group in predictions.dropna(subset=["target_green"]).groupby(["fold", "model"], sort=True):
        values = group.copy()
        values["bucket"] = pd.cut(values["calibrated_probability"], bins=np.linspace(0, 1, 11), include_lowest=True)
        for bucket, subset in values.groupby("bucket", observed=True):
            rows.append({"fold": fold, "model": model, "bucket": str(bucket), "observations": len(subset), "mean_predicted_probability": float(subset["calibrated_probability"].mean()), "observed_green_rate": float(subset["target_green"].mean())})
    return pd.DataFrame(rows)


def _ranking_comparison(predictions: pd.DataFrame, config: RobustnessConfig) -> pd.DataFrame:
    rows: list[dict] = []
    usable = predictions.dropna(subset=["target_green", "target_return"])
    for year, group in usable.groupby(usable["signal_date"].dt.year, sort=True):
        daily_classification = group.sort_values(["signal_date", "calibrated_probability"], ascending=[True, False], kind="stable").groupby("signal_date", sort=False).head(5).groupby("signal_date", sort=False)["target_return"].mean()
        daily_regression = group.sort_values(["signal_date", "predicted_return"], ascending=[True, False], kind="stable").groupby("signal_date", sort=False).head(5).groupby("signal_date", sort=False)["target_return"].mean()
        for name, daily in (("classification_probability_rank", daily_classification), ("regression_return_rank", daily_regression)):
            mean, low, high = block_bootstrap_mean(daily, repetitions=config.bootstrap_repetitions, block_size=config.bootstrap_block_days, random_state=config.random_state)
            rows.append({"year": int(year), "ranking": name, "daily_top5_mean_realized_return": mean, "block_bootstrap_ci_low": low, "block_bootstrap_ci_high": high, "trading_days": len(daily)})
    return pd.DataFrame(rows)


def _permutation_ablation(features: pd.DataFrame, manifest: list[dict], config: RobustnessConfig) -> pd.DataFrame:
    groups: dict[str, list[str]] = {}
    for spec in default_feature_registry():
        groups.setdefault(spec.group, []).append(spec.name)
    rows: list[dict] = []
    rng = np.random.default_rng(config.random_state)
    for item in manifest:
        if item["model"] != "lightgbm_classifier":
            continue
        prediction = pd.read_parquet(item["prediction_path"]).dropna(subset=["target_green"])
        sample = deterministic_train_sample(prediction, min(config.ablation_max_rows_per_fold, len(prediction)))
        subset = features.merge(sample[["symbol", "signal_date", "target_green"]], left_on=["symbol", "date"], right_on=["symbol", "signal_date"], how="inner", validate="one_to_one")
        artifact = joblib.load(item["artifact_path"])
        order = artifact["feature_order"]
        baseline_raw = artifact["model"].predict_proba(subset[order])[:, 1]
        baseline = _apply_calibrator(artifact, baseline_raw)
        base_metric = classification_metrics(subset["target_green"], baseline, subset["signal_date"], top_k=5)
        for group, columns in groups.items():
            selected = [column for column in columns if column in order]
            if not selected:
                continue
            altered = subset[order].copy()
            for column in selected:
                altered[column] = rng.permutation(altered[column].to_numpy())
            score = _apply_calibrator(artifact, artifact["model"].predict_proba(altered)[:, 1])
            metric = classification_metrics(subset["target_green"], score, subset["signal_date"], top_k=5)
            rows.append({"fold": item["fold"], "group": group, "method": "permutation_ablation", "sample_rows": len(subset), "baseline_roc_auc": base_metric.roc_auc, "ablated_roc_auc": metric.roc_auc, "roc_auc_delta": float(metric.roc_auc - base_metric.roc_auc), "baseline_log_loss": base_metric.log_loss, "ablated_log_loss": metric.log_loss, "log_loss_delta": metric.log_loss - base_metric.log_loss})
    return pd.DataFrame(rows)


def _importance_drift(manifest: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    groups = {spec.name: spec.group for spec in default_feature_registry()}
    for item in manifest:
        if item["model"] != "lightgbm_classifier":
            continue
        artifact = joblib.load(item["artifact_path"])
        importance = artifact["model"].feature_importances_
        for rank, (name, value) in enumerate(sorted(zip(artifact["feature_order"], importance, strict=True), key=lambda pair: pair[1], reverse=True), start=1):
            rows.append({"fold": item["fold"], "feature": name, "group": groups.get(name, "unknown"), "importance": float(value), "rank": rank})
    return pd.DataFrame(rows)


def _backtest_sensitivity(manifest: list[dict], config: RobustnessConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    lightgbm_paths = [item["prediction_path"] for item in manifest if item["model"] == "lightgbm_classifier"]
    if not lightgbm_paths:
        return pd.DataFrame(), pd.DataFrame()
    glob_path = str(Path(lightgbm_paths[0]).parent / "lightgbm_classifier_*.parquet")
    candidates = load_backtest_candidates(glob_path, config.normalized_path)
    fees = load_fee_schedule(config.fee_schedule_path)
    base = BacktestConfig(predictions_glob=glob_path, normalized_path=config.normalized_path, fee_schedule_path=config.fee_schedule_path, min_signal_day_turnover=5_000_000)
    topk_rows: list[dict] = []
    for threshold in (0.45, 0.50, 0.55, 0.60):
        for top_k in (1, 3, 5, 10):
            scenario = base.model_copy(update={"max_positions": top_k, "max_allocation_per_stock": min(1 / top_k, 0.20), "min_probability": threshold})
            trades, daily = simulate(candidates, fees, scenario, strategy="top5_probability", slippage_bps=5.0)
            topk_rows.append({"minimum_probability": threshold, "top_k": top_k, "slippage_bps_per_side": 5.0, **performance_summary(trades, daily, scenario.initial_capital)})
    fee_rows: list[dict] = []
    for multiplier in (0.75, 1.0, 1.25):
        adjusted = [period.model_copy(update={"brokerage_rate_per_side": period.brokerage_rate_per_side * multiplier, "exchange_transaction_rate_per_side": period.exchange_transaction_rate_per_side * multiplier, "sebi_rate_per_side": period.sebi_rate_per_side * multiplier, "stt_sell_rate": period.stt_sell_rate * multiplier, "stamp_buy_rate": period.stamp_buy_rate * multiplier}) for period in fees]
        trades, daily = simulate(candidates, adjusted, base, strategy="top5_probability", slippage_bps=5.0)
        fee_rows.append({"cost_multiplier": multiplier, "slippage_bps_per_side": 5.0, **performance_summary(trades, daily, base.initial_capital)})
    return pd.DataFrame(topk_rows), pd.DataFrame(fee_rows)


def run_robustness(config: RobustnessConfig) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((config.phase4_dir / "model_manifest.json").read_text(encoding="utf-8"))
    prediction_frames = []
    for item in manifest:
        frame = pd.read_parquet(item["prediction_path"])
        frame["fold"] = item["fold"]
        prediction_frames.append(frame)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    features = pd.read_parquet(config.features_path)
    reliability = _reliability(predictions)
    ranking = _ranking_comparison(predictions.loc[predictions["model"] == "lightgbm_classifier"], config)
    ablation = _permutation_ablation(features, manifest, config)
    importance = _importance_drift(manifest)
    topk, fee = _backtest_sensitivity(manifest, config)
    outputs = {"reliability": config.output_dir / "calibration_reliability.csv", "ranking": config.output_dir / "classification_vs_regression.csv", "ablation": config.output_dir / "feature_permutation_ablation.csv", "importance": config.output_dir / "feature_importance_drift.csv", "topk": config.output_dir / "threshold_topk_sensitivity.csv", "fee": config.output_dir / "fee_sensitivity.csv", "status": config.output_dir / "robustness_status.csv"}
    reliability.to_csv(outputs["reliability"], index=False)
    ranking.to_csv(outputs["ranking"], index=False)
    ablation.to_csv(outputs["ablation"], index=False)
    importance.to_csv(outputs["importance"], index=False)
    topk.to_csv(outputs["topk"], index=False)
    fee.to_csv(outputs["fee"], index=False)
    status = pd.DataFrame([
        {"experiment": "feature_ablation", "status": "completed", "detail": "LightGBM out-of-sample permutation ablation by registered feature group."},
        {"experiment": "market_index_ablation", "status": "skipped", "detail": "Verified Nifty index history was not supplied, so market features were never emitted."},
        {"experiment": "sector_ablation", "status": "skipped", "detail": "Verified historical sector mapping was not supplied, so sector features were never emitted."},
        {"experiment": "classification_vs_regression", "status": "completed", "detail": "Matched daily Top-5 realized-return ranking comparison with block-bootstrap intervals."},
        {"experiment": "threshold_topk_sensitivity", "status": "completed", "detail": "Probability thresholds 0.45/0.50/0.55/0.60 and K=1/3/5/10 at 5 bps per side."},
        {"experiment": "fee_slippage_sensitivity", "status": "completed", "detail": "Fee multipliers at 5 bps plus Phase 5's 0/5/10 bps scenarios."},
        {"experiment": "retraining_frequency", "status": "skipped", "detail": "Only annual Phase 4 artifacts exist; quarterly models must be trained before comparison."},
        {"experiment": "importance_drift", "status": "completed", "detail": "LightGBM feature importance ranking by fold."},
        {"experiment": "calibration_analysis", "status": "completed", "detail": "Reliability buckets from persisted OOS predictions."},
    ])
    status.to_csv(outputs["status"], index=False)
    ablation_summary = ablation.groupby("group", as_index=False)["roc_auc_delta"].mean().sort_values("roc_auc_delta") if not ablation.empty else pd.DataFrame()
    calibration_gap = float((reliability["mean_predicted_probability"] - reliability["observed_green_rate"]).abs().max()) if not reliability.empty else np.nan
    best_sensitivity = topk.loc[topk["cagr"].idxmax()] if not topk.empty else None
    report_lines = [
        "# Phase 6 robustness report",
        "",
        "## Interpretation boundary",
        "",
        "This report is descriptive and does not establish profitability or predictive superiority. Results use a non-point-in-time source universe and estimated historical fee inputs; they must not be used as a deployment recommendation.",
        "",
        "## Completed experiments",
        "",
        "- OOS LightGBM permutation feature sensitivity by registry group.",
        "- OOS classification-versus-regression daily Top-5 realized-return comparison with chronological moving-block bootstrap intervals.",
        "- Probability threshold / Top-K / 5 bps-per-side sensitivity.",
        "- Fee-multiplier sensitivity and Phase 5 slippage scenarios.",
        "- Reliability buckets and feature-importance drift.",
        "",
        "## Key descriptive summaries",
        "",
        f"- Largest reliability-bin absolute gap: {calibration_gap:.4f}. Inspect `calibration_reliability.csv`; bucket counts matter.",
    ]
    if best_sensitivity is not None:
        report_lines.append(f"- Highest CAGR among the *sensitivity settings* was {best_sensitivity['cagr']:.2%} (threshold={best_sensitivity['minimum_probability']:.2f}, K={int(best_sensitivity['top_k'])}, 5 bps per side). This is a selection result, not a recommended setting.")
    if not ablation_summary.empty:
        report_lines.extend(["", "Average OOS permutation ROC-AUC deltas (negative means performance declined after permutation):"])
        report_lines.extend([f"- {row.group}: {row.roc_auc_delta:+.4f}" for row in ablation_summary.itertuples(index=False)])
    report_lines.extend(["", "## Deliberately skipped", "", "- Market-index and sector ablations: verified index/sector history was not supplied, so these features were never emitted.", "- Retraining-frequency comparison: only annual Phase 4 model artifacts exist; quarterly models must be trained before comparison.", "", "## Artifact guide", "", "- `classification_vs_regression.csv`: daily realized-return ranking comparison and block-bootstrap intervals.", "- `feature_permutation_ablation.csv`: inference-time permutation sensitivity, not a retrained feature-removal study.", "- `threshold_topk_sensitivity.csv` and `fee_sensitivity.csv`: friction and selection sensitivity.", "- `robustness_status.csv`: experiment coverage and explicit skips."])
    report = config.output_dir / "research_report.md"
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    outputs["report"] = report
    return outputs


def run_robustness_from_config(path: str | Path) -> dict[str, Path]:
    return run_robustness(load_robustness_config(path))
