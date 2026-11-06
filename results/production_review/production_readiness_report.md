# Production-readiness review: short-model configurations

## Decision

The historical evidence supports one candidate for a controlled live pilot: the 20% allocation priority merge with 4% stop / 8% target. The other two shortlist entries are not independent portfolios to run at the same time: the priority merge already contains Baseline first and raw-tail as capacity filler.

## Top-three shortlist

| Rank | Portfolio | Allocation | Exit | Return | CAGR | Max DD | Sharpe | Trade PF | Trades | Role |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | Priority merge: Baseline then raw-tail | 20% | SL 4% / TP 8% (conservative) | 2432.52% | 60.99% | -5.23% | 4.08 | 2.34 | 790 | Primary candidate (only one to consider for live pilot) |
| 2 | Baseline q99.97 | 20% | SL 4% / TP 8% (conservative) | 1202.81% | 45.97% | -5.42% | 3.74 | 2.72 | 558 | Shadow benchmark / fallback; do not run alongside the merge |
| 3 | Corrected raw-tail q99.97 | 20% | SL 4% / TP 8% (conservative) | 263.28% | 20.93% | -5.04% | 2.48 | 2.36 | 339 | Shadow capacity monitor; already a component of the merge |

## Why the 20% 4%/8% shortlist

- It keeps maximum historical drawdown near 5%, unlike the 33% allocation configurations (8% to 16% drawdown) and the 10% stop configurations (roughly 6% to 16% drawdown).
- Baseline and the merge have positive returns in all seven tested calendar years. Raw-tail has no 2020 trades and is positive in each of its six active years. The merge provides the highest return and Sharpe among the low-drawdown 20% target configurations.
- The 4% / 8% target was evaluated conservatively: when daily OHLC touches both stop and target, the stop is assumed first. Intraday data is still required to validate this ordering.

## Reasons this is not production-approved yet

- The gap filter requires observing the opening price; the current backtest still assumes a fill at that official open after the rule passes. A post-open intraday execution model is required.
- Daily OHLC cannot validate target/stop sequencing, partial fills, borrow availability/cost, market impact, or queue position. The strategy is short-only and needs an executable SLB/shorting workflow.
- Historical data/model checks were strong for the corrected raw-tail pipeline, but the research universe may have survivorship bias and does not prove live performance.
- Before a pilot: use 5- or 15-minute data, add realistic post-open entries and borrow costs, run a paper-trading period, and enforce daily loss, exposure, kill-switch, audit-log, and broker-compliance controls.

## Configuration review

See `all_configuration_production_review.csv` for every configuration's return, drawdown, Sharpe, Calmar, year/month consistency, 5th-percentile daily result, losing streaks, and target/stop collision counts.
