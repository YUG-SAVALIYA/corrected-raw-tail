"""Train fresh 7 walk-forward HGB models for 2020-2026."""

import time
import sys
from pathlib import Path
from nifty_research.empirical import run_empirical_from_config

print("Starting fresh walk-forward training for 7 models (2020 - 2026)...", flush=True)
start_time = time.time()
config_path = Path("configs/retrained_hgb_direct_short_market_index_raw_tail_walkforward_2020_2026.yaml")

result = run_empirical_from_config(config_path)

elapsed = time.time() - start_time
print(f"Training completed successfully in {elapsed:.1f} seconds!", flush=True)
print(f"Result summary: {result}", flush=True)

