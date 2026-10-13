#!/usr/bin/env bash
# Full pipeline for the paper's definition. Safe to re-run: every stage skips completed work.
# Set CHD_ERA5_DIR (input) and CHD_DATA_ROOT (outputs) first; see README.md.
set -euo pipefail
cd "$(dirname "$0")"
LOG="${CHD_DATA_ROOT:-data}/logs"
mkdir -p "$LOG"
run() { echo "=== [$(date)] $1 ==="; shift; python3 "$@" 2>&1 | tee "$LOG/$(basename "$1" .py).log"; }
run "L1 daily fields"            scripts/01_build_daily.py --workers 6
run "L2 PET + water balance"     scripts/02_pet_waterbalance.py
run "L3 SPEI"                    scripts/03_spei.py
run "L4 T90 + heatwaves"         scripts/04_thresholds_heatwave.py
run "L5 compound flags"          scripts/05_compound.py
run "L6 trends"                  scripts/06_trends.py
run "Tables + aux layers"        scripts/11_manuscript_figures.py
run "Figure cache"               scripts/12_nature_figures.py --cache
echo "=== [$(date)] PIPELINE COMPLETE ==="
