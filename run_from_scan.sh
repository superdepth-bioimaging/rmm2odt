#!/usr/bin/env bash
# Resume the rmm2odt pipeline from the depth-scan through ODT, reusing an
# already-completed multilayer_base/. Runs: multilayer(scan only) -> gvas -> odt.
#
# The base reconstruction is SKIPPED by pointing multilayer.init_dir at the
# completed base dir (via $RMM2ODT_INIT_DIR, which reproduce_sample5.yaml reads);
# only the depth scan runs, warm-started from that base.
#
# Prerequisite: a completed base recon (output_layer_*_epoch_*.txt) in the
# base dir — by default $RMM2ODT_OUT/multilayer_base.
#
# Usage:
#   export RMM2ODT_DATA=/path/to/data      # folder with rrmat.mat + illumination .mat
#   export RMM2ODT_OUT=/path/to/run        # depth-scan/gvas/odt written here
#   bash run_from_scan.sh                   # scan -> gvas -> odt   (stops at odt)
#   bash run_from_scan.sh --to render       # ... and render the figures too
#   bash run_from_scan.sh --dry-run         # validate paths + plan, run nothing
#
# Base dir defaults to $RMM2ODT_OUT/multilayer_base; override with RMM2ODT_INIT_DIR
# (e.g. if the base recon lives in a different run directory).
# Config defaults to configs/reproduce_sample5.yaml; override with RMM2ODT_CONFIG.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${RMM2ODT_CONFIG:-configs/reproduce_sample5.yaml}"

: "${RMM2ODT_DATA:?set RMM2ODT_DATA to the folder holding rrmat.mat + the illumination .mat}"
: "${RMM2ODT_OUT:?set RMM2ODT_OUT to the output run dir}"
export RMM2ODT_INIT_DIR="${RMM2ODT_INIT_DIR:-$RMM2ODT_OUT/multilayer_base}"

# The completed base recon must exist — its layer .txt files warm-start the scan.
if ! ls "$RMM2ODT_INIT_DIR"/output_layer_*_epoch_*.txt >/dev/null 2>&1; then
  echo "ERROR: no base-recon layers (output_layer_*_epoch_*.txt) in:" >&2
  echo "       $RMM2ODT_INIT_DIR" >&2
  echo "       Run the base recon first (e.g. run.py --to multilayer), or point" >&2
  echo "       RMM2ODT_INIT_DIR at the dir that has it." >&2
  exit 1
fi

cd "$HERE"
echo "[run_from_scan] config=$CONFIG"
echo "[run_from_scan] base (skipped, warm-start) = $RMM2ODT_INIT_DIR"
echo "[run_from_scan] out=$RMM2ODT_OUT  -> depth-scan, gvas, odt"
echo "[run_from_scan] NOTE: overwrites any existing multilayer_scan/ gvas/ odt/ in that dir."
# --from multilayer + init_dir set => base skipped, scan runs. Default stop at odt;
# a --to passed in "$@" overrides (argparse takes the last --to value).
python -u run.py --config "$CONFIG" --from multilayer --to odt "$@" 2>&1 | tee "$RMM2ODT_OUT/run_from_scan.log"
