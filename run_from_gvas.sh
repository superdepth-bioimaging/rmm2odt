#!/usr/bin/env bash
# Resume the rmm2odt pipeline from the GVAS stage through render, reusing an
# already-completed multilayer_base/ + multilayer_scan/ under $RMM2ODT_OUT.
# Runs: gvas -> odt -> render  (i.e. `run.py --from gvas`).
#
# Prerequisite: multilayer_scan/output_layer_stack.npy already exists in the run.
#
# Usage:
#   export RMM2ODT_DATA=/path/to/data      # folder with rrmat.mat + illumination .mat
#   export RMM2ODT_OUT=/path/to/run        # existing run dir containing multilayer_scan/
#   bash run_from_gvas.sh                   # gvas -> odt -> render
#   bash run_from_gvas.sh --to odt          # stop after odt (skip render)
#   bash run_from_gvas.sh --dry-run         # validate paths + plan, run nothing
#
# Config defaults to configs/reproduce_sample5.yaml; override with RMM2ODT_CONFIG.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${RMM2ODT_CONFIG:-configs/reproduce_sample5.yaml}"

: "${RMM2ODT_DATA:?set RMM2ODT_DATA to the folder holding rrmat.mat + the illumination .mat}"
: "${RMM2ODT_OUT:?set RMM2ODT_OUT to the run dir that already contains multilayer_scan/}"

STACK="$RMM2ODT_OUT/multilayer_scan/output_layer_stack.npy"
if [[ ! -f "$STACK" ]]; then
  echo "ERROR: $STACK not found." >&2
  echo "       multilayer_base + multilayer_scan must be completed first" >&2
  echo "       (run the full pipeline, or 'run.py --to multilayer')." >&2
  exit 1
fi

cd "$HERE"
echo "[run_from_gvas] config=$CONFIG"
echo "[run_from_gvas] out=$RMM2ODT_OUT  (reusing multilayer_scan/)"
echo "[run_from_gvas] NOTE: this overwrites any existing gvas/ odt/ render/ in that dir."
python -u run.py --config "$CONFIG" --from gvas "$@" 2>&1 | tee "$RMM2ODT_OUT/run_from_gvas.log"
