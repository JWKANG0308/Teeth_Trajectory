#!/usr/bin/env bash
set -euo pipefail

# (Optional) Suggest margin from the ORI geometry only
# Usage:
#   bash scripts/run_stage0_suggest_margin.sh CASE_DIR [JAW] [OUT_CSV]
#
# Example:
#   bash scripts/run_stage0_suggest_margin.sh /path/to/0001 L /path/to/0001/margin_suggest_pairs.csv

CASE_DIR="${1:-}"
JAW="${2:-L}"
OUT_CSV="${3:-}"

if [[ -z "${CASE_DIR}" ]]; then
  echo "ERROR: CASE_DIR is required."
  echo "Usage: bash scripts/run_stage0_suggest_margin.sh CASE_DIR [JAW] [OUT_CSV]"
  exit 1
fi

ORI_JSON="${CASE_DIR}/ori/${JAW}_Ori.json"
if [[ ! -f "${ORI_JSON}" ]]; then
  echo "ERROR: Missing ori json: ${ORI_JSON}"
  exit 1
fi

echo "[RUN] Stage0 Suggest Margin"
echo "  ORI_JSON=${ORI_JSON}"
echo "  OUT_CSV=${OUT_CSV}"

ARGS=(--out_root "${CASE_DIR}/pseudo_staging_lightcol_${JAW}" --ori_json "${ORI_JSON}" --suggest_margin)

if [[ -n "${OUT_CSV}" ]]; then
  ARGS+=(--suggest_out_csv "${OUT_CSV}")
fi

python -m preprocess.evaluate_collision "${ARGS[@]}"
