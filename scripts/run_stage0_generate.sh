#!/usr/bin/env bash
set -euo pipefail

# Stage 0-1) Generate pseudo staging trajectories
# Usage:
#   bash scripts/run_stage0_generate.sh CASE_DIR [JAW] [NUM_STEPS] [NUM_TRAJ]
#
# Example:
#   bash scripts/run_stage0_generate.sh /path/to/0001 L 20 8

CASE_DIR="${1:-}"
JAW="${2:-L}"
NUM_STEPS="${3:-20}"
NUM_TRAJ="${4:-8}"

if [[ -z "${CASE_DIR}" ]]; then
  echo "ERROR: CASE_DIR is required."
  echo "Usage: bash scripts/run_stage0_generate.sh CASE_DIR [JAW] [NUM_STEPS] [NUM_TRAJ]"
  exit 1
fi

echo "[RUN] Stage0 Generate"
echo "  CASE_DIR=${CASE_DIR}"
echo "  JAW=${JAW}"
echo "  NUM_STEPS=${NUM_STEPS}"
echo "  NUM_TRAJ=${NUM_TRAJ}"

python -m preprocess.generate_pseudo_staging \
  --case_dir "${CASE_DIR}" \
  --jaw "${JAW}" \
  --num_steps "${NUM_STEPS}" \
  --num_traj "${NUM_TRAJ}"
