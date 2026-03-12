#!/usr/bin/env bash
set -euo pipefail

# Stage 0-2) Evaluate collisions (+ optional heatmaps)
# Usage:
#   bash scripts/run_stage0_eval.sh CASE_DIR [JAW] [MARGIN] [PAIR_MODE] [NEIGHBOR_RADIUS] [MAKE_HEATMAPS]
#
# Example:
#   bash scripts/run_stage0_eval.sh /path/to/0001 L 0.0015 neighbors 4.0 yes

CASE_DIR="${1:-}"
JAW="${2:-L}"
MARGIN="${3:-0.0015}"
PAIR_MODE="${4:-neighbors}"         # neighbors | all
NEIGHBOR_RADIUS="${5:-4.0}"
MAKE_HEATMAPS="${6:-yes}"           # yes | no

if [[ -z "${CASE_DIR}" ]]; then
  echo "ERROR: CASE_DIR is required."
  echo "Usage: bash scripts/run_stage0_eval.sh CASE_DIR [JAW] [MARGIN] [PAIR_MODE] [NEIGHBOR_RADIUS] [MAKE_HEATMAPS]"
  exit 1
fi

OUT_ROOT="${CASE_DIR}/pseudo_staging_lightcol_${JAW}"
ORI_JSON="${CASE_DIR}/ori/${JAW}_Ori.json"

if [[ ! -f "${ORI_JSON}" ]]; then
  echo "ERROR: Missing ori json: ${ORI_JSON}"
  exit 1
fi

echo "[RUN] Stage0 Evaluate"
echo "  OUT_ROOT=${OUT_ROOT}"
echo "  ORI_JSON=${ORI_JSON}"
echo "  MARGIN=${MARGIN}"
echo "  PAIR_MODE=${PAIR_MODE}"
echo "  NEIGHBOR_RADIUS=${NEIGHBOR_RADIUS}"
echo "  MAKE_HEATMAPS=${MAKE_HEATMAPS}"

ARGS=(
  --out_root "${OUT_ROOT}"
  --ori_json "${ORI_JSON}"
  --margin "${MARGIN}"
  --pair_mode "${PAIR_MODE}"
  --neighbor_radius "${NEIGHBOR_RADIUS}"
)

if [[ "${MAKE_HEATMAPS}" == "yes" ]]; then
  ARGS+=(--save_heatmaps)
fi

python -m trajectory_synthesis.evaluate_collision "${ARGS[@]}"
