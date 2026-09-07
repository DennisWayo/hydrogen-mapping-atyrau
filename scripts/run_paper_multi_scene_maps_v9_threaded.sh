#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
SCENE_WORKERS="${SCENE_WORKERS:-4}"
MAP_SIZE="${MAP_SIZE:-1600}"
FULL_SCALE="${FULL_SCALE:-0}"
PNG_DPI="${PNG_DPI:-320}"
PNG_COMPRESS_LEVEL="${PNG_COMPRESS_LEVEL:-6}"
PNG_TARGET_SIZE_MB="${PNG_TARGET_SIZE_MB:-7}"
PNG_TARGET_TOL="${PNG_TARGET_TOL:-0.15}"
QML_BATCH_SIZE="${QML_BATCH_SIZE:-256}"
ARGS=(
  scripts/paper_multi_scene_maps.py
  --config paper_runs/configs/south_kazakhstan_region_paper_run.json
  --search-csv paper_runs/copernicus/south_kazakhstan_region/search_20260319T070227Z.csv
  --scene-id S2B_MSIL2A_20251022T060819_N0511_R134_T42TXM_20251022T072708
  --scene-id S2B_MSIL2A_20251022T060819_N0511_R134_T42TXN_20251022T072708
  --scene-id S2C_MSIL2A_20251020T061911_N0511_R034_T42TVN_20251020T095804
  --scene-id S2C_MSIL2A_20251020T061911_N0511_R034_T42TWN_20251020T095804
  --models ml_anomaly,qml_qnn
  --reuse-existing
  --auto-fix-degenerate-labels
  --max-positive-fraction 0.35
  --qml-batch-size "${QML_BATCH_SIZE}"
  --error-map-mode signed_disagreement
  --map-size "${MAP_SIZE}"
  --scene-workers "${SCENE_WORKERS}"
  --png-dpi "${PNG_DPI}"
  --png-compress-level "${PNG_COMPRESS_LEVEL}"
  --png-target-size-mb "${PNG_TARGET_SIZE_MB}"
  --png-target-tolerance "${PNG_TARGET_TOL}"
  --out-root paper_runs/runs/south_kazakhstan_region/scenes
  --out-fig-dir figures/paper_results/scenes
)

if [[ "${FULL_SCALE}" == "1" ]]; then
  ARGS+=(--full-scale)
fi

"${PYTHON_BIN}" "${ARGS[@]}"
