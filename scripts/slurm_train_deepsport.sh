#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH -J deepsport_yolo
#SBATCH -o slurm-deepsport-%j.out
#SBATCH -e slurm-deepsport-%j.err
#
# Long overnight-style YOLO job for DeepSport ball/player detection.
#
# Usage (from repo root on Oscar):
#   export REPO=/oscar/data/class/csci1430/students/$USER/automated-play-annotations   # adjust
#   sbatch scripts/slurm_train_deepsport.sh
#
# Override via environment:
#   DATA_ROOT MODEL BATCH IMGSZ EPOCHS RUN_NAME SKIP_EXPORT COPY_IMAGES WORKERS MEM_REQ
#
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"

DATA_ROOT="${DATA_ROOT:-data}"
YOLO_OUT="${YOLO_OUT:-results/yolo_deepsport}"
RUN_NAME="${RUN_NAME:-deepsport_overnight_${SLURM_JOB_ID:-manual}}"
MODEL="${MODEL:-yolov8m.pt}"
BATCH="${BATCH:-8}"
IMGSZ="${IMGSZ:-1280}"
EPOCHS="${EPOCHS:-120}"
WORKERS="${WORKERS:-8}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
COPY_IMAGES="${COPY_IMAGES:-1}"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "REPO:   $REPO"
echo "CUDA:   ${CUDA_VISIBLE_DEVICES:-unset}"
echo "DATA:   $DATA_ROOT → YOLO $YOLO_OUT"
echo "$MODEL  epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH run=$RUN_NAME"
echo "=========================================="

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
if command -v uv >/dev/null 2>&1; then
  source ~/.local/bin/env 2>/dev/null || true
  uv sync
else
  echo "WARN: uv not on PATH — activate your env manually if imports fail." >&2
fi

EXP_FLAGS=()
if [[ "$COPY_IMAGES" == "1" ]]; then
  EXP_FLAGS+=(--copy-images)
fi

if [[ "$SKIP_EXPORT" != "1" ]]; then
  uv run cs1430-export-yolo --data-root "$DATA_ROOT" --output "$YOLO_OUT" "${EXP_FLAGS[@]}"
else
  if [[ ! -f "$YOLO_OUT/data.yaml" ]]; then
    echo "ERROR: SKIP_EXPORT=1 but $YOLO_OUT/data.yaml missing" >&2
    exit 1
  fi
fi

uv run cs1430-train-yolo \
  --data "$YOLO_OUT/data.yaml" \
  --model "$MODEL" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device auto \
  --project results/train_runs \
  --name "$RUN_NAME" \
  --patience 40 \
  --workers "$WORKERS" \
  --cos-lr \
  --preset deepsport \
  --mixup 0.08 \
  --close-mosaic 20 \
  --cache off

BEST="$REPO/results/train_runs/$RUN_NAME/weights/best.pt"
echo ""
echo "Training finished. Best weights (if run completed): $BEST"
if [[ -f "$BEST" ]]; then
  uv run cs1430-val-yolo --weights "$BEST" --data "$YOLO_OUT/data.yaml" --imgsz "$IMGSZ" --batch "$BATCH" --device auto --plots
fi
