#!/usr/bin/env bash
# Export YOLO from data/ then train. Override: DATA_ROOT, YOLO_OUT, RUN_NAME, SKIP_EXPORT=1,
# MODEL, BATCH, IMGSZ, EPOCHS, WORKERS. Smoke: SKIP_EXPORT=1 EPOCHS=2 MODEL=yolov8n.pt BATCH=8 IMGSZ=640 ./scripts/train_deepsport_comprehensive.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_ROOT="${DATA_ROOT:-data}"
YOLO_OUT="${YOLO_OUT:-results/yolo_deepsport}"
RUN_NAME="${RUN_NAME:-deepsport_comprehensive_$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-yolov8l.pt}"
BATCH="${BATCH:-4}"
IMGSZ="${IMGSZ:-1024}"
EPOCHS="${EPOCHS:-120}"
WORKERS="${WORKERS:-4}"

echo "=========================================="
echo "Repo:      $ROOT"
echo "Data root: $DATA_ROOT"
echo "YOLO out:  $YOLO_OUT"
echo "Run name:  $RUN_NAME"
echo "Model:     $MODEL  epochs=$EPOCHS imgsz=$IMGSZ batch=$BATCH workers=$WORKERS"
echo "=========================================="

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "ERROR: data root not found: $DATA_ROOT" >&2
  exit 1
fi

if [[ "${SKIP_EXPORT:-0}" != "1" ]]; then
  echo ""
  echo "== [1/2] Export =="
  uv run cs1430-export-yolo --data-root "$DATA_ROOT" --output "$YOLO_OUT"
else
  echo "== [1/2] SKIP_EXPORT=1 → $YOLO_OUT =="
  if [[ ! -f "$YOLO_OUT/data.yaml" ]]; then
    echo "ERROR: $YOLO_OUT/data.yaml missing" >&2
    exit 1
  fi
fi

echo ""
echo "== [2/2] Train =="
uv run cs1430-train-yolo \
  --data "$YOLO_OUT/data.yaml" \
  --model "$MODEL" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --batch "$BATCH" \
  --device auto \
  --project results/train_runs \
  --name "$RUN_NAME" \
  --patience 50 \
  --workers "$WORKERS" \
  --cos-lr \
  --mixup 0.08 \
  --close-mosaic 20 \
  --cache off

echo ""
echo "Weights: runs/detect/results/train_runs/${RUN_NAME}/weights/best.pt"
