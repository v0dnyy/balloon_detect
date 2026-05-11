#!/bin/bash
# Примеры:
#   ./run_inference.sh                         # боевой режим (без экрана)
#   ./run_inference.sh --show                  # с окном (для отладки)
#   ./run_inference.sh --no_mav --show         # без MAVLink, с окном
#   ./run_inference.sh --int8                  # INT8 модель (указать ENGINE ниже)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Конфигурация ─────────────────────────────────────────────────────────────
SCRIPT="infer_stream.py"
ENGINE="best.pt"        # путь к TensorRT движку
CAMERA_ID=0                 # индекс камеры (/dev/video0)
IMGSZ=640                   # размер изображения для инференса
CONF=0.55                   # порог уверенности
IOU=0.65                    # порог IoU
MAV_PORT="/dev/ttyTHS0"     # UART порт MAVLink на Jetson

for arg in "$@"; do
    case $arg in
        --camera_id=*) CAMERA_ID="${arg#*=}" ;;
        --conf=*)  CONF="${arg#*=}" ;;
        --iou=*)   IOU="${arg#*=}" ;;
        --imgsz=*) IMGSZ="${arg#*=}" ;;
        --mav_port=*) MAV_PORT="${arg#*=}" ;;
    esac
done

# ── Базовые аргументы (всегда передаются) ────────────────────────────────────
ARGS=(
    --model      "$ENGINE"
    --camera_id  "$CAMERA_ID"
    --imgsz      "$IMGSZ"
    --conf       "$CONF"
    --iou        "$IOU"
    --mav_port   "$MAV_PORT"
    --half                   # FP16 — обязательно на Jetson
    --save_video             # всегда пишем видео на борту
    --save_logs              # всегда пишем JSONL лог детекций
)

# ── Доп. аргументы из командной строки (--show, --no_mav и т.д.) ─────────────
EXTRA_ARGS=()
for arg in "$@"; do
    case $arg in
        --camera_id=*|--conf=*|--iou=*|--imgsz=*|--mav_port=*) ;;  # уже обработаны выше
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

# ── Проверки перед запуском ───────────────────────────────────────────────────
if [ ! -f "$ENGINE" ]; then
    echo "[ERROR] Файл модели не найден: $ENGINE"
    exit 1
fi

if [ ! -f "$SCRIPT" ]; then
    echo "[ERROR] Скрипт не найден: $SCRIPT"
    exit 1
fi

# ── Запуск ────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  UAV Balloon Detection — Inference"
echo "============================================================"
echo "  Модель   : $ENGINE"
echo "  Камера   : /dev/video$CAMERA_ID"
echo "  imgsz    : $IMGSZ  |  conf: $CONF  |  iou: $IOU"
echo "  MAVLink  : $MAV_PORT"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "  Доп. флаги: ${EXTRA_ARGS[*]}"
fi
echo "============================================================"

python "$SCRIPT" "${ARGS[@]}" "${EXTRA_ARGS[@]}"

echo ""
echo "=== Инференс завершён ==="