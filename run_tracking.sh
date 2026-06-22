#!/bin/bash
# Примеры:
#   ./run_follow.sh                                      # боевой режим (без окна)
#   ./run_follow.sh --show                               # с окном
#   ./run_follow.sh --no_mav --show                      # отладка без дрона
#   ./run_follow.sh --model=balloon.engine --camera_id=1 # своя модель/камера
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ── Базовые настройки ───────────────────────────────────────
SCRIPT="follow_stream.py"
MODEL="best.engine"          # по умолчанию TensorRT-движок
DATA="data.yaml"
CAMERA_ID=0
IMGSZ=640
CONF=0.65
IOU=0.55
MAV_PORT="/dev/ttyACM0"      # см. default в follow_stream.py
TARGET_AREA=0.005
EMA_ALPHA=0.3
MAX_LOST=15

# ── Разбор простых параметров вида --key=value ──────────────
for arg in "$@"; do
    case $arg in
        --model=*)       MODEL="${arg#*=}" ;;
        --data=*)        DATA="${arg#*=}" ;;
        --camera_id=*)   CAMERA_ID="${arg#*=}" ;;
        --imgsz=*)       IMGSZ="${arg#*=}" ;;
        --conf=*)        CONF="${arg#*=}" ;;
        --iou=*)         IOU="${arg#*=}" ;;
        --mav_port=*)    MAV_PORT="${arg#*=}" ;;
        --target_area=*) TARGET_AREA="${arg#*=}" ;;
        --ema_alpha=*)   EMA_ALPHA="${arg#*=}" ;;
        --max_lost=*)    MAX_LOST="${arg#*=}" ;;
    esac
done

# ── Обязательные аргументы (всегда пойдут в Python) ─────────
ARGS=(
    --model       "$MODEL"
    --data        "$DATA"
    --camera_id   "$CAMERA_ID"
    --imgsz       "$IMGSZ"
    --conf        "$CONF"
    --iou         "$IOU"
    --mav_port    "$MAV_PORT"
    --target_area "$TARGET_AREA"
    --ema_alpha   "$EMA_ALPHA"
    --max_lost    "$MAX_LOST"
    --half                 # для Jetson/FP16
    --save_video           # писать видео
    --save_logs            # писать jsonl-логи
)

# ── Пробрасываем остальные флаги как есть (--show, --no_mav и т.п.) ──
EXTRA_ARGS=()
for arg in "$@"; do
    case $arg in
        --model=*|--data=*|--camera_id=*|--imgsz=*|--conf=*|--iou=*|\
        --mav_port=*|--target_area=*|--ema_alpha=*|--max_lost=*)
            ;;  # уже обработаны
        *)
            EXTRA_ARGS+=("$arg")
            ;;
    esac
done

# ── Проверки ────────────────────────────────────────────────
if [ ! -f "$SCRIPT" ]; then
    echo "[ERROR] Не найден $SCRIPT"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "[WARN] Файл модели '$MODEL' не найден. Если используешь .pt, просто передай --model=best.pt"
fi

# ── Инфо перед стартом ──────────────────────────────────────
echo "============================================================"
echo "  Balloon Follow Mode — PID tracking"
echo "============================================================"
echo "  Скрипт   : $SCRIPT"
echo "  Модель   : $MODEL"
echo "  Камера   : /dev/video$CAMERA_ID"
echo "  imgsz    : $IMGSZ  |  conf: $CONF  |  iou: $IOU"
echo "  MAVLink  : $MAV_PORT"
echo "  target_area : $TARGET_AREA  |  ema_alpha: $EMA_ALPHA  |  max_lost: $MAX_LOST"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "  Доп. флаги: ${EXTRA_ARGS[*]}"
fi
echo "============================================================"

python "$SCRIPT" "${ARGS[@]}" "${EXTRA_ARGS[@]}"

echo ""
echo "=== Инференс завершён ==="