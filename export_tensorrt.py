"""
Экспорт обученной модели в TensorRT (.engine) для Jetson Orin Nano.
Запускать ТОЛЬКО на самом Jetson — TensorRT движок привязан к железу.

Пример запуска:
    python export_tensorrt.py --model runs/train/best.pt
    python export_tensorrt.py --model best.pt --imgsz 416 --int8  # быстрее, чуть хуже mAP
"""
import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO model to TensorRT")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--half", action="store_true", default=True, help="FP16 (default: True)")
    parser.add_argument("--int8", action="store_true", help="INT8 quantization (fastest, needs calibration data)")
    parser.add_argument("--data", type=str, default=None, help="data.yaml for INT8 calibration")
    parser.add_argument("--simplify", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    export_kwargs = dict(
        format="engine",
        imgsz=args.imgsz,
        half=args.half and not args.int8,  # half и int8 взаимно исключают
        int8=args.int8,
        simplify=args.simplify,
        device=0,  # GPU
    )

    if args.int8 and args.data:
        export_kwargs["data"] = args.data
        logger.info("INT8 calibration will use validation data from data.yaml")
    elif args.int8 and not args.data:
        logger.warning(
            "INT8 without --data: калибровка пройдёт на случайных данных. "
            "Качество детекции может снизиться. Укажи --data path/to/data.yaml."
        )

    precision = "INT8" if args.int8 else ("FP16" if args.half else "FP32")
    logger.info(f"Exporting to TensorRT ({precision}, imgsz={args.imgsz})...")

    engine_path = model.export(**export_kwargs)
    logger.info(f"✅ Saved: {engine_path}")
    logger.info("")
    logger.info("Использование:")
    logger.info(f"  python infer_stream.py --model {engine_path} --half")
    logger.info(f"  python infer_video.py  --model {engine_path} --input video.mp4")


if __name__ == "__main__":
    main()
