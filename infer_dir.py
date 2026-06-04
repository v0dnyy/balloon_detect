"""
Пакетный инференс на директории с изображениями/видео.
Сохраняет аннотированные файлы и JSON-лог по каждому файлу отдельно.

Пример запуска:
    python infer_dir.py --model best.engine --input ./images
    python infer_dir.py --model best.engine --input ./images --exts jpg png
"""
import argparse
import json
import logging
from pathlib import Path
import cv2

from config import InferenceConfig
from detector import BalloonDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Поддерживаемые расширения файлов
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference on directory of files")
    parser.add_argument("--model", type=str, required=True, help="Path to model (.pt or .engine)")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml (required for .engine)")
    parser.add_argument("--input", type=str, required=True, help="Path to input directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="IoU threshold")
    parser.add_argument("--half", action="store_true", help="Use FP16")
    parser.add_argument("--exts", nargs="+", default=None, help="Filter by extensions, e.g. jpg png")
    return parser.parse_args()


def collect_files(directory: Path, exts_filter: list[str] | None) -> list[Path]:
    """Рекурсивно собирает файлы поддерживаемых форматов."""
    all_exts = IMAGE_EXTS | VIDEO_EXTS
    if exts_filter:
        all_exts = {f".{e.lower().lstrip('.')}" for e in exts_filter}

    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in all_exts)
    return files


def process_image(detector: BalloonDetector, file_path: Path, cfg: InferenceConfig) -> list[dict]:
    """Инференс одного изображения. Сохраняет аннотированный файл и лог."""

    frame = cv2.imread(str(file_path))
    if frame is None:
        logger.warning(f"Cannot read image: {file_path}")
        return []

    results = detector.predict(frame)
    detections = detector.extract_detections(results, frame.shape)

    # Сохранить аннотированное изображение
    vis = detector.draw(frame, results, detections=detections)
    out_img = cfg.output_dir / file_path.name
    cv2.imwrite(str(out_img), vis)

    # Сохранить JSON-лог для этого файла
    log_file = cfg.logs_dir / (file_path.stem + "_detections.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)

    return detections


def process_video(detector: BalloonDetector, file_path: Path, cfg: InferenceConfig) -> list[dict]:
    """Инференс видеофайла. Сохраняет аннотированное видео и JSONL-лог."""

    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {file_path}")
        return []

    fps_src = int(cap.get(cv2.CAP_PROP_FPS)) or cfg.camera_fps_fallback
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = cfg.output_dir / (file_path.stem + "_annotated.mp4")
    log_path = cfg.logs_dir / (file_path.stem + "_detections.jsonl")

    writer = cv2.VideoWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_src,
        (frame_width, frame_height),
    )

    all_detections = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = detector.predict(frame)
            detections = detector.extract_detections(results, frame.shape)
            all_detections.extend(detections)

            vis = detector.draw(frame, results, detections=detections)
            writer.write(vis)

            if detections:
                detector.append_log(log_path, detections)
    finally:
        cap.release()
        writer.release()

    return all_detections


def main():
    args = parse_args()
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input is not a directory: {input_dir}")

    cfg = InferenceConfig(
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        half=args.half,
        data_yaml=args.data
    )
    detector = BalloonDetector(cfg)

    files = collect_files(input_dir, args.exts)
    if not files:
        logger.warning(f"No supported files found in {input_dir}")
        return

    logger.info(f"Found {len(files)} files in {input_dir}")

    total_detections = 0
    for i, file_path in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] Processing: {file_path.name}")

        if file_path.suffix.lower() in IMAGE_EXTS:
            dets = process_image(detector, file_path, cfg)
        else:
            dets = process_video(detector, file_path, cfg)

        total_detections += len(dets)
        logger.info(f"  → {len(dets)} detections")

    logger.info(
        f"Done. {len(files)} files processed, "
        f"{total_detections} total detections."
    )
    logger.info(f"Results saved to: {cfg.output_dir}")
    logger.info(f"Logs saved to:    {cfg.logs_dir}")


if __name__ == "__main__":
    main()
