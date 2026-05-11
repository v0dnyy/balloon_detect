"""
Инференс на видеофайле (оффлайн).
Сохраняет аннотированное видео и/или JSONL-лог.

Пример запуска:
    python infer_video.py --model best.engine --input video.mp4 --save_video --save_logs
"""
import argparse
import datetime
import logging
import time
from pathlib import Path

import cv2
from tqdm import tqdm

from config import InferenceConfig
from detector import BalloonDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

FRAME_SKIP = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Offline video inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model (.pt or .engine)")
    parser.add_argument("--input", type=str, required=True, help="Path to input video file")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="IoU threshold")
    parser.add_argument("--half", action="store_true", help="Use FP16")
    parser.add_argument("--show", action="store_true", help="Show video window while processing")
    parser.add_argument("--save_video", action="store_true", help="Save annotated video")
    parser.add_argument("--save_logs", action="store_true", help="Save detections to JSONL")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cfg = InferenceConfig(
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        half=args.half,
    )
    detector = BalloonDetector(cfg)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps_src = int(cap.get(cv2.CAP_PROP_FPS)) or cfg.camera_fps_fallback
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        f"Video: {frame_width}x{frame_height} @ {fps_src} FPS, "
        f"{total_frames} frames | inference every {FRAME_SKIP} frames"
    )

    # ── Выходные файлы ────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%d%m%Y_%H-%M-%S")
    stem = input_path.stem

    writer = None
    if args.save_video:
        out_path = cfg.output_dir / f"{stem}_annotated_{ts}.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_src,
            (frame_width, frame_height),
        )
        logger.info(f"Output video: {out_path}")

    log_path = None
    if args.save_logs:
        log_path = cfg.logs_dir / f"{stem}_{ts}.jsonl"
        logger.info(f"Output log: {log_path}")

    # ── Состояние цикла ───────────────────────────────────────────────────────
    frame_idx = 0
    last_results = None  # результаты последнего реального инференса
    t_start = time.perf_counter()
    prev_time = t_start

    # ── Основной цикл ─────────────────────────────────────────────────────────
    pbar = tqdm(
        total=total_frames,
        unit="fr",
        desc="Inference",
        dynamic_ncols=True,
        colour="green",
    )
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            run_inference = (frame_idx % FRAME_SKIP == 0)

            # ── Frame skip: инференс каждые FRAME_SKIP кадров ─────────────────
            if run_inference:
                last_results = detector.predict(frame)

            # На пропущенных кадрах detections=[] — лог не срабатывает
            detections = detector.extract_detections(last_results) if run_inference else []

            # ── FPS ───────────────────────────────────────────────────────────
            fps_real = None
            if args.show:
                curr_time = time.perf_counter()
                fps_real = 1.0 / max(curr_time - prev_time, 1e-6)
                prev_time = curr_time

            # Рисуем last_results на каждом кадре — боксы не мигают
            vis = detector.draw(frame, last_results, fps=fps_real)

            if writer:
                writer.write(vis)

            if log_path and detections:
                detector.append_log(log_path, detections)

            if args.show:
                cv2.imshow("Video Inference", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit by user")
                    break

            # Обновляем прогресс-бар каждый кадр
            elapsed = time.perf_counter() - t_start
            avg_fps = frame_idx / max(elapsed, 1e-6)
            det_count = len(detections) if detections else 0
            pbar.set_postfix({
                "FPS": f"{avg_fps:.1f}",
                "det": det_count,
            }, refresh=False)
            pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    logger.info(
        f"Done. {frame_idx} frames in {elapsed:.1f}s "
        f"(avg {frame_idx / elapsed:.1f} FPS)"
    )


if __name__ == "__main__":
    main()
