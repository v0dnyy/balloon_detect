"""
Инференс в режиме реального времени с камеры БПЛА.
Оптимизирован для Jetson Orin Nano: TensorRT + FP16.

Пример запуска:
    python infer_stream.py --model best.engine --half
    python infer_stream.py --model best.engine --camera_id 0 --show --save_video --save_logs
"""
import argparse
import datetime
import logging
import time

import cv2
import numpy as np

from config import InferenceConfig
from detector import BalloonDetector
from mavlink_communication import MAVLinkCommunication

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

FRAME_SKIP = 1


def parse_args():
    parser = argparse.ArgumentParser(description="UAV real-time inference from camera")
    parser.add_argument("--model", type=str, required=True, help="Path to model (.pt or .engine)")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml (required for .engine)")
    parser.add_argument("--camera_id", type=int, default=0, help="Camera device index")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.50, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="IoU threshold")
    parser.add_argument("--half", action="store_true", help="Use FP16 (recommended on Jetson)")
    parser.add_argument("--show", action="store_true", help="Show video window")
    parser.add_argument("--save_video", action="store_true", help="Save annotated video")
    parser.add_argument("--save_logs", action="store_true", help="Save detections to JSONL")
    parser.add_argument("--mav_port", type=str, default="/dev/ttyTHS0", help="MAVLink serial port")
    parser.add_argument("--no_mav", action="store_true", help="Disable MAVLink (debug mode)")
    parser.add_argument("--max_lost", type=int, default=15, help="Max consecutive lost frames before reconnect")
    return parser.parse_args()


def open_camera(camera_id: int) -> cv2.VideoCapture:
    """Открывает камеру и возвращает VideoCapture. Бросает RuntimeError если не удалось."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    return cap


def main():
    args = parse_args()

    cfg = InferenceConfig(
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        half=args.half,
        mav_port=args.mav_port,
        data_yaml=args.data
    )

    detector = BalloonDetector(cfg)

    # ── MAVLink ───────────────────────────────────────────────────────────────
    mav = None
    if not args.no_mav:
        try:
            mav = MAVLinkCommunication(port=cfg.mav_port)
            logger.info(f"MAVLink connected on {cfg.mav_port}")
        except Exception as e:
            logger.warning(f"MAVLink init failed: {e}. Continuing without MAVLink.")

    # ── Камера ────────────────────────────────────────────────────────────────
    cap = open_camera(args.camera_id)

    fps_src = int(cap.get(cv2.CAP_PROP_FPS)) or cfg.camera_fps_fallback
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Camera: {frame_width}x{frame_height} @ {fps_src} FPS | inference every {FRAME_SKIP} frames")

    # ── Выходной видеофайл ────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%d%m%Y_%H-%M-%S")

    writer = None
    if args.save_video:
        out_path = cfg.output_dir / f"stream_{ts}.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_src,
            (frame_width, frame_height),
        )
        logger.info(f"Saving video to {out_path}")

    # ── Лог-файл ──────────────────────────────────────────────────────────────
    log_path = None
    if args.save_logs:
        log_path = cfg.logs_dir / f"stream_{ts}.jsonl"
        logger.info(f"Saving logs to {log_path}")

    # ── Состояние цикла ───────────────────────────────────────────────────────
    frame_idx = 0
    last_good_frame: np.ndarray | None = None  # для freeze при потере кадра
    last_results = None  # последние результаты YOLO
    lost_streak = 0  # счётчик потерянных кадров подряд
    prev_time = time.perf_counter()
    last_detections: list[dict] = []     # последние детекции с зонами
    last_alert_time: dict[str, float] = {}  # кулдаун по зонам

    # ── Основной цикл ─────────────────────────────────────────────────────────
    try:
        while True:
            ret, frame = cap.read()

            # ── Обработка потери кадра ────────────────────────────────────────
            if not ret:
                lost_streak += 1
                logger.warning(f"Frame lost ({lost_streak}/{args.max_lost})")

                if lost_streak >= args.max_lost:
                    logger.error("Too many lost frames — reconnecting camera...")
                    cap.release()
                    try:
                        cap = open_camera(args.camera_id)
                        logger.info("Camera reconnected ✓")
                    except RuntimeError as e:
                        logger.error(f"Reconnect failed: {e}. Retrying in 1s...")
                        time.sleep(1)
                    lost_streak = 0

                # Freeze frame: пишем последний успешный кадр чтобы видео не ускорялось
                if last_good_frame is not None and writer:
                    writer.write(last_good_frame)
                continue

            # Кадр получен успешно
            lost_streak = 0
            last_good_frame = frame.copy()
            frame_idx += 1

            # ── Frame skip: инференс каждые FRAME_SKIP кадров ─────────────────
            run_inference = (frame_idx % FRAME_SKIP == 0)

            if run_inference:
                last_results = detector.predict(frame)
                last_detections = detector.extract_detections(last_results, frame.shape)


            # На пропущенных кадрах detections=[] — MAVLink и лог не срабатывают
            # detections = detector.extract_detections(last_results) if run_inference else []

            # ── MAVLink — только при наличии детекций ─────────────────────────
            if last_detections and mav:
                for det in last_detections:
                    zone = det["distance_zone"]
                    area_ratio = det["area_ratio"]
                    now = time.monotonic()
                    # Кулдаун: не отправляем одну зону чаще раза в N секунд
                    if now - last_alert_time.get(zone, 0) >= cfg.mav_alert_cooldown_s:
                        last_alert_time[zone] = now
                        try:
                            mav.send_detection_alert(
                                detection_count=len(last_detections),
                                class_names=[d["class"] for d in last_detections],
                                zone=zone,
                                area_ratio=area_ratio,
                            )
                            # Команда уклонения — срабатывает только при CLOSE
                            mav.send_avoidance_command(zone)
                        except Exception as e:
                            logger.error(f"MAVLink send error: {e}")
            # ── FPS (только если нужен оверлей) ──────────────────────────────
            fps_real = None
            if args.show:
                curr_time = time.perf_counter()
                fps_real = 1.0 / max(curr_time - prev_time, 1e-6)
                prev_time = curr_time

            # ── Визуализация ──────────────────────────────────────────────────
            # last_results содержит боксы с последнего инференса —
            # на пропущенных кадрах рисуем их поверх нового кадра
            vis = detector.draw(frame, last_results, detections=last_detections, fps=fps_real)

            if writer:
                writer.write(vis)

            if log_path and last_detections:
                detector.append_log(log_path, last_detections)

            if args.show:
                cv2.imshow("UAV Stream", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit by user")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cap.release()
        if writer:
            writer.release()
        if mav:
            mav.close()
        cv2.destroyAllWindows()
        logger.info("Resources released ✓")


if __name__ == "__main__":
    main()
