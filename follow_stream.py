"""
follow_stream.py — основной цикл режима следования за шариком.

Запуск:
  python follow_stream.py --model best.engine --half
  python follow_stream.py --model best.pt --no_mav --show   # отладка без дрона
"""

import argparse
import datetime
import logging
import time
from typing import Optional

import cv2
import numpy as np

from config import InferenceConfig
from detector import BalloonDetector
from mavlink_communication import MAVLinkCommunication
from pid_config import PIDConfig
from pid_controller import ControlOutput, PIDController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

FRAME_SKIP = 1


def parse_args():
    p = argparse.ArgumentParser(description="Balloon follow mode with PID")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--data", type=str, default="data.yaml")
    p.add_argument("--camera_id", type=int, default=0)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.50)
    p.add_argument("--iou", type=float, default=0.65)
    p.add_argument("--half", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--save_video", action="store_true")
    p.add_argument("--save_logs", action="store_true")
    p.add_argument("--mav_port", type=str, default="/dev/ttyTHS0")
    p.add_argument("--no_mav", action="store_true", help="Отладка без дрона")
    p.add_argument(
        "--target_area",
        type=float,
        default=0.005,
        help="Целевая площадь bbox/frame (default 0.005)",
    )
    p.add_argument(
        "--ema_alpha", type=float, default=0.3, help="Коэффициент EMA (default 0.35)"
    )
    p.add_argument("--max_lost", type=int, default=15)
    return p.parse_args()


def open_camera(camera_id: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")
    return cap


def main():
    args = parse_args()

    # ── Конфиги ───────────────────────────────────────────────────────────
    infer_cfg = InferenceConfig(
        model_path=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        half=args.half,
        mav_port=args.mav_port,
        data_yaml=args.data,
        ema_alpha=args.ema_alpha,
    )
    pid_cfg = PIDConfig(target_area_ratio=args.target_area)

    # ── Компоненты ────────────────────────────────────────────────────────
    detector = BalloonDetector(infer_cfg)
    pid = PIDController(pid_cfg)

    # ── MAVLink ───────────────────────────────────────────────────────────
    mav = None
    if not args.no_mav:
        try:
            mav = MAVLinkCommunication(port=infer_cfg.mav_port)
            # mav._switch_to_mode(pid_cfg.guided_mode)
            logger.info(f"MAVLink ready → mode {pid_cfg.guided_mode}")
        except Exception as e:
            logger.warning(f"MAVLink init failed: {e}. Continuing without MAVLink.")

    # ── Камера ────────────────────────────────────────────────────────────
    cap = open_camera(args.camera_id)

    fps_src = int(cap.get(cv2.CAP_PROP_FPS)) or infer_cfg.camera_fps_fallback
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(
        f"Camera: {frame_w}x{frame_h} @ {fps_src} FPS | inference every {FRAME_SKIP} frames"
    )

    # ── Видеозапись ───────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%d%m%Y_%H-%M-%S")

    writer = None
    if args.save_video:
        out_path = infer_cfg.output_dir / f"follow_{ts}.mp4"
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_src,
            (frame_w, frame_h),
        )
        logger.info(f"Saving video to {out_path}")

    # ── Лог-файл ──────────────────────────────────────────────────────────
    log_path = None
    if args.save_logs:
        log_path = infer_cfg.logs_dir / f"follow_{ts}.jsonl"
        logger.info(f"Saving logs → {log_path}")

    # ── Состояние цикла ───────────────────────────────────────────────────
    frame_idx = 0
    lost_streak = 0
    last_good_frame: Optional[np.ndarray] = None
    last_track: Optional[dict] = None
    last_output = ControlOutput(
        track_id=None,
        class_name=None,
        cx_norm=0.5,
        cy_norm=0.5,
        area_ratio=0.0,
        error_x=0.0,
        error_y=0.0,
        error_z=0.0,
        vx=0.0,
        vy=0.0,
        vz=0.0,
        lost=True,
    )
    prev_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()

            # ── Потеря кадра ──────────────────────────────────────────────
            if not ret:
                lost_streak += 1
                logger.warning(f"Frame lost ({lost_streak}/{args.max_lost})")

                if lost_streak >= args.max_lost:
                    logger.error("Too many lost frames — reconnecting...")
                    cap.release()
                    try:
                        cap = open_camera(args.camera_id)
                        logger.info("Camera reconnected ✓")
                    except RuntimeError as e:
                        logger.error(f"Reconnect failed: {e}. Retrying in 1s...")
                        time.sleep(1)
                    lost_streak = 0

                if last_good_frame is not None and writer:
                    writer.write(last_good_frame)
                continue

            lost_streak = 0
            last_good_frame = frame.copy()
            frame_idx += 1

            if frame_idx % FRAME_SKIP == 0:
                # ── ШАГ 1: детекция + трекинг + EMA ──────────────────────
                last_track = detector.track(frame)

                # ── ШАГ 2+3: ошибки → PID → скорости ─────────────────────
                last_output = pid.update(last_track, frame.shape)

            # ── ШАГ 4: MAVLink → команда дрону ───────────────────────────
            if last_output.lost:
                logger.info("[HOVERING] Объектов не найдено")
                if mav:
                    mav.send_hover()
            else:
                logger.info(
                        f"[TRACKING] ID:{last_output.track_id} class: {last_output.class_name}| "
                        f"err=({last_output.error_x:.3f}, {last_output.error_y:.3f}, {last_output.error_z:.3f}) | "
                        f"roll_angle=({last_output.vx:.3f}, thrust={last_output.vy:.3f}, pitch_angle={last_output.vz:.3f}) | "
                        f"area={last_output.area_ratio:.5f}"
                    )
                if mav:
                    # Маппинг PID-осей → MAV_FRAME_BODY_NED:
                    #   вперёд/назад ← PID_z (управляет дальностью)
                    #   право/лево   ← PID_x (управляет горизонталью)
                    #   вниз/вверх   ← PID_y (управляет вертикалью, NED!)
                    mav.send_command(roll_angle=last_output.vx,
                                     pitch_angle=last_output.vz,
                                     
                                     thrust=last_output.vy
                                     )
                    # mav.send_velocity_body(
                    #     vx_body=last_output.vz,
                    #     vy_body=last_output.vx,
                    #     vz_body=last_output.vy,
                    # )                


            # ── FPS ───────────────────────────────────────────────────────
            curr_time = time.perf_counter()
            fps_real = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            # ── Визуализация ──────────────────────────────────────────────
            vis = detector.draw_track(
                frame,
                last_track,
                last_output,
                target_cx_norm=pid_cfg.target_cx_norm,
                target_cy_norm=pid_cfg.target_cy_norm,
                fps=fps_real,
            )

            if writer:
                writer.write(vis)

            # ── Логирование ───────────────────────────────────────────────
            if log_path and last_track is not None:
                detector.log_track(log_path, last_track, last_output)

            if args.show:
                cv2.imshow("Follow Stream", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("Quit by user")
                    break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if mav:
            mav.send_hover()
            mav.close()
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        logger.info("Resources released ✓")


if __name__ == "__main__":
    main()
