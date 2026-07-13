"""
Единственная точка работы с YOLO-моделью.
Все остальные скрипты используют класс BalloonDetector — не YOLO напрямую.
"""

import datetime
import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from config import InferenceConfig
from smoothing import EMAFilter

logger = logging.getLogger(__name__)

ZONE_FAR = "FAR"
ZONE_MEDIUM = "MEDIUM"
ZONE_CLOSE = "CLOSE"
ZONE_UNKNOWN = "UNKNOWN"

TRACKER_CONFIG = "bytetrack.yaml"


def get_device(model_path: str) -> str:
    """
    Автоматически выбирает лучшее устройство с учётом формата модели.
    """
    suffix = Path(model_path).suffix.lower()
    if suffix == ".onnx":
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_names_from_yaml(data_yaml_path: str) -> dict[int, str]:
    """Загружает имена классов из data.yaml (list или dict формат)."""
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if names is None:
        raise ValueError(f"Field 'names' not found in {data_yaml_path}")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError(f"Unsupported 'names' format in {data_yaml_path}: {type(names)}")


class BalloonDetector:
    def __init__(self, cfg: InferenceConfig):
        torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()

        self.cfg = cfg
        self.device = get_device(str(cfg.model_path))

        logger.info(f"Loading model: {cfg.model_path} | device: {self.device}")
        self._ema = EMAFilter(alpha=cfg.ema_alpha)

        self.model = YOLO(str(cfg.model_path), task="detect")
        if str(cfg.model_path).endswith(".pt"):
            self.model.fuse()

        is_engine = str(cfg.model_path).endswith(".engine")

        if is_engine:
            if not cfg.data_yaml:
                raise ValueError("data_yaml required for .engine models")
            self.class_names = _load_names_from_yaml(cfg.data_yaml)
            try:
                self.model.names = self.class_names
            except Exception:
                pass
            logger.info(
                f"Loaded {len(self.class_names)} class names "
                f"from {cfg.data_yaml}: {list(self.class_names.values())}"
            )
        else:
            self.class_names = dict(self.model.names)
            logger.info(
                f"Using built-in class names "
                f"({len(self.class_names)} classes): {list(self.class_names.values())}"
            )
        self._warmup()

    def _get_class_name(self, cls_id: int) -> str:
        """Возвращает имя класса по индексу. Никогда не падает."""
        return self.class_names.get(int(cls_id), f"class_{int(cls_id)}")

    # ── Инициализация ─────────────────────────────────────────────────────────

    def _warmup(self) -> None:
        """
        Прогрев CUDA и TensorRT-движка до начала основного цикла.
        Без прогрева первые кадры обрабатываются в 3-5 раз медленнее.
        """
        logger.info(f"Warming up model ({self.cfg.warmup_runs} runs)...")
        dummy = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), dtype=np.uint8)
        for _ in range(self.cfg.warmup_runs):
            self.model.predict(
                dummy,
                imgsz=self.cfg.imgsz,
                half=self.cfg.half,
                device=self.device,
                verbose=False,
            )
        logger.info("Warmup done")

    # ── Инференс ──────────────────────────────────────────────────────────────

    def predict(self, frame: np.ndarray):
        """Один прогон инференса. Возвращает ultralytics Results."""
        return self.model.predict(
            frame,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            half=self.cfg.half,
            device=self.device,
            verbose=False,
        )

    # ── Треккинг ────────────────────────────────────────────────────────────

    def track(self, frame: np.ndarray) -> dict | None:
        """
        Инференс + ByteTrack + EMA-сглаживание.
        Возвращает ОДИН трек (с наибольшим confidence) или None.
        """
        results = self.model.track(
            frame,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.conf,
            iou=self.cfg.iou,
            half=self.cfg.half,
            device=self.device,
            tracker=TRACKER_CONFIG,
            persist=True,
            verbose=False,
        )

        if not self.has_detections(results):
            self._ema.reset()  # сброс фильтра при потере объекта
            return None

        boxes = results[0].boxes
        confs = boxes.conf.cpu().numpy()

        # Выбираем бокс с наибольшим confidence
        best_idx = int(confs.argmax())

        xyxy = boxes.xyxy.cpu().numpy().astype(float)[best_idx]
        conf = float(confs[best_idx])
        cls_id = int(boxes.cls.cpu().numpy()[best_idx])
        class_name = self._get_class_name(cls_id)
        track_id = (
            int(boxes.id.cpu().numpy()[best_idx]) if boxes.id is not None else None
        )

        # EMA-сглаживание
        smooth_bbox = self._ema.update(list(xyxy))

        return {
            "track_id": track_id,
            "bbox": smooth_bbox,
            "conf": conf,
            "class_id": cls_id,
            "class": class_name,
        }

    def has_detections(self, results) -> bool:
        return (
            results is not None
            and len(results) > 0
            and results[0].boxes is not None
            and len(results[0].boxes) > 0
        )

    # ── Расстояние ────────────────────────────────────────────────────────────

    def estimate_distance_zone(
        self, box: np.ndarray, frame_shape: tuple
    ) -> tuple[float, str]:
        """
        Оценка расстояния методом Bbox Area Ratio.
        Формула:
            area_ratio = (bbox_w * bbox_h) / (frame_w * frame_h)
        Чем больше шарик занимает кадр — тем он ближе.
        Пороги настраиваются в InferenceConfig.
        Возвращает (area_ratio, zone).
        """
        x1, y1, x2, y2 = box
        frame_h, frame_w = frame_shape[:2]
        frame_area = frame_w * frame_h
        if frame_area == 0:
            return 0.0, ZONE_UNKNOWN
        bbox_area = (x2 - x1) * (y2 - y1)
        area_ratio = bbox_area / frame_area
        if area_ratio >= self.cfg.distance_close_threshold:
            zone = ZONE_CLOSE
        elif area_ratio <= self.cfg.distance_far_threshold:
            zone = ZONE_FAR
        else:
            zone = ZONE_MEDIUM
        return round(float(area_ratio), 5), zone

    # ── Данные для логирования ────────────────────────────────────────────────

    def extract_detections(
        self, results, frame_shape: Optional[tuple] = None
    ) -> list[dict]:
        """
        Извлекает детекции в сериализуемый список словарей.
        Если передан frame_shape — добавляет area_ratio и distance_zone.
        """
        if not self.has_detections(results):
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        detections = []
        for box, cls_id, confidence in zip(boxes, classes, confs):
            area_ratio, zone = (
                self.estimate_distance_zone(box, frame_shape)
                if frame_shape is not None
                else (None, ZONE_UNKNOWN)
            )
            detections.append(
                {
                    "class": self._get_class_name(cls_id),
                    "confidence": round(float(confidence), 4),
                    "area_ratio": area_ratio,  # доля площади кадра
                    "distance_zone": zone,  # FAR / MEDIUM / CLOSE / UNKNOWN
                    "bounding_box": {
                        "x1": int(box[0]),
                        "y1": int(box[1]),
                        "x2": int(box[2]),
                        "y2": int(box[3]),
                    },
                }
            )
        return detections

    # ── Визуализация ──────────────────────────────────────────────────────────

    def draw(
        self,
        frame: np.ndarray,
        results,
        detections: Optional[list[dict]] = None,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        """
        Рисует боксы на кадре.
        Цвет бокса = зона расстояния: зелёный(FAR) / оранжевый(MEDIUM) / красный(CLOSE).
        detections передаются отдельно — зона уже посчитана, не пересчитываем.
        """
        vis = frame.copy()

        if self.has_detections(results):
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for i, (box, cls_id, confidence) in enumerate(zip(boxes, classes, confs)):
                if detections and i < len(detections):
                    zone = detections[i].get("distance_zone", ZONE_UNKNOWN)
                    ratio = detections[i].get("area_ratio")
                    color = self.cfg.zone_color(zone)
                    ratio_str = f" ({ratio:.3f})" if ratio is not None else ""
                    label = f"{self._get_class_name(cls_id)} {confidence:.2f} [{zone}{ratio_str}]"
                else:
                    color = self.cfg.zone_color(ZONE_UNKNOWN)
                    label = f"{self._get_class_name(cls_id)} {confidence:.2f}"

                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    vis,
                    (box[0], box[1] - th - 6),
                    (box[0] + tw, box[1]),
                    color,
                    -1,
                )
                cv2.putText(
                    vis,
                    label,
                    (box[0], box[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        if fps is not None and self.cfg.show_fps_overlay:
            cv2.putText(
                vis,
                f"FPS: {fps:.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return vis

    def draw_track(
        self,
        frame: np.ndarray,
        track: Optional[dict],
        output,
        target_cx_norm: float = 0.5,
        target_cy_norm: float = 0.5,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        """
        Визуализация для режима track().
        Рисует: bbox объекта, крестик цели, крестик текущего центра,
        линию между ними и HUD с ошибками и скоростями.

        target_cx_norm / target_cy_norm передаются из PIDConfig.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Крестик цели (синий)
        tx = int(target_cx_norm * w)
        ty = int(target_cy_norm * h)
        cv2.drawMarker(
            vis,
            (tx, ty),
            (255, 100, 0),
            cv2.MARKER_CROSS,
            20,
            2,
            cv2.LINE_AA,
        )

        if track is not None and not output.lost:
            x1, y1, x2, y2 = [int(v) for v in track["bbox"]]
            _, zone = self.estimate_distance_zone(
                np.array([x1, y1, x2, y2]), frame.shape
            )
            color = self.cfg.zone_color(zone)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            tid = track.get("track_id")
            conf = track.get("conf", 0.0)
            label = (
                f"ID:{tid} {conf:.2f} [{zone} {output.area_ratio:.3f}]"
                if tid is not None
                else f"{conf:.2f} [{zone} {output.area_ratio:.3f}]"
            )
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                vis,
                (x1, y1 - th - 6),
                (x1 + tw, y1),
                color,
                -1,
            )
            cv2.putText(
                vis,
                label,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # Крестик текущего центра (красный) и линия к цели
            cx = int(output.cx_norm * w)
            cy = int(output.cy_norm * h)
            cv2.drawMarker(
                vis,
                (cx, cy),
                (0, 0, 255),
                cv2.MARKER_CROSS,
                15,
                2,
                cv2.LINE_AA,
            )
            # cv2.line(vis, (tx, ty), (cx, cy), (0, 200, 255), 1, cv2.LINE_AA)

        # HUD
        status = "LOST" if output.lost else f"TRACKING  ID:{output.track_id}"
        hud = [
            f"Status : {status}",
            f"Error X:{output.error_x:+.3f}  Y:{output.error_y:+.3f}  Z:{output.error_z:+.3f}",
            f"Yaw:{output.yaw_angle_deg:+.2f}°  Thrust:{output.thrust:+.2f}  Pitch:{output.pitch_deg:+.2f}°",
            f"Area {output.area_ratio:.4f}",
        ]
        if fps is not None and self.cfg.show_fps_overlay:
            hud.append(f"FPS    {fps:.1f}")

        for i, line in enumerate(hud):
            cv2.putText(
                vis,
                line,
                (10, 24 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return vis

    # ── Логирование ───────────────────────────────────────────────────────────

    def append_log(self, log_path: Path, detections: list[dict]) -> None:
        """
        Дописывает одну запись в JSONL-файл (JSON Lines).
        Не накапливает данные в RAM — безопасно для долгих сессий.
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "detected_objects": detections,
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_track(self, log_path: Path, track: Optional[dict], output) -> None:
        """
        JSONL-логирование для режима track().
        Записывает трек + ошибки PID + скорости.
        """
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "track": {
                "track_id": track.get("track_id") if track else None,
                "conf": track.get("conf") if track else None,
                "bbox": track.get("bbox") if track else None,
            },
            "control": {
                "lost": output.lost,
                "area_ratio": output.area_ratio,
                "error_x": output.error_x,
                "error_y": output.error_y,
                "error_z": output.error_z,
                "yaw": output.yaw_angle_deg,
                "thrust": output.thrust,
                "pitch": output.pitch_deg,
            },
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
