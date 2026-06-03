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
from ultralytics import YOLO

from config import InferenceConfig

logger = logging.getLogger(__name__)

ZONE_FAR = "FAR"
ZONE_MEDIUM = "MEDIUM"
ZONE_CLOSE = "CLOSE"
ZONE_UNKNOWN = "UNKNOWN"


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


class BalloonDetector:
    def __init__(self, cfg: InferenceConfig):
        torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
        self.cfg = cfg
        self.device = get_device(str(cfg.model_path))
        logger.info(f"Loading model: {cfg.model_path}")
        logger.info(f"Inference device: {self.device}")
        self.model = YOLO(str(cfg.model_path), task="detect")
        if str(cfg.model_path).endswith(".pt"):
            self.model.fuse()
        self._warmup()

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

        boxes   = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs   = results[0].boxes.conf.cpu().numpy()

        detections = []
        for box, cls_id, confidence in zip(boxes, classes, confs):
            area_ratio, zone = (
                self.estimate_distance_zone(box, frame_shape)
                if frame_shape is not None
                else (None, ZONE_UNKNOWN)
            )
            detections.append({
                "class":         self.model.names[int(cls_id)],
                "confidence":    round(float(confidence), 4),
                "area_ratio":    area_ratio,  # доля площади кадра
                "distance_zone": zone,         # FAR / MEDIUM / CLOSE / UNKNOWN
                "bounding_box": {
                    "x1": int(box[0]), "y1": int(box[1]),
                    "x2": int(box[2]), "y2": int(box[3]),
                },
            })
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
            boxes   = results[0].boxes.xyxy.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confs   = results[0].boxes.conf.cpu().numpy()

            for i, (box, cls_id, confidence) in enumerate(zip(boxes, classes, confs)):
                if detections and i < len(detections):
                    zone      = detections[i].get("distance_zone", ZONE_UNKNOWN)
                    ratio     = detections[i].get("area_ratio")
                    color     = self.cfg.zone_color(zone)
                    ratio_str = f" ({ratio:.3f})" if ratio is not None else ""
                    label     = f"{self.model.names[int(cls_id)]} {confidence:.2f} [{zone}{ratio_str}]"
                else:
                    color = self.cfg.class_color(cls_id)
                    label = f"{self.model.names[int(cls_id)]} {confidence:.2f}"

                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), color, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    vis,
                    (box[0], box[1] - th - 6),
                    (box[0] + tw, box[1]),
                    color, -1,
                )
                cv2.putText(
                    vis, label, (box[0], box[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
                )

        if fps is not None and self.cfg.show_fps_overlay:
            cv2.putText(
                vis, f"FPS: {fps:.1f}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
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
