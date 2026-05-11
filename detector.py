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
        logger.info("Warmup done ✓")

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

    # ── Данные для логирования ────────────────────────────────────────────────

    def extract_detections(self, results) -> list[dict]:
        """
        Извлекает детекции в сериализуемый список словарей.
        Используется и для логов, и для MAVLink.
        """
        if not self.has_detections(results):
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        detections = []
        for box, cls_id, confidence in zip(boxes, classes, confs):
            detections.append({
                "class": self.model.names[int(cls_id)],
                "confidence": round(float(confidence), 4),
                "bounding_box": {
                    "x1": int(box[0]), "y1": int(box[1]),
                    "x2": int(box[2]), "y2": int(box[3]),
                },
            })
        return detections

    # ── Визуализация ──────────────────────────────────────────────────────────

    def draw(self, frame: np.ndarray, results, fps: Optional[float] = None) -> np.ndarray:
        """
        Рисует боксы и (опционально) FPS-оверлей на кадре.
        Не мутирует оригинальный frame — работает с копией.
        """
        vis = frame.copy()

        if self.has_detections(results):
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()

            for box, cls_id, confidence in zip(boxes, classes, confs):
                color = self.cfg.class_color(cls_id)
                label = f"{self.model.names[int(cls_id)]} {confidence:.2f}"

                cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), color, 2)

                # Подложка под текст для читаемости
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(vis, (box[0], box[1] - th - 6), (box[0] + tw, box[1]), color, -1)
                cv2.putText(vis, label, (box[0], box[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if fps is not None and self.cfg.show_fps_overlay:
            cv2.putText(vis, f"FPS: {fps:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

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
