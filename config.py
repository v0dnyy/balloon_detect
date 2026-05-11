from dataclasses import dataclass, field
from pathlib import Path
import warnings


@dataclass
class InferenceConfig:
    """
    Общие настройки для всех скриптов инференса.
    Все изменяемые константы — здесь. Скрипты импортируют только это.
    """
    # ── Модель ────────────────────────────────────────────────────────────────
    model_path: str = "best.pt"  # .pt / .engine (TensorRT) / .onnx
    imgsz: int = 640
    conf: float = 0.50
    iou: float = 0.65
    half: bool = True  # FP16; True на Jetson GPU
    warmup_runs: int = 3  # кол-во прогревочных прогонов

    # ── Камера ────────────────────────────────────────────────────────────────
    camera_id: int = 0
    camera_fps_fallback: int = 30  # если камера вернула 0 FPS

    # ── MAVLink ───────────────────────────────────────────────────────────────
    mav_port: str = "/dev/ttyTHS0"  # UART на Jetson; /dev/ttyUSB0 для USB

    # ── Вывод ─────────────────────────────────────────────────────────────────
    output_dir: Path = Path("./output")
    logs_dir: Path = Path("./logs")
    show_fps_overlay: bool = True

    # ── Визуализация ──────────────────────────────────────────────────────────
    # Детерминированная палитра цветов по классу (без random.seed)
    class_palette: list = field(default_factory=lambda: [
        (255, 56, 56),  # 0 — красный
        (72, 209, 204),  # 1 — бирюзовый
        (255, 212, 67),  # 2 — жёлтый
        (0, 161, 255),  # 3 — синий
        (138, 43, 226),  # 4 — фиолетовый
        (50, 205, 50),  # 5 — зелёный
    ])

    def class_color(self, cls_id: int) -> tuple:
        """Возвращает BGR-цвет для класса без побочных эффектов."""
        return self.class_palette[int(cls_id) % len(self.class_palette)]

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.logs_dir = Path(self.logs_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        if str(self.model_path).endswith(".onnx") and self.half:
            warnings.warn(
                "half=True несовместим с ONNX — автоматически сброшен в False.",
                UserWarning,
                stacklevel=2,
            )
            self.half = False
