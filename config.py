import warnings
from dataclasses import dataclass, field
from pathlib import Path


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
    data_yaml: str | None = "data.yaml"

    # ── Камера ────────────────────────────────────────────────────────────────
    camera_id: int = 0
    camera_fps_fallback: int = 30  # если камера вернула 0 FPS

    # ── MAVLink ───────────────────────────────────────────────────────────────
    mav_port: str = "/dev/ttyTHS0"

    # ── Расстояние (Bbox Area Ratio) ──────────────────────────────────────────
    distance_close_threshold: float = 0.01  # >= 8% площади кадра → CLOSE
    distance_far_threshold: float = 0.002  # <= 2% площади кадра → FAR

    # Кулдаун между MAVLink-алертами одной зоны (секунды).
    mav_alert_cooldown_s: float = 5.0

    # ── Вывод ─────────────────────────────────────────────────────────────────
    output_dir: Path = Path("./output")
    logs_dir: Path = Path("./logs")
    show_fps_overlay: bool = True

    # ── Визуализация ──────────────────────────────────────────────────────────
    zone_colors: dict = field(
        default_factory=lambda: {
            "FAR": (50, 205, 50),  # зелёный
            "MEDIUM": (255, 212, 67),  # оранжевый
            "CLOSE": (0, 0, 220),  # красный
            "UNKNOWN": (180, 180, 180),  # серый
        }
    )

    def zone_color(self, zone: str) -> tuple:
        return self.zone_colors.get(zone, self.zone_colors["UNKNOWN"])

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
