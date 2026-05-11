"""
Экспорт обученной модели в ONNX формат.
Пример запуска:
    python export_onnx.py --model best.pt
    python export_onnx.py --model best.pt --imgsz 416 --dynamic
    python export_onnx.py --model best.pt --verify  # проверить модель после экспорта
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
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument("--model", type=str, required=True, help="Path to .pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch size (нужен если batch != 1)")
    parser.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX graph (default: True)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    parser.add_argument("--verify", action="store_true", help="Verify exported model via onnxruntime")
    return parser.parse_args()


def verify_onnx(onnx_path: Path, imgsz: int) -> None:
    """Прогоняет тестовый кадр через onnxruntime и проверяет что модель не падает."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        logger.warning("onnxruntime не установлен — пропускаем верификацию.")
        logger.warning("Установи: pip install onnxruntime")
        return

    logger.info("Верификация модели через onnxruntime...")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape  # [batch, C, H, W] или ['batch', C, H, W]

    # Заменяем динамические измерения на 1
    resolved = [dim if isinstance(dim, int) else 1 for dim in input_shape]
    dummy = np.random.rand(*resolved).astype(np.float32)

    outputs = session.run(None, {input_name: dummy})
    logger.info(f"  Input:  {input_name} {resolved}")
    logger.info(f"  Output: {[o.shape for o in outputs]}")
    logger.info("✅ Верификация пройдена — модель корректна")


def main():
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if model_path.suffix != ".pt":
        raise ValueError(f"Ожидается .pt файл, получен: {model_path.suffix}")

    logger.info(f"Загружаем модель: {model_path}")
    model = YOLO(str(model_path))

    export_kwargs = dict(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
        half=False,
        device="cpu",
    )

    logger.info(
        f"Экспортируем в ONNX "
        f"(imgsz={args.imgsz}, opset={args.opset}, "
        f"dynamic={args.dynamic}, simplify={args.simplify})..."
    )

    onnx_path = Path(model.export(**export_kwargs))
    logger.info(f"✅ Сохранено: {onnx_path}")
    logger.info(f"   Размер файла: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")

    if args.verify:
        verify_onnx(onnx_path, args.imgsz)


if __name__ == "__main__":
    main()
