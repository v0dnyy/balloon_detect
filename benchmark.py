import argparse
import time
import numpy as np
import cv2
from ultralytics import YOLO

def run_benchmark(model_path, data_yaml, n_warmup=10, n_runs=100, imgsz=640):
    model = YOLO(model_path, task="detect")
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    is_engine = model_path.endswith(".engine")
    device = "0" if not model_path.endswith(".onnx") else "cpu"
    half = not model_path.endswith(".onnx")

    # Прогрев
    for _ in range(n_warmup):
        model.predict(dummy, imgsz=imgsz, half=half, device=device, verbose=False)

    # Замер
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(dummy, imgsz=imgsz, half=half, device=device, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)  # в мс

    times = np.array(times)
    return {
        "mean_ms":   round(times.mean(), 2),
        "min_ms":    round(times.min(), 2),
        "max_ms":    round(times.max(), 2),
        "p95_ms":    round(np.percentile(times, 95), 2),
        "fps":       round(1000 / times.mean(), 1),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt",     type=str, default=None)
    parser.add_argument("--onnx",   type=str, default=None)
    parser.add_argument("--engine", type=str, default=None)
    parser.add_argument("--data",   type=str, default="data.yaml")
    parser.add_argument("--runs",   type=int, default=100)
    parser.add_argument("--imgsz",  type=int, default=640)
    args = parser.parse_args()

    models = {}
    if args.pt:     models["PyTorch (.pt)"]     = args.pt
    if args.onnx:   models["ONNX"]              = args.onnx
    if args.engine: models["TensorRT (.engine)"] = args.engine

    if not models:
        print("Укажи хотя бы одну модель: --pt / --onnx / --engine")
        return

    print(f"\n{'Формат':<22} {'mean':>8} {'min':>8} {'max':>8} {'p95':>8} {'FPS':>8}")
    print("-" * 62)

    for name, path in models.items():
        print(f"  Бенчмарк: {name} ({path})...")
        try:
            r = run_benchmark(path, args.data, n_runs=args.runs, imgsz=args.imgsz)
            print(f"{name:<22} {r['mean_ms']:>7}ms {r['min_ms']:>7}ms {r['max_ms']:>7}ms {r['p95_ms']:>7}ms {r['fps']:>7} FPS")
        except Exception as e:
            print(f"{name:<22} ОШИБКА: {e}")

    print()

if __name__ == "__main__":
    main()