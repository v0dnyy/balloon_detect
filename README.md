# UAV Balloon Detection — Inference Scripts

Набор скриптов для инференса YOLOv11s/YOLO26s на **Nvidia Jetson Orin Nano**.

## ⚙️ Конфигурация

Все параметры инференса задаются в `config.py` через датакласс `InferenceConfig`.
Ключевые параметры:

| Параметр | По умолчанию | Описание |
|---|---|---|
| `model_path` | `best.pt` | Путь к весам (`.pt`, `.engine`, `.onnx`) |
| `imgsz` | `640` | Размер изображения |
| `conf` | `0.50` | Порог уверенности |
| `iou` | `0.65` | Порог IoU |
| `half` | `True` | FP16 (рекомендуется на Jetson) |
| `camera_id` | `0` | Индекс камеры |
| `mav_port` | `/dev/ttyTHS0` | UART-порт MAVLink |
| `output_dir` | `./output` | Директория для аннотированных файлов |
| `logs_dir` | `./logs` | Директория для JSONL-логов |

## 📦 Экспорт модели

### ONNX

```bash
python export_onnx.py --model best.pt
python export_onnx.py --model best.pt --imgsz 416 --dynamic
python export_onnx.py --model best.pt --verify   # верификация через onnxruntime
```

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--model` | — | Путь к `.pt` файлу (обязательный) |
| `--imgsz` | `640` | Размер изображения |
| `--dynamic` | `False` | Динамический batch size |
| `--opset` | `17` | ONNX opset версия |
| `--verify` | `False` | Проверить модель через onnxruntime |

### TensorRT (только на Jetson)

> ⚠️ Запускать **только на Jetson** — `.engine` файл привязан к железу.

```bash
python export_tensorrt.py --model best.pt
python export_tensorrt.py --model best.pt --imgsz 416 --int8   # INT8, быстрее
```

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--model` | — | Путь к `.pt` файлу (обязательный) |
| `--imgsz` | `640` | Размер изображения |
| `--half` | `True` | FP16 |
| `--int8` | `False` | INT8 квантизация |
| `--data` | `None` | `data.yaml` для калибровки INT8 |

## 🚀 Запуск инференса

### 1. Стрим с камеры (реальное время)

Оптимизирован для Jetson: TensorRT + FP16, автопереподключение камеры, MAVLink.

```bash
# Боевой режим
python infer_stream.py --model best.engine --half

# С отображением окна и записью видео
python infer_stream.py --model best.engine --camera_id 0 --show --save_video --save_logs

# Без MAVLink (режим отладки)
python infer_stream.py --model best.engine --no_mav --show
```

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--model` | — | Путь к модели (обязательный) |
| `--camera_id` | `0` | Индекс камеры |
| `--conf` | `0.50` | Порог уверенности |
| `--iou` | `0.65` | Порог IoU |
| `--half` | `False` | FP16 (рекомендуется на Jetson) |
| `--show` | `False` | Показать видеоокно |
| `--save_video` | `False` | Сохранить аннотированное видео |
| `--save_logs` | `False` | Сохранить лог в JSONL |
| `--mav_port` | `/dev/ttyTHS0` | MAVLink UART-порт |
| `--no_mav` | `False` | Отключить MAVLink |
| `--max_lost` | `15` | Кадров подряд без сигнала до переподключения |

### 2. Инференс на видеофайле (оффлайн)

```bash
python infer_video.py --model best.engine --input video.mp4 --save_video --save_logs

# С отображением в реальном времени
python infer_video.py --model best.pt --input video.mp4 --show
```

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--model` | — | Путь к модели (обязательный) |
| `--input` | — | Путь к видеофайлу (обязательный) |
| `--half` | `False` | FP16 |
| `--show` | `False` | Показать окно во время обработки |
| `--save_video` | `False` | Сохранить аннотированное видео |
| `--save_logs` | `False` | Сохранить JSONL-лог |

### 3. Пакетный инференс на директории

Обрабатывает изображения и видео рекурсивно. Поддерживаемые форматы: `jpg`, `png`, `bmp`, `tiff`, `webp`, `mp4`, `avi`, `mov`, `mkv`.

```bash
python infer_dir.py --model best.engine --input ./images
python infer_dir.py --model best.engine --input ./images --exts jpg png
```

| Аргумент | По умолчанию | Описание |
|---|---|---|
| `--model` | — | Путь к модели (обязательный) |
| `--input` | — | Путь к директории (обязательный) |
| `--exts` | все форматы | Фильтр расширений, например `jpg png` |

## 🛠️ Запуск через Shell-скрипт

`run_inference.sh` — основной скрипт для запуска боевого инференса. Автоматически передаёт в `infer_stream.py` все нужные параметры, проверяет наличие файлов модели и скрипта.

**Конфигурация в начале скрипта:**

```bash
ENGINE="best.pt"        # путь к модели
CAMERA_ID=0             # /dev/video0
IMGSZ=640
CONF=0.55
IOU=0.65
MAV_PORT="/dev/ttyTHS0"
```

**Запуск:**

```bash
chmod +x run_inference.sh

# Боевой режим (без окна, с записью видео и логов)
./run_inference.sh

# С отображением окна (отладка)
./run_inference.sh --show

# Без MAVLink с окном
./run_inference.sh --no_mav --show

# Переопределить параметры через аргументы
./run_inference.sh --conf=0.6 --camera_id=1 --show
```

Скрипт поддерживает переопределение параметров вида `--ключ=значение` (`--conf`, `--iou`, `--imgsz`, `--camera_id`, `--mav_port`) и произвольные флаги (`--show`, `--no_mav`).

## 📡 MAVLink

`mavlink_communication.py` реализует отправку уведомлений о детекции через протокол MAVLink (STATUSTEXT) и переключение режима полёта в LOITER. При обнаружении объектов `infer_stream.py` автоматически вызывает `send_detection_alert()`.

Подключение по умолчанию: UART `/dev/ttyTHS0`, baud `57600`. Для USB-адаптера замените порт на `/dev/ttyUSB0`.

## 📂 Выходные данные

- `output/` — аннотированные изображения и видео
- `logs/` — JSONL-файлы с детекциями в формате:

```json
{
  "timestamp": "2026-05-12T10:00:00.000",
  "detected_objects": [
    {
      "class": "balloon",
      "confidence": 0.921,
      "bounding_box": {"x1": 120, "y1": 80, "x2": 240, "y2": 200}
    }
  ]
}
```