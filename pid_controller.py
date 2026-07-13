import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from simple_pid import PID

from pid_config import PIDConfig
import time

logger = logging.getLogger(__name__)


@dataclass
class ControlOutput:
    """
    Результат одного шага PID-управления.
    Содержит всё необходимое для отправки команды дрону и для визуализации.
    """
    # Из bbox
    track_id:   Optional[int]
    class_name: Optional[str]

    # Положение объекта в кадре [0..1]
    cx_norm:    float          # нормализованный центр X ∈ [-1, 1]
    cy_norm:    float          # нормализованный центр Y ∈ [-1, 1]
    area_ratio: float          # текущая площадь bbox / площадь кадра

    # Ошибки после deadband ∈ [-1, 1]
    error_x: float
    error_y: float
    error_z: float

    yaw_angle_deg: float   # накопленный абсолютный угол, право+ / лево-
    thrust: float         # [0..1], 0.5 = hover
    pitch_deg: float      # °,  вперёд+ / назад-


    lost: bool = False         # True — объект не найден, нужен hover


class PIDController:
    """
    Три независимых PID-контроллера (X, Y, Z).

    Принимает: один трек dict | None  (от BalloonDetector.track())
    Возвращает: ControlOutput

    Про знак входа в simple_pid:
      simple_pid вычисляет: output = K * (setpoint - input)
      setpoint=0 → output = -K * input
      Нам нужно output = +K * error → передаём input = -error
    """

    def __init__(self, cfg: PIDConfig):
        self.cfg = cfg
        self._lost_counter = 0
        self._last_time: Optional[float] = None
        self._target_yaw_deg: float = 0.0

        # YAW — горизонтальное смещение, единицы °/с
        self.pid_yaw = PID(
            Kp=cfg.pid_x_kp, Ki=cfg.pid_x_ki, Kd=cfg.pid_x_kd,
            setpoint=0.0,
            output_limits=(-cfg.max_yaw_rate_deg, cfg.max_yaw_rate_deg),
            sample_time=None,   # dt управляется нами вручную
        )

        self.pid_thrust = PID(
            Kp=cfg.pid_y_kp, Ki=cfg.pid_y_ki, Kd=cfg.pid_y_kd,
            setpoint=0.0,
            output_limits=(-cfg.max_thrust_delta, cfg.max_thrust_delta),
            sample_time=None,
        )

        self.pid_pitch = PID(Kp=cfg.pid_z_kp, Ki=cfg.pid_z_ki, Kd=cfg.pid_z_kd,
            setpoint=0.0,
            output_limits=(-cfg.max_pitch_deg, cfg.max_pitch_deg),
            sample_time=None,
        )

        logger.info(
            "PIDController ready | target=(%.2f, %.2f) | area=%.3f",
            cfg.target_cx_norm, cfg.target_cy_norm, cfg.target_area_ratio,
        )

    # ── Публичный метод ────────────────────────────────────────────────────

    def update(
        self,
        track: Optional[dict],  # результат BalloonDetector.track() или None
        frame_shape: tuple,     # (H, W) или (H, W, C)
    ) -> ControlOutput:
        """
        Главный метод — вызывается каждый кадр.

        track — словарь от detector.track():
            {
                "track_id": int | None,
                "bbox":     [x1, y1, x2, y2],  # сглаженные EMA координаты
                "conf":     float,
                "class_id": int,
                "class": str
            }
        """
        now = time.monotonic()
        dt = (now - self._last_time) if self._last_time is not None else 0.033
        self._last_time = now
        # Защита: dt не может быть нулём или аномально большим
        dt = float(np.clip(dt, 0.005, 0.5))

        if track is None:
            return self._handle_lost()

        self._lost_counter = 0
        frame_h, frame_w = frame_shape[:2]
        frame_area = frame_w * frame_h

        x1, y1, x2, y2 = track["bbox"]
        track_id = track.get("track_id")

        # ── Шаг 1: нормализация положения и площади ───────────────────────
        cx_norm    = (x1 + x2) / 2.0 / frame_w   # ∈ [0, 1]
        cy_norm    = (y1 + y2) / 2.0 / frame_h   # ∈ [0, 1]
        area_ratio = ((x2 - x1) * (y2 - y1)) / frame_area

        # ── Шаг 2: сырые ошибки ───────────────────────────────────────────
        # error_x > 0 → шарик правее цели  → лети вправо
        # Делим на 0.5: смещение на полкадра = error 1.0
        error_x_raw = (cx_norm - self.cfg.target_cx_norm) / 0.5

        # error_y > 0 → шарик ниже цели → лети вниз
        error_y_raw = (cy_norm - self.cfg.target_cy_norm) / 0.5

        # error_z: логарифмическая нормировка — линеаризует зависимость 1/d²
        # error_z > 0 → объект дальше цели → нужно лететь вперёд (pitch-)
        if area_ratio > 0:
            error_z_raw = float(np.log(self.cfg.target_area_ratio / area_ratio))
        else:
            error_z_raw = 1.0   # объект не виден → считаем максимально далёким
        error_z_raw = float(np.clip(error_z_raw, -1.0, 1.0))

        # ── Шаг 3: deadband ───────────────────────────────────────────────
        # Обнуляем малые ошибки → нет накопления I-члена → нет дрожания.
        error_x = 0.0 if abs(error_x_raw) < self.cfg.deadband_x else error_x_raw
        error_y = 0.0 if abs(error_y_raw) < self.cfg.deadband_y else error_y_raw
        error_z = 0.0 if abs(error_z_raw) < self.cfg.deadband_z else error_z_raw

        # ── Шаг 4: PID → команды ──────────────────────────────────────────
        # Передаём -error потому что simple_pid считает setpoint - input:
        # при setpoint=0: output = -K*input → чтобы получить +K*error, нужен input=-error
        
        # YAW: error_x > 0 → объект правее → yaw rate вправо (положительный)
        yaw_rate = float(self.pid_yaw(-error_x, dt=dt))
        self._target_yaw_deg += yaw_rate * dt
        self._target_yaw_deg = (self._target_yaw_deg + 180.0) % 360.0 - 180.0

        # PITCH: error_z > 0 → далеко → pitch вперёд (положительный)
        pitch_deg = float(self.pid_pitch(error_z, dt=dt))

        # THRUST: error_y > 0 → объект ниже → снижаемся → уменьшаем тягу
        # thrust_delta < 0 при error_y > 0 (объект ниже → меньше газа)
        thrust_delta = float(self.pid_thrust(error_y, dt=dt))

        pitch_rad = np.radians(pitch_deg)
        cos_pitch = float(np.cos(pitch_rad))
        cos_pitch = max(cos_pitch, 0.5)   # не делим на ~0 при больших углах

        raw_thrust = self.cfg.base_thrust + thrust_delta

        thrust = float(np.clip(raw_thrust / cos_pitch, self.cfg.thrust_min, self.cfg.thrust_max))

        output = ControlOutput(
            track_id=track_id,
            class_name=track.get("class"),
            cx_norm=cx_norm,
            cy_norm=cy_norm,
            area_ratio=area_ratio,
            error_x=error_x,
            error_y=error_y,
            error_z=error_z,
            yaw_angle_deg=self._target_yaw_deg,
            thrust=thrust,
            pitch_deg=pitch_deg,
            lost=False,
        )

        logger.debug(
            "[ID:%s|%s] err=(%.3f, %.3f, %.3f) | yaw=%.1f° thrust=%.3f pitch=%.1f°",
            track_id, track.get("class"),
            error_x, error_y, error_z,
            self._target_yaw_deg, thrust, pitch_deg,
        )
        return output

    def reset(self) -> None:
        """
        Сбрасывает I и D составляющие всех трёх PID и таймер dt.
        Вызывать при потере объекта или при старте нового трека.
        """
        for pid in (self.pid_yaw, self.pid_thrust, self.pid_pitch):
            pid.reset()
        self._last_time = None
        logger.debug("PIDs reset")

    # ── Внутренние методы ──────────────────────────────────────────────────

    def _handle_lost(self) -> ControlOutput:
        self._lost_counter += 1
        if self._lost_counter >= self.cfg.max_lost_frames:
            self.reset()
        logger.debug("Object lost (%d/%d)", self._lost_counter, self.cfg.max_lost_frames)

        # ВАЖНО: thrust = base_thrust, а не 0 — дрон удерживает высоту
        return ControlOutput(
            track_id=None,
            class_name=None,
            cx_norm=0.5,
            cy_norm=0.5,
            area_ratio=0.0,
            error_x=0.0,
            error_y=0.0,
            error_z=0.0,
            yaw_angle_deg=self._target_yaw_deg,
            thrust=self.cfg.base_thrust,
            pitch_deg=0.0,
            lost=True,
        )
    
