"""
pid_controller.py — PID-контроллер для режима следования за шариком.

Получает положение объекта в кадре, возвращает скорости для дрона.

Что происходит внутри update():
  1. Вычисляет нормализованные ошибки по X, Y, Z
  2. Применяет deadband — обнуляет малые ошибки
  3. Прогоняет ошибки через три simple_pid → скорости vx, vy, vz
  4. Возвращает ControlOutput со всеми промежуточными значениями
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from simple_pid import PID

from pid_config import PIDConfig

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
    cx_norm:    float          # нормализованный центр X ∈ [0, 1]
    cy_norm:    float          # нормализованный центр Y ∈ [0, 1]
    area_ratio: float          # текущая площадь bbox / площадь кадра

    # Ошибки после deadband ∈ [-1, 1]
    error_x: float
    error_y: float
    error_z: float

    # Выходные скорости (м/с)
    # vx: float                  # право+ / лево-
    # vy: float                  # вниз+  / вверх- (NED!)
    # vz: float                  # вперёд+ / назад-

    vx: float                  # roll_deg  (наклон вправо+/влево-)
    vy: float                  # thrust    (0..1)
    vz: float                  # pitch_deg (наклон вперёд+/назад-)


    lost: bool = False         # True — объект не найден, нужен hover


class PIDController:
    """
    Три независимых PID-контроллера (X, Y, Z).

    Принимает: один трек dict | None  (от BalloonDetector.track())
    Возвращает: ControlOutput со скоростями vx, vy, vz

    Про setpoint=0 и знак входа:
        simple_pid считает: output = K * (setpoint - input)
        setpoint=0 → output = -K * input
        Нам нужно output = +K * error (лети туда, где ошибка положительна).
        Решение: передаём input = -error → output = K * error ✓
    """

    def __init__(self, cfg: PIDConfig):
        self.cfg = cfg
        self._lost_counter = 0

        # self.pid_x = PID(
        #     Kp=cfg.pid_x_kp, Ki=cfg.pid_x_ki, Kd=cfg.pid_x_kd,
        #     setpoint=0.0,
        #     output_limits=(-cfg.max_vx, cfg.max_vx),
        # )
        # self.pid_y = PID(
        #     Kp=cfg.pid_y_kp, Ki=cfg.pid_y_ki, Kd=cfg.pid_y_kd,
        #     setpoint=0.0,
        #     output_limits=(-cfg.max_vy, cfg.max_vy),
        # )
        # self.pid_z = PID(
        #     Kp=cfg.pid_z_kp, Ki=cfg.pid_z_ki, Kd=cfg.pid_z_kd,
        #     setpoint=0.0,
        #     output_limits=(-cfg.max_vz, cfg.max_vz),
        # )

        self.pid_roll = PID(
            Kp=cfg.pid_x_kp, Ki=cfg.pid_x_ki, Kd=cfg.pid_x_kd,
            setpoint=0.0,
            output_limits=(-cfg.max_roll_deg, cfg.max_roll_deg),
        )
        self.pid_thrust = PID(
            Kp=cfg.pid_y_kp, Ki=cfg.pid_y_ki, Kd=cfg.pid_y_kd,
            setpoint=0.0,
            output_limits=(-cfg.max_thrust_delta, cfg.max_thrust_delta),
        )
        self.pid_pitch = PID(
            Kp=cfg.pid_z_kp, Ki=cfg.pid_z_ki, Kd=cfg.pid_z_kd,
            setpoint=0.0,
            output_limits=(-cfg.max_pitch_deg, cfg.max_pitch_deg),
        )

        logger.info(
            f"PIDController ready | "
            f"target=({cfg.target_cx_norm}, {cfg.target_cy_norm}) | "
            f"area={cfg.target_area_ratio}"
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

        # error_z > 0 → шарик меньше целевой площади → он далеко → лети вперёд
        error_z_raw = (self.cfg.target_area_ratio - area_ratio) / self.cfg.target_area_ratio
        error_z_raw = float(np.clip(error_z_raw, -1.0, 1.0))

        # ── Шаг 3: deadband ───────────────────────────────────────────────
        # Обнуляем малые ошибки → нет накопления I-члена → нет дрожания.
        error_x = 0.0 if abs(error_x_raw) < self.cfg.deadband_x else error_x_raw
        error_y = 0.0 if abs(error_y_raw) < self.cfg.deadband_y else error_y_raw
        error_z = 0.0 if abs(error_z_raw) < self.cfg.deadband_z else error_z_raw

        # ── Шаг 4: PID → скорости ─────────────────────────────────────────
        # vx = float(self.pid_x(-error_x))
        # vy = float(self.pid_y(-error_y))
        # vz = float(self.pid_z(-error_z))

            # ── Шаг 4: PID → УГЛЫ и ТЯГА ─────────────────────────────────────
        # error_x > 0 → шар правее → нужно наклониться вправо (roll > 0)
        roll_deg = float(self.pid_roll(-error_x))

        # error_z > 0 → шар далеко (area меньше целевой) → наклониться вперёд (pitch > 0)
        pitch_deg = float(self.pid_pitch(-error_z))

        # error_y > 0 → шар ниже центра → дрону надо опуститься (уменьшить тягу)
        thrust_delta = float(self.pid_thrust(+error_y))   # знак можно подправить по месту
        thrust = float(np.clip(self.cfg.base_thrust - thrust_delta, 0.0, 1.0))


        output = ControlOutput(
            track_id=track_id, class_name=track.get("class", None),
            cx_norm=cx_norm, cy_norm=cy_norm, area_ratio=area_ratio,
            error_x=error_x, error_y=error_y, error_z=error_z,
            vx=roll_deg, vy=thrust, vz=pitch_deg,
            lost=False,
        )
        logger.debug(
            f"[ID:{track_id}, class_name: {track.get('class', None)}] "
            f"err=({error_x:.3f}, {error_y:.3f}, {error_z:.3f}) | "
            f"vel=({roll_deg:.3f}, {thrust:.3f}, {pitch_deg:.3f}) | "
            f"area: {area_ratio}"
        )
        return output

    def reset(self) -> None:
        """
        Сбрасывает I и D составляющие всех трёх PID.
        Вызывать при длительной потере объекта или при старте нового трека.
        Без сброса накопленный I-член при повторном обнаружении объекта
        даст ложный импульс скорости.
        """
        # for pid in (self.pid_x, self.pid_y, self.pid_z):
        #     pid.reset()
        for pid in (self.pid_roll, self.pid_thrust, self.pid_pitch):
            pid.reset()
        logger.debug("PIDs reset")

    # ── Внутренние методы ──────────────────────────────────────────────────

    def _handle_lost(self) -> ControlOutput:
        self._lost_counter += 1
        if self._lost_counter >= self.cfg.max_lost_frames:
            self.reset()
        logger.debug(
            f"Object lost ({self._lost_counter}/{self.cfg.max_lost_frames})"
        )
        return ControlOutput(
            track_id=None, class_name=None,
            cx_norm=0.5, cy_norm=0.5, area_ratio=0.0,
            error_x=0.0, error_y=0.0, error_z=0.0,
            vx=0.0, vy=0.0, vz=0.0,
            lost=True,
        )
