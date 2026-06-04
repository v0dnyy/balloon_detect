import logging
import threading
import time

import pymavlink.dialects.v20.all as dialect
import pymavlink.mavutil as utility
from pymavlink import mavutil

from send_angle import set_attitude

logger = logging.getLogger(__name__)


class MAVLinkCommunication:
    def __init__(
        self, port: str = "COM12", source_system: int = 1, source_component: int = 0
    ):
        self.port = port
        self._avoidance_active = False

        try:
            self.connection = utility.mavlink_connection(
                device=port,
                baud=57600,
                source_system=source_system,
                source_component=source_component,
                autoreconnect=True,
            )
            heartbeat = self.connection.wait_heartbeat(timeout=5)
            if not heartbeat:
                raise TimeoutError("Телеметрия не отвечает")
            logger.info(f"MAVLink подключен к {port}")
        except Exception as e:
            logger.error(f"Ошибка подключения MAVLink: {e}")
            self.connection = None

    def _get_current_mode(self) -> str | None:
        """Возвращает название текущего режима полёта или None если не удалось."""
        try:
            msg = self.connection.recv_match(type="HEARTBEAT", blocking=True, timeout=2)
            if not msg:
                return None
            mode_map = self.connection.mode_mapping()
            # Разворачиваем словарь: id → name
            id_to_mode = {v: k for k, v in mode_map.items()}
            return id_to_mode.get(msg.custom_mode)
        except Exception as e:
            logger.error(f"Ошибка получения текущего режима: {e}")
            return None

    def _switch_to_mode(self, mode_name: str) -> bool:
        """Переключает в указанный режим по имени."""
        try:
            mode_map = self.connection.mode_mapping()
            if mode_name not in mode_map:
                logger.warning(f"Режим {mode_name} недоступен")
                return False
            self.connection.mav.set_mode_send(
                self.connection.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_map[mode_name],
            )
            logger.info(f"Режим изменён → {mode_name}")
            return True
        except Exception as e:
            logger.error(f"Ошибка смены режима: {e}")
            return False

    def send_status(self, text: str, severity=dialect.MAV_SEVERITY_INFO) -> None:
        if not self.connection:
            logger.warning("Нет соединения — STATUSTEXT не отправлен")
            return
        text = str(text)[:50]
        msg = dialect.MAVLink_statustext_message(
            severity=severity,
            text=text.encode("utf-8"),
        )
        self.connection.mav.send(msg)
        logger.info(f"STATUSTEXT: {text}")

    def send_detection_alert(
        self,
        detection_count: int,
        class_names: list[str],
        zone: str = "UNKNOWN",
        area_ratio: float = None,
        severity=dialect.MAV_SEVERITY_ALERT,
    ) -> None:
        ratio_str = f" r={area_ratio:.3f}" if area_ratio is not None else ""
        msg = f"DET {detection_count} [{zone}{ratio_str}]: {', '.join(class_names)}"
        self.send_status(msg, severity)

    def send_avoidance_command(self, zone: str) -> None:
        if zone != "CLOSE":
            return
        if not self.connection:
            logger.warning("Нет соединения — avoidance не выполнено")
            return
        if self._avoidance_active:
            logger.info("Avoidance уже выполняется — пропускаем")
            return

        self._avoidance_active = True
        self.send_status(
            "AVOIDANCE: UP+FORWARD", severity=dialect.MAV_SEVERITY_CRITICAL
        )
        logger.warning("CLOSE zone — запуск манёвра уклонения")

        def _avoidance():
            previous_mode = None  # объявляем ДО try — доступна в finally
            mode_switched = False

            try:
                previous_mode = self._get_current_mode()
                logger.info(f"Текущий режим перед манёвром: {previous_mode}")

                if not self._switch_to_mode("GUIDED_NOGPS"):
                    logger.error("Avoidance отменён — не удалось переключить режим")
                    return

                mode_switched = True
                time.sleep(0.3)

                logger.info("Avoidance: фаза 1 — UP (2с)")
                set_attitude(
                    self.connection, pitch_angle=0.0, thrust=0.65, duration=2.0
                )

                logger.info("Avoidance: фаза 2 — FORWARD (2с)")
                set_attitude(
                    self.connection, pitch_angle=-10.0, thrust=0.5, duration=2.0
                )

                logger.info("Avoidance: завершено")

            except Exception as e:
                logger.error(f"Avoidance error: {e}")

            finally:
                if mode_switched:
                    if previous_mode:
                        logger.info(f"Возврат в режим {previous_mode}")
                        self._switch_to_mode(previous_mode)
                    else:
                        logger.warning("Исходный режим неизвестен — переключаем в AUTO")
                        self._switch_to_mode("AUTO")

                self._avoidance_active = False

        threading.Thread(target=_avoidance, daemon=True).start()

    def change_to_loiter(self) -> bool:
        if not self.connection:
            logger.warning("Нет соединения — режим не изменён")
            return False
        try:
            mode_map = self.connection.mode_mapping()
            if "LOITER" not in mode_map:
                logger.warning("LOITER недоступен на этом автопилоте")
                return False
            self.connection.mav.set_mode_send(
                self.connection.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_map["LOITER"],
            )
            logger.info("Режим изменён → LOITER")
            return True
        except Exception as e:
            logger.error(f"Ошибка при смене режима: {e}")
            return False

    def close(self) -> None:
        if self.connection:
            self.connection.close()
            logger.info("MAVLink соединение закрыто")
