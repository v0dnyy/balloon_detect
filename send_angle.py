import time
import math

def to_quaternion(roll=0.0, pitch=0.0, yaw=0.0):
    """
   	Преобразовать углы Эйлера из градусов в кватернионы
    """
    t0 = math.cos(math.radians(yaw * 0.5))
    t1 = math.sin(math.radians(yaw * 0.5))
    t2 = math.cos(math.radians(roll * 0.5))
    t3 = math.sin(math.radians(roll * 0.5))
    t4 = math.cos(math.radians(pitch * 0.5))
    t5 = math.sin(math.radians(pitch * 0.5))

    w = t0 * t2 * t4 + t1 * t3 * t5
    x = t0 * t3 * t4 - t1 * t2 * t5
    y = t0 * t2 * t5 + t1 * t3 * t4
    z = t1 * t2 * t4 - t0 * t3 * t5

    return [w, x, y, z]

def set_attitude(connection, roll_angle=0.0, pitch_angle=0.0,
                 yaw_angle=0, yaw_rate=0.0, use_yaw_rate=0,
                 thrust=0.5, duration=0):
    """
	Устанавливает положение дрона на определенное время
    """
    send_attitude_target(connection, roll_angle, pitch_angle,
                         yaw_angle, yaw_rate, use_yaw_rate,
                         thrust)
    start = time.time()
    while (time.time() - start) < duration:
        send_attitude_target(connection, roll_angle, pitch_angle,
                            yaw_angle, yaw_rate, use_yaw_rate,
                            thrust)
        time.sleep(0.1)


def send_attitude_target(connection, roll_angle=0.0, pitch_angle=0.0,
                         yaw_angle=None, yaw_rate=0.0, use_yaw_rate=False,
                         thrust=0.5):
    """
    Итоговая функция возмет угоы преобразует в кватернионый и отправить сигнал.
    будет работать только на GuidedNoGPS/Guided
    """
    if yaw_angle is None:
        yaw_angle = 0.0

    # Calcular time_boot_ms de forma segura
    if not hasattr(connection, 'start_time'):
        connection.start_time = time.time()
    time_boot_ms = int((time.time() - connection.start_time) * 1000)
    print("envio senal")
    connection.mav.set_attitude_target_send(
        time_boot_ms,
        connection.target_system,
        connection.target_component,
        0b00000111,  # 0b00000011
        to_quaternion(roll_angle, pitch_angle, yaw_angle),
        0, 0, math.radians(yaw_rate), thrust
    )