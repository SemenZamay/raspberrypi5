import cv2
import numpy as np
import time
from picamera2 import Picamera2


# Переменные для хранения координат и времени
prev_center = None
prev_time = None

# Известный диаметр апельсина (в метрах)
real_diameter = 0.07  # 7 см

# Фокусное расстояние (в пикселях) — найдите заранее калибровкой
focal_length = 500  # Пример значения

# Инициализация параметров фильтра Калмана
dt = 1  # Шаг времени (1 секунда)
state = np.array([0, 0, 0, 0])  # [x, y, v_x, v_y]
P = np.eye(4) * 100  # Начальная ковариация
A = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])  # Модель движения
Q = np.eye(4) * 0.1  # Шум процесса
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])  # Модель измерений
R = np.eye(2) * 5  # Шум измерений

def kalman_filter(z, state, P):
    # Предсказание
    state_pred = A @ state
    P_pred = A @ P @ A.T + Q

    # Обновление
    y = z - H @ state_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    state_new = state_pred + K @ y
    P_new = (np.eye(4) - K @ H) @ P_pred
    return state_new, P_new

# Настройка детекции апельсина (цвет в HSV)
lower_orange = np.array([10, 100, 100])  # Нижняя граница оранжевого
upper_orange = np.array([25, 255, 255])  # Верхняя граница оранжевого

# Инициализация камеры
picam2 = Picamera2()
video_config = picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)})
picam2.start()

while True:
    frame = picam2.capture_array()
    # Convert from RGBA to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

    # Конвертация кадра в HSV и поиск апельсина
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Выбор самого большого контура
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)

        if radius > 10:  # Если радиус значимый
            x, y = int(x), int(y)

            # Текущее время и координаты
            current_time = time.time()
            current_center = (x, y)

            # Расчёт расстояния
            distance = (real_diameter * focal_length) / (2 * radius)
            print(f"Расстояние до апельсина: {distance:.2f} метров")

            # Отрисовка информации на кадре
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.putText(frame, f"Distance: {distance:.2f}m", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Обновление фильтра Калмана
            z = np.array([x, y])  # Измеренные координаты
            state, P = kalman_filter(z, state, P)

            # Предсказанное положение через секунду
            predicted_x = int(state[0] + state[2] * dt)
            predicted_y = int(state[1] + state[3] * dt)

            # Отрисовка текущего положения
            cv2.circle(frame, (x, y), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # Центр объекта

            # Отрисовка предсказанного положения
            cv2.circle(frame, (predicted_x, predicted_y), 10, (255, 0, 0), -1)  # Предсказанное положение

            if prev_center is not None and prev_time is not None:
                # Расчёт скорости в пикселях в секунду
                dx = current_center[0] - prev_center[0]
                dy = current_center[1] - prev_center[1]
                distance = (dx**2 + dy**2)**0.5  # Расстояние в пикселях
                delta_time = current_time - prev_time

                if delta_time > 0:  # Избегаем деления на ноль
                    speed = distance / delta_time
                    print(f"Скорость апельсина: {speed:.2f} пикселей/сек")

            # Обновление предыдущих значений
            prev_center = current_center
            prev_time = current_time

    # Отображение кадра
    cv2.imshow('Orange Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
picam2.stop()
