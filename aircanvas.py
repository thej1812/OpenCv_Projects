import cv2
import numpy as np
import mediapipe as mp
import math
from collections import deque

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

# Canvas setup
canvas = None
color = (255, 255, 255)  # White pen
thickness = 5

# Webcam
cap = cv2.VideoCapture(0)

# Last finger position
last_x, last_y = None, None

# Moving average history
window_size = 7
x_history = deque(maxlen=window_size)
y_history = deque(maxlen=window_size)

# Exponential smoothing
smooth_x, smooth_y = 0, 0
alpha = 0.2  # smoothing factor

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    # Mediapipe detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Index fingertip
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            # Thumb tip
            thumb_x = int(hand_landmarks.landmark[4].x * w)
            thumb_y = int(hand_landmarks.landmark[4].y * h)

            # Append to history
            x_history.append(x)
            y_history.append(y)

            # Moving average
            avg_x = int(np.mean(x_history))
            avg_y = int(np.mean(y_history))

            # Exponential smoothing
            smooth_x = int(alpha * avg_x + (1 - alpha) * smooth_x)
            smooth_y = int(alpha * avg_y + (1 - alpha) * smooth_y)

            # Distance between index and thumb
            distance = math.hypot(thumb_x - x, thumb_y - y)

            # 👉 Write ONLY when thumb & index are close
            if distance < 40:  # pinch threshold
                if last_x is not None and last_y is not None:
                    cv2.line(canvas, (last_x, last_y), (smooth_x, smooth_y), color, thickness)

                last_x, last_y = smooth_x, smooth_y
            else:
                last_x, last_y = None, None

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Merge canvas with frame
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY)
    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, cv2.bitwise_not(inv))
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("AirCanvas", frame)
    cv2.imshow("Canvas Only", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
