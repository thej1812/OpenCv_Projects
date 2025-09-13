import cv2
import time
from fer import FER
import pyttsx3  # Text-to-speech

# Load face & eye detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# FER emotion detector
emotion_detector = FER(mtcnn=True)

# Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # Speed
engine.setProperty("volume", 1)  # Max volume

# Video capture
cap = cv2.VideoCapture(0)

eye_closed_start = None
SLEEP_THRESHOLD = 3  # seconds
alert_given = False  # To avoid repeating voice too often

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Eye detection
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) == 0:  # No eyes detected
            if eye_closed_start is None:
                eye_closed_start = time.time()
                alert_given = False
            else:
                elapsed = time.time() - eye_closed_start
                if elapsed >= SLEEP_THRESHOLD and not alert_given:
                    cv2.putText(frame, "SLEEPING!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    engine.say("Wake up! Wake up!")
                    engine.runAndWait()
                    alert_given = True
        else:
            eye_closed_start = None
            alert_given = False

        # Emotion detection
        emotions = emotion_detector.detect_emotions(frame)
        if emotions:
            top_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
            cv2.putText(frame, f"Emotion: {top_emotion}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face & Emotion + Sleep Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
