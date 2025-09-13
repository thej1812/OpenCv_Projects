import cv2
from fer import FER

# Create detector
detector = FER(mtcnn=True)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions
    emotions = detector.detect_emotions(frame)

    for emotion in emotions:
        (x, y, w, h) = emotion["box"]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Get the top emotion
        top_emotion, score = max(emotion["emotions"].items(), key=lambda x: x[1])
        cv2.putText(frame, f"{top_emotion} ({int(score*100)}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
