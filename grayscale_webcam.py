import cv2  # Import OpenCV

# Start webcam (0 = default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Show the grayscale video
    cv2.imshow("Grayscale Webcam", gray)

    # If 'q' is pressed → quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam & close window
cap.release()
cv2.destroyAllWindows()