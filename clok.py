import cv2
import numpy as np
import time

# Start webcam
cap = cv2.VideoCapture(0)

# Warm up camera
time.sleep(2)

# Capture background (keep still for a few seconds)
for i in range(60):
    ret, background = cap.read()
background = np.flip(background, axis=1)

# Kernel for mask cleanup
kernel = np.ones((3,3), np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = np.flip(frame, axis=1)   # Mirror effect
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 🎨 Dark Green cloak range
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # --- Improve mask ---
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)  # remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1) # expand cloak area
    mask = cv2.GaussianBlur(mask, (3,3), 0)  # smooth edges

    # Invert mask
    mask_inv = cv2.bitwise_not(mask)

    # Segment out non-cloak part of frame
    res1 = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Segment out cloak part from background
    res2 = cv2.bitwise_and(background, background, mask=mask)

    # Combine
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Show windows
    cv2.imshow("Green Cloak Mask", mask)  # Debug view
    cv2.imshow("Harry Potter Cloak (Dark Green)", final_output)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
