



# import cv2
# import numpy as np
# import time

# # ------------------ Camera Setup ------------------
# cap = cv2.VideoCapture(0)
# time.sleep(2)  # Warm up camera

# # Capture background (no person)
# background = 0
# for i in range(60):
#     ret, background = cap.read()
# background = np.flip(background, axis=1)

# # ------------------ Main Loop ------------------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = np.flip(frame, axis=1)  # Mirror view
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # ------------------ Pink Color Range ------------------
#     lower_pink1 = np.array([145, 50, 50])
#     upper_pink1 = np.array([170, 255, 255])
#     lower_pink2 = np.array([160, 50, 50])
#     upper_pink2 = np.array([180, 255, 255])

#     mask1 = cv2.inRange(hsv, lower_pink1, upper_pink1)
#     mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
#     mask = mask1 + mask2

#     # Remove noise and smooth edges
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5,5), np.uint8))

#     mask_inv = cv2.bitwise_not(mask)

#     # ------------------ Segment Cloak / Body ------------------
#     res1 = cv2.bitwise_and(background, background, mask=mask)  # replace cloak/body with background
#     res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)        # keep rest of the scene

#     # Combine
#     final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

#     cv2.imshow('Invisible Body', final_output)

#     if cv2.waitKey(1) & 0xFF == 27:  # ESC
#         break

# cap.release()
# cv2.destroyAllWindows()




import cv2
import numpy as np
import time

# ------------------ Camera Setup ------------------
cap = cv2.VideoCapture(0)
time.sleep(2)  # wait for camera to warm up

# Capture background
background = 0
for i in range(60):
    ret, background = cap.read()
background = np.flip(background, axis=1)  # flip for consistency

# ------------------ Main Loop ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = np.flip(frame, axis=1)  # mirror view

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # ------------------ Color Range for Red Cloak ------------------
    # Red can have two ranges in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine masks
    mask = mask1 + mask2

    # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))

    # ------------------ Create Inverse Mask ------------------
    mask_inv = cv2.bitwise_not(mask)

    # Segment cloak and background
    res1 = cv2.bitwise_and(background, background, mask=mask)  # background where cloak is
    res2 = cv2.bitwise_and(frame, frame, mask=mask_inv)        # rest of the frame

    # Combine images
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Display
    cv2.imshow('Invisible Cloak', final_output)

    # Exit
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()