import cv2
import numpy as np

cap = cv2.VideoCapture('traffic.avi')

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# Menyiapkan citra HSV untuk visualisasi flow
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Hitung dense optical flow
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2  # Arah gerakan dalam hue
    hsv[..., 1] = 255  # saturasi maksimum
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # magnitudo flow

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('Farneback Dense Optical Flow', rgb)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    prvs = next_frame

cap.release()
cv2.destroyAllWindows()
