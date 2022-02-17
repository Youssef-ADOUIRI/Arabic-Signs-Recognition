import numpy as np
import cv2

M = np.ones((2,3), dtype=int) * 2
print(M)

cam = cv2.VideoCapture(0 , cv2.CAP_DSHOW)

while True:
    check, frame = cam.read()

    cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()