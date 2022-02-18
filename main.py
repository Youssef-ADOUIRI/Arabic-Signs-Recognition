import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2

Mcc = np.ones((2, 3), dtype=int) * 2
print(Mcc)

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    check, frame = cam.read()

    cv2.imshow('video', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()
