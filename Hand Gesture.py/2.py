import cv2
import mediapipe as mp
from math import hypot
import numpy as np
import os

cap = cv2.VideoCapture(0)
cap.set(3, 1280)  
cap.set(4, 720)   

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

volMin, volMax = 0, 100

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    lmList = []

    if results.multi_hand_landmarks:
        for handLandmarks in results.multi_hand_landmarks:
            for landmark in mpHands.HandLandmark:
                lm = handLandmarks.landmark[landmark]
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([landmark, cx, cy])
                mpDraw.draw_landmarks(img, handLandmarks, mpHands.HAND_CONNECTIONS)

    thumb_x, thumb_y = None, None
    index_x, index_y = None, None
    if lmList:
        for lm in lmList:
            landmark, x, y = lm
            if landmark == mpHands.HandLandmark.THUMB_TIP:
                thumb_x, thumb_y = x, y
            elif landmark == mpHands.HandLandmark.INDEX_FINGER_TIP:
                index_x, index_y = x, y

        if thumb_x is not None and thumb_y is not None and index_x is not None and index_y is not None:
            cv2.circle(img, (thumb_x, thumb_y), 15, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (index_x, index_y), 15, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 3)
            length = hypot(index_x - thumb_x, index_y - thumb_y)
            vol = np.interp(length, [15, 220], [volMin, volMax])
            print(vol, length)
            osascript_command = f"osascript -e 'set volume output volume {vol}'"
            os.system(osascript_command)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xff == ord('q'): 
        break  

cap.release()
cv2.destroyAllWindows()