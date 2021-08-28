import cv2
import mediapipe as mp
import time
import sys

import HandTrackingModule as htm




cTime = 0
pTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

def cleanAndExit():
        
    print("Bye!")
    sys.exit()

while True:
    try:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])# 4 = 親指
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (30, 60), cv2.FONT_HERSHEY_COMPLEX,1,
                    (255,0,0),3)


        cv2.imshow('Image',img)
        cv2.waitKey(1)

    except (KeyboardInterrupt, SystemExit):
        cleanAndExit()