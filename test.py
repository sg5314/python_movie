import cv2
import time 
import numpy as np
import sys
#import HandTrackingModule as htm

#########################
wCam, hCam = 1280,720
#########################

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
print(cap.isOpened())
pTime = 0

def cleanAndExit():
        
    print("Bye!")
    sys.exit()

while True:
    try:
        success, img = cap.read()

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (30, 60), cv2.FONT_HERSHEY_COMPLEX,1,
                    (255,0,0),3)

        cv2.imshow('Img',img)
        cv2.waitKey(1)
    except (KeyboardInterrupt, SystemExit):
        cleanAndExit()
