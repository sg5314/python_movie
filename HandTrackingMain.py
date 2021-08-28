import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
cTime = 0
pTime = 0


def cleanAndExit():
        
    print("Bye!")
    sys.exit()

while True:
    try:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    #if id == 4:
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, f'FPS: {int(fps)}', (30, 60), cv2.FONT_HERSHEY_COMPLEX,1,
                    (255,0,0),3)


        cv2.imshow('Image',img)
        cv2.waitKey(1)

    except (KeyboardInterrupt, SystemExit):
        cleanAndExit()
