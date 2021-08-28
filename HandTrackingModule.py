import cv2
import mediapipe as mp
import time
import sys


class handDetector():
    def __init__(self, mode = False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw = True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:#手を認識したら（recognize hands）
            for handLms in self.results.multi_hand_landmarks:# 両方の手を描写
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, 
                                            self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo = 0, draw = True):

        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]# 両手ならどちらの手を選択
            
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 7, (255,0,0), cv2.FILLED)

        return lmList
        
        

def cleanAndExit():
        
    print("Bye!")
    sys.exit()

def main():
    cTime = 0
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

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


if __name__ == '__main__':
    main()