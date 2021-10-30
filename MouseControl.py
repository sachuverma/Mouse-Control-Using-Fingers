import cv2 
import numpy as np
import HandTrackingModule as htm 
import time 
import autopy

wCam, hCam = 640, 480 # Camera Dimensions
frameR = 100 # Frame Reduction
wScr, hScr = autopy.screen.size() # Screen Dimensions
pTime = 0 # Previous Time

smoothening = 7
plocX, plocY = 0, 0 # Previous location of X,Y 
clocX, clocY = 0, 0 # Current location of X,Y

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)

while True:
    # FIND HAND LANDMARKS
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # GET TIP OF INDEX AND MIDDLE FINGERS
    if lmList:
        x1, y1 = lmList[8][1:]  # Index Finger
        x2, y2 = lmList[12][1:] # Middle Finger
        # print(x1, y1, x2, y2)

        # CHECK WHICH FINGERS ARE UP
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR), (255, 0, 255), 2)

        # ONLY INDEX FINGER = MOVING MODE
        if fingers[1] == 1 and fingers[2] == 0:
            # CONVERT COORDINATES
            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

            # SMOOTHEN VALUES
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # MOVE MOUSE
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

    # MIDDLE + INDEX FINGER = CLICKING MODE
        if fingers[1] == 1 and fingers[2] == 1:
            # FIND DISTANCE B/W FINGERS
            length, img, lineInfo = detector.findDistance(8, 12, img)
            # print(length)

            # CLICK MOUSE IF DISTANCE SHORT
            if length < 25:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # SHOW FRAME RATE
    # cTime = time.time()
    # fps = 1 / (cTime - pTime)
    # pTime = cTime 
    # cv2.putText(img, str(int(fps)), (17, 35), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

    # DISPLAY VIDEO FEED
    # cv2.imshow("Image", img)
    cv2.imshow("Image", cv2.flip(img, 1))
    cv2.waitKey(1)