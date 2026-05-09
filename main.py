import cv2
import mediapipe as mp
import numpy as np

# Hand Detector Class
class HandDetector:

    def __init__(self, maxHands=1, detectionCon=0.7):

        self.mpHands = mp.solutions.hands

        self.hands = self.mpHands.Hands(
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon
        )

        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:

            for handLms in self.results.multi_hand_landmarks:

                self.mpDraw.draw_landmarks(
                    img,
                    handLms,
                    self.mpHands.HAND_CONNECTIONS
                )

        return img

    def findPosition(self, img):

        lmList = []

        if self.results.multi_hand_landmarks:

            myHand = self.results.multi_hand_landmarks[0]

            for id, lm in enumerate(myHand.landmark):

                h, w, c = img.shape

                cx, cy = int(lm.x * w), int(lm.y * h)

                lmList.append([cx, cy])

        return lmList

    def fingersUp(self, lmList):

        fingers = []

        # Index Finger
        if lmList[8][1] < lmList[6][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Middle Finger
        if lmList[12][1] < lmList[10][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        return fingers


# Webcam
cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)

# Hand Detector
detector = HandDetector()

# Drawing Variables
xp, yp = 0, 0

canvas = np.zeros((720, 1280, 3), np.uint8)

while True:

    success, img = cap.read()

    img = cv2.flip(img, 1)

    img = detector.findHands(img)

    lmList = detector.findPosition(img)

    if len(lmList) != 0:

        x1, y1 = lmList[8]   # Index Finger
        x2, y2 = lmList[12]  # Middle Finger

        fingers = detector.fingersUp(lmList)

        # Drawing Mode
        if fingers == [1, 0]:

            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.line(img, (xp, yp), (x1, y1), (255, 0, 0), 5)

            cv2.line(canvas, (xp, yp), (x1, y1), (255, 0, 0), 5)

            xp, yp = x1, y1

        # Eraser Mode
        elif fingers == [1, 1]:

            cv2.circle(img, (x1, y1), 20, (0, 0, 0), cv2.FILLED)

            cv2.circle(canvas, (x1, y1), 20, (0, 0, 0), cv2.FILLED)

        else:

            xp, yp = 0, 0

    # Merge Drawing with Webcam
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, inv)

    img = cv2.bitwise_or(img, canvas)

    cv2.imshow("Virtual Canvas", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()