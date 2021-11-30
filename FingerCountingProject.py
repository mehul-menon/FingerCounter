import cv2
import mediapipe as mp
import os
import HandTrackingModule as htm
import time


detector = htm.HandDetector()
PathToFolder = "FingerImages"
List = os.listdir(PathToFolder)
ImageList = []
for im in List:
    im = cv2.imread(f'{PathToFolder}/{im}')
    image = cv2.resize(im, (100, 100))
    ImageList.append(image)
pTime = 0
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img, draw=False)
    tip_id = [4, 8, 12, 16, 20]
    fingers = []
    if len(lmlist)!=0:
        if lmlist[tip_id[0]][1]>lmlist[tip_id[0]-2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if lmlist[tip_id[id]][2]<lmlist[tip_id[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        tot_fingers = fingers.count(1)
        #print(tot_fingers)
        h, w, c = ImageList[tot_fingers-1].shape
        H, W, C = img.shape
        img[H-h:H,W-w:W] = ImageList[tot_fingers-1]
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 2)
    cv2.imshow("frame", img)
    cv2.waitKey(1)