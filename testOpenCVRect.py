import cv2
import numpy as np

img = cv2.imread('./test1_images/WIN_20220926_20_04_23_Pro.jpg')
h, l, d = img.shape

minArea = int(h * l * 0.15) 

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
new_cnts = []
for cnt in cnts:
    approx = cv2.contourArea(cnt)
    if approx > minArea:
        new_cnts.append(cnt)
    #print(approx)

cv2.drawContours(img, new_cnts, -1, (0,255,0), 3)
img = cv2.resize(img, (int(l/2), int(h/2)))
cv2.imshow('image', img)
#cv2.imshow('Binary',thresh_img)
cv2.waitKey()