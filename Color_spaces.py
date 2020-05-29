# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:59:41 2019

@author: Nirmal
"""

import cv2
import numpy as np

frame = cv2.imread("a0.jpg")
HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
YCB = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
LAB = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

#python
bgr = [214.28347298983775, 218.3708771617044, 215.17871575444227]
thresh = 30
minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
 
maskBGR = cv2.inRange(frame,minBGR,maxBGR)
resultBGR = cv2.bitwise_and(frame, frame, mask = maskBGR)
 
#convert 1D array to 3D, then convert it to HSV and take the first element 
# this will be same as shown in the above figure [65, 229, 158]
hsv = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2HSV)[0][0]
 
minHSV = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
maxHSV = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])
 
maskHSV = cv2.inRange(HSV, minHSV, maxHSV)
resultHSV = cv2.bitwise_and(HSV, HSV, mask = maskHSV)
 
#convert 1D array to 3D, then convert it to YCrCb and take the first element 
ycb = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2YCrCb)[0][0]
 
minYCB = np.array([ycb[0] - thresh, ycb[1] - thresh, ycb[2] - thresh])
maxYCB = np.array([ycb[0] + thresh, ycb[1] + thresh, ycb[2] + thresh]) 
#minYCB = np.array([62,46,71])
#maxYCB = np.array([159,88,129])
 
maskYCB = cv2.inRange(YCB, minYCB, maxYCB)
resultYCB = cv2.bitwise_and(YCB, YCB, mask = maskYCB)
 
#convert 1D array to 3D, then convert it to LAB and take the first element 
lab = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2LAB)[0][0]
 
minLAB = np.array([lab[0] - thresh, lab[1] - thresh, lab[2] - thresh])
maxLAB = np.array([lab[0] + thresh, lab[1] + thresh, lab[2] + thresh])
#minLAB = np.array([111,49,137])
#maxLAB = np.array([174,114,207])
 
maskLAB = cv2.inRange(LAB, minLAB, maxLAB)
resultLAB = cv2.bitwise_and(LAB, LAB, mask = maskLAB)
 
#cv2.imshow("Result BGR", resultBGR)
cv2.imshow("Result HSV", resultHSV)
cv2.imwrite('img10.bmp',resultHSV)
cv2.imshow("Result YCB", resultYCB)
cv2.imshow("Output LAB", resultLAB)
cv2.waitKey(0)
cv2.destroyAllWindows()
