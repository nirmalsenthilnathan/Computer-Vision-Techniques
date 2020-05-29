# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 12:27:00 2020

@author: Nirmal
"""


import cv2
import numpy as np

# Import relevant libraries
image = cv2.imread("a0.jpg")

# convert to gray and binarize
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

 # note: erosion and dilation works on white forground
binary_img = cv2.bitwise_not(binary_img)

 # dilate the image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
dilated_img = cv2.morphologyEx(binary_img, cv2.MORPH_DILATE, kernel,iterations=1)
    
# find contours, discard contours which do not belong to a rectangle
(cnts, _) = cv2.findContours(dilated_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
sq_cnts = []  # contours of interest to us

mask = np.full(image.shape[:2], 0, dtype=np.uint8)

for cnt in cnts:
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        print("Area: " + str(area))
        if area > 500 and area < 3000:
            sq_cnts.append(cnt)
            img_cnt = cv2.drawContours(mask, [cnt], -1, (255,255,255), 1)
            


result = cv2.bitwise_and(image,image,mask = mask)
#
cv2.imshow("Output LAB", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("result.jpg", result)  