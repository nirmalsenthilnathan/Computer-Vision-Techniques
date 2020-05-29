# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:42:03 2019

@author: Nirmal
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_float

img = cv2.imread('5.bmp')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for x1,y1,x2,y2 in lines[0]:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
cv2.imwrite('houghlines2.jpg',img)
cv2.imshow('BG', img)
cv2.waitKey(5000)
cv2.destroyAllWindows()