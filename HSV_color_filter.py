"""
Created on Mon Aug 19 10:37:37 2019

@author: Nirmal
"""

#import cv2
#import numpy as np
#
#img = cv2.imread("img2.bmp")
#Image2 = np.array(img, copy=True)
##cv2.imshow('original image',img)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
#i=1
#Image2 = np.array(img, copy=True)
#row = img.shape[0]
#col = img.shape[1]
#for r in range(0,row):
#    for c in range(0,col):
#        if img[r,c,1]!=0:
#            px = img[r][c]
#    #        print("count...",i)
#    #        print(px)
#            i=i+1
#            if (px[1] > 75):
#                Image2[r][c] = ([0,0,0])
#    #            print("new value.....",Image2[r][c])
#            if all(px < ([50,50,50])):
#                px = ([0,0,0])
#    #            print("new value.....",Image2[r][c])
#
##            
##
##cv2.imshow("show", hsv_channels[0])
##cv2.imshow("show", hsv_channels[2])
#cv2.imshow("show2", Image2)
##cv2.imwrite("img2.bmp",Image2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

import cv2
import numpy as np

image_hsv = None   # global ;(
pixel = (20,60,80) # some stupid default

# mouse callback function
def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]

        #you might want to adjust the ranges(+-10, etc):
        lower = np.array([65,60,60])
        upper = np.array([80,255,255])
        print(pixel, lower, upper)

        image_mask = cv2.inRange(image_hsv,lower,upper)
        cv2.imshow("mask",image_mask)

def main():
    import sys
    global image_hsv, pixel # so we can use it in mouse callback

    image_src = cv2.imread('img1.bmp')  # pick.py my.png
    if image_src is None:
        print ("the image read is None............")
        return
    cv2.imshow("bgr",image_src)

    ## NEW ##
    cv2.namedWindow('hsv')
    cv2.setMouseCallback('hsv', pick_color)

    # now click into the hsv img , and look at values:
    image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv",image_hsv)
#    cv2.imwrite("img3.bmp",image_hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()

