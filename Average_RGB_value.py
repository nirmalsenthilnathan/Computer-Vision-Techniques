# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:02:50 2019

@author: Nirmal
"""

#   iterate through each pixel in an image and
#   determine the average rgb color

# you will need to install the PIL module

import cv2
from PIL import Image
import numpy as np

def getAverageRGBN(image):
  """
  Given PIL Image, return average value of color as (r, g, b)
  """
  # get image as numpy array
  im = np.array(image)
  # get shape
  w,h,d = im.shape
  # change shape
  im.shape = (w*h, d)
  # get average
  return tuple(im.mean(axis=0))

if __name__ == '__main__':
  # assumes you have a test.jpg in the working directory! 
  pc = getAverageRGBN(cv2.imread('a0.jpg'))
  print("(red, green, blue, total_pixel_count)")
  print(pc)
#from scipy import misc
#print(misc.imread('img.bmp').mean(axis=(0,1)))

# for my picture the ouput rgb values are:
#   (red, green, blue, total_pixel_count)
#   (135, 122, 107, 10077696)
#
# you can see that my image had 10,077,696 pixels and python/PIL
#   still churned right through it!