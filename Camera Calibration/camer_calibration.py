# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 18:06:51 2019

@author: Nirmal
"""



import cv2
import numpy as np
from matplotlib import pyplot as plt
from glob import glob


im_left = cv2.imread('1.jpg')
im_right = cv2.imread('5.jpg')

print(im_left.shape)
print(im_right.shape)

plt.subplot(121)
plt.imshow(im_left[...,::-1])
plt.subplot(122)
plt.imshow(im_right[...,::-1])
plt.show()

ret, corners = cv2.findChessboardCorners(im_left, (7,6))
print(corners.shape)
print(corners[0])

corners=corners.reshape(-1,2)
print(corners.shape)
print(corners[0])

im_left_vis=im_left.copy()
cv2.drawChessboardCorners(im_left_vis, (7,6), corners, ret) 
plt.imshow(im_left_vis)
plt.show()

x,y=np.meshgrid(range(7),range(6))
print("x:\n",x)
print("y:\n",y)


world_points=np.hstack((x.reshape(42,1),y.reshape(42,1),np.zeros((42,1)))).astype(np.float32)
print(world_points)

print(corners[0],'->',world_points[0])
print(corners[35],'->',world_points[35])

_3d_points=[]
_2d_points=[]

img_paths=glob('*.jpg') #get paths of all all images
for path in img_paths:
    im=cv2.imread(path)
    ret, corners = cv2.findChessboardCorners(im, (7,6))
    
    if ret: #add points only if checkerboard was correctly detected:
        _2d_points.append(corners) #append current 2D points
        _3d_points.append(world_points) #3D points are always the same


#print(_2d_points)
#print(_3d_points)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(_3d_points, _2d_points, (im.shape[1],im.shape[0]), None, None)
print("Ret:",ret)
print("Mtx:",mtx," ----------------------------------> [",mtx.shape,"]")
print("Dist:",dist," ----------> [",dist.shape,"]")
print("rvecs:",rvecs," --------------------------------------------------------> [",rvecs[0].shape,"]")
print("tvecs:",tvecs," -------------------------------------------------------> [",tvecs[0].shape,"]")

img = cv2.imread('6.jpg')
h, w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult1.png',dst)

#mean_error = 0
#tot_error = 0
#for i in range(len(_3d_points)):
#    imgpoints2, _ = cv2.projectPoints(_3d_points[i], rvecs[i], tvecs[i], mtx, dist)
#    error = cv2.norm(_2d_points[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
#    tot_error += error
#print("total error: ", mean_error/len(_3d_points))
#
#_3d_corners = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0], [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])
#
#image_index=4
#cube_corners_2d,_ = cv2.projectPoints(_3d_corners,rvecs[image_index],tvecs[image_index],mtx,dist) 
##the underscore allows to discard the second output parameter (see doc)
#
#print(cube_corners_2d.shape) #the output consists in 8 2-dimensional points
#
#img=cv2.imread(img_paths[image_index]) #load the correct image
#
#red=(0,0,255) #red (in BGR)
#blue=(255,0,0) #blue (in BGR)
#green=(0,255,0) #green (in BGR)
#line_width=5
#
##first draw the base in red
#cv2.line(img, tuple(cube_corners_2d[0][0]), tuple(cube_corners_2d[1][0]),red,line_width)
#cv2.line(img, tuple(cube_corners_2d[1][0]), tuple(cube_corners_2d[2][0]),red,line_width)
#cv2.line(img, tuple(cube_corners_2d[2][0]), tuple(cube_corners_2d[3][0]),red,line_width)
#cv2.line(img, tuple(cube_corners_2d[3][0]), tuple(cube_corners_2d[0][0]),red,line_width)
#
##now draw the pillars
#cv2.line(img, tuple(cube_corners_2d[0][0]), tuple(cube_corners_2d[4][0]),blue,line_width)
#cv2.line(img, tuple(cube_corners_2d[1][0]), tuple(cube_corners_2d[5][0]),blue,line_width)
#cv2.line(img, tuple(cube_corners_2d[2][0]), tuple(cube_corners_2d[6][0]),blue,line_width)
#cv2.line(img, tuple(cube_corners_2d[3][0]), tuple(cube_corners_2d[7][0]),blue,line_width)
#
##finally draw the top
#cv2.line(img, tuple(cube_corners_2d[4][0]), tuple(cube_corners_2d[5][0]),green,line_width)
#cv2.line(img, tuple(cube_corners_2d[5][0]), tuple(cube_corners_2d[6][0]),green,line_width)
#cv2.line(img, tuple(cube_corners_2d[6][0]), tuple(cube_corners_2d[7][0]),green,line_width)
#cv2.line(img, tuple(cube_corners_2d[7][0]), tuple(cube_corners_2d[4][0]),green,line_width)
#
##cv2.line(img, tuple(start_point), tuple(end_point),(0,0,255),3) #we set the color to red (in BGR) and line width to 3
#    
#plt.imshow(img[...,::-1])
#plt.show()

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

for fname in glob.glob('*.jpg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img,corners2,imgpts)
        cv2.imshow('img',img)
        k = cv2.waitKey(0) & 0xff
        if k == 's':
            cv2.imwrite(fname[:6]+'.png', img)
cv2.destroyAllWindows()



