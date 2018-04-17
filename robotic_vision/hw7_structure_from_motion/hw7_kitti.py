import numpy as np
import cv2
import time
from matplotlib import pyplot as plt
import os

# 54 cm baseline between cameras
# This projection maps from 3D points to homogeneous pixel coordinates (u,v,1)^T for the left cam
P0 = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00],
               [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00],
               [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]])

P1 = np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02],
               [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00],
               [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]])


filepath = "../sequences/07"
image_names = os.listdir(filepath+"/image_0")
num_images = len(image_names)

image_names.sort()

# Initiate ORB detector
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

left_img_last = cv2.cvtColor(cv2.imread(filepath+"/image_0/"+image_names[0]), cv2.COLOR_BGR2GRAY)
right_img_last = cv2.cvtColor(cv2.imread(filepath+"/image_1/"+image_names[0]), cv2.COLOR_BGR2GRAY)

# Find the keypoints in the images and compute the descriptors
kp1, des1 = orb.detectAndCompute(left_img,None)
kp2, des2 = orb.detectAndCompute(right_img,None)
    
# Match descriptors
matches = bf.match(des1, des2)
    
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
N = int(len(matches)*.8)


for i in xrange(1,num_images):
        
    left_img = cv2.cvtColor(cv2.imread(filepath+"/image_0/"+image_names[i]), cv2.COLOR_BGR2GRAY)
    right_img = cv2.cvtColor(cv2.imread(filepath+"/image_1/"+image_names[i]), cv2.COLOR_BGR2GRAY)

    # Find the keypoints in the images and compute the descriptors
    kp1, des1 = orb.detectAndCompute(left_img,None)
    kp2, des2 = orb.detectAndCompute(right_img,None)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    N = int(len(matches)*.8)
    N = 3
    
    # Draw first N matches
    img3 = cv2.drawMatches(left_img,kp1,right_img,kp2,matches[:N],None,matchColor=(0,255,0), flags=2)

    # Make the keypoints into a numpy array of points
    good_left = np.float32([ kp1[m.queryIdx].pt for m in matches[:N] ]).reshape(-1,1,2)
    good_right = np.float32([ kp2[m.trainIdx].pt for m in matches[:N] ]).reshape(-1,1,2)

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(left_img,right_img)

    three_d_pts = cv2.triangulatePoints(P0,P1,good_left,good_right)

    for i in range(0,three_d_pts.shape[1]):
        three_d_pts[:,i] = three_d_pts[:,i]/three_d_pts[3,i]
    
    print(three_d_pts)


    cv2.imshow("matches",img3)
    cv2.imshow("disparity",disparity*-16.0/255)
    # cv2.imshow("right",right_img)

    cv2.waitKey(0)


