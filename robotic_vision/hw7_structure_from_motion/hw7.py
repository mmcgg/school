import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

mymap = np.ones([100,100,3])*255
my_x = 0.0
my_y = 0.0
my_z = 0.0

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import sys

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

app = pg.mkQApp()
pw = pg.plot(title='x odom')


est_x = []
est_z = []
theta = 0
tz = 0


def obj_func(x, myargs):
    cost = 0
    
    theta = x[0]
    tz = x[1]    

    A = myargs['A']
    b = myargs['b']    

    T = np.array([[np.cos(theta), 0, np.sin(theta), 0],
                  [0, 1, 0, 0],
                  [-np.sin(theta), 0, np.cos(theta), tz],
                  [0, 0, 0, 1]])

    for k in xrange(0,len(A)):
        bhat = np.dot(T,A[k,:].T)
        error = 2*pow(np.linalg.norm(bhat - b[k,:].T),.5)
        cost = cost + error

    # if(abs(theta)>.1):
    #     cost = cost + pow(10*theta,2)

    if(tz<0):
        cost = cost + pow(10000*tz,2)

    cost = cost + pow(8*theta,2) + 0.1*pow(tz,2)
    return cost


filepath = "../sequences/07/"

current_T = np.eye(4)

# From calibration for 07
P0 = np.array([[7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02, 0.000000000000e+00],
               [0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 0.000000000000e+00],
               [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]])

P1 = np.array([[7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02, -3.798145000000e+02],
               [0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02, 0.000000000000e+00],
               [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]])

f = 7.070912000000e+02
B = 3.798145000000e+02/f


stereo = cv2.StereoBM_create(numDisparities=16*20, blockSize=15)

image_names = os.listdir(filepath+"image_0/")
image_names.sort()
num_images = len(image_names)


# Initiate ORB detector
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

current_R = np.eye(3)


for i in range(0,39):
    # Read in images
    img1 = cv2.imread('city_holodeck/img_'+str(i)+'.jpg')
    img2 = cv2.imread('city_holodeck/img_'+str(i+1)+'.jpg')

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Find the keypoints in the images and compute the descriptors
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    N = int(len(matches)*.8)
    
    # Draw first N matches
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:N],None,matchColor=(0,255,0), flags=2)

    # Make the keypoints into a numpy array of points
    good_old = np.float32([ kp1[m.queryIdx].pt for m in matches[:N] ]).reshape(-1,1,2)
    good_new = np.float32([ kp2[m.trainIdx].pt for m in matches[:N] ]).reshape(-1,1,2)

    
    
    [E,mask] = cv2.findEssentialMat(good_old,good_new,method=cv2.RANSAC,threshold=.1,prob=.999)

    # Make a calibration matrix based on a focal length of one
    f = 1.0
    pp = np.array([img1.shape[0]/2,img1.shape[1]/2])
    K = np.array([[f, 0.0, pp[0]],
                  [0, f, pp[1]],
                  [0, 0, 1]])
    K = np.eye(3)

    # Get the pose from the Essential Matrix
    [junk,R,t,mask,junk2] = cv2.recoverPose(E,good_new,good_old,cameraMatrix=K,distanceThresh=100000.0)

    
    current_R = np.dot(current_R,R.T)
    my_x = my_x + t[0]
    my_y = my_y + t[1]
    my_z = my_z + t[2]

    current_pos = np.array([my_x, my_y, my_z]).T
    
    print("R:\n",current_R)
    print("t:\n",current_pos)

    plt.imshow(img3),plt.show()

    

left_img_last = cv2.cvtColor(cv2.imread(filepath+"image_0/"+image_names[0]),cv2.COLOR_BGR2GRAY)
right_img_last = cv2.cvtColor(cv2.imread(filepath+"image_1/"+image_names[0]),cv2.COLOR_BGR2GRAY)
[height, width] = left_img_last.shape

Q = np.array([[1, 0, 0, -width/2],
              [0, 1, 0, -height/2],
              [0, 0, 0, -f],
              [0, 0, -1/B, 0]])


kp1_last, des1_last = orb.detectAndCompute(left_img_last,None)
disparity_last = stereo.compute(left_img_last,right_img_last)
three_d_image_last = cv2.reprojectImageTo3D(disparity_last,Q)



for i in xrange(1,num_images):

    if i==345:
        i += 1

    # print "Reading in images..."
    # Read in the new images
    if(i>1):
        left_img_last = left_img
    left_img = cv2.cvtColor(cv2.imread(filepath+"image_0/"+image_names[i]),cv2.COLOR_BGR2GRAY)
    right_img = cv2.cvtColor(cv2.imread(filepath+"image_1/"+image_names[i]),cv2.COLOR_BGR2GRAY)

    # Find keypoints and descriptors in the new left image
    if(i>1):
        kp1_last = kp1
        des1_last = des1    
    kp1, des1 = orb.detectAndCompute(left_img,None)

    # print "Computing disparity..."    
    # Compute disparity using left and right images
    if(i>1):
        disparity_last = disparity
    disparity = stereo.compute(left_img,right_img)

    # print "Matching..."    
    # Find temporal matches between the left and left_last images
    temporal_matches = bf.match(des1,des1_last)
    temporal_matches = sorted(temporal_matches, key = lambda x:x.distance)

    # Keep the pixel space points of the best 75%
    N = int(len(temporal_matches)*.6)
    # print N

    # print "Getting good points..."        
    good_left_last = np.float32([ kp1_last[m.queryIdx].pt for m in temporal_matches[:N] ]).reshape(-1,1,2)    
    good_left = np.float32([ kp1[m.queryIdx].pt for m in temporal_matches[:N] ]).reshape(-1,1,2)


    # print "Reprojecting to 3D..."            
    # At this point we have good_left_last and good_left
    if(i>1):
        three_d_image_last = three_d_image
    three_d_image = cv2.reprojectImageTo3D(disparity,Q)

    # print "Getting 3D points for matched pixels..."                

    # Calculate the 3d position of the pixels in good_left_last
    good_left_positions_last = np.zeros([N,3])
    for j in xrange(0,N):
        # Get the pixel coordinates of this point
        u = int(good_left_last[j,0,0])
        v = int(good_left_last[j,0,1])
        # Get the 3D position at this pixel
        good_left_positions_last[j,:] = three_d_image_last[v,u,:]
    
    # Calculate the 3d position of the pixels in good_left
    good_left_positions = np.zeros([N,3])
    for j in xrange(0,N):
        # Get the pixel coordinates of this point
        u = int(good_left[j,0,0])
        v = int(good_left[j,0,1])
        # Get the 3D position at this pixel
        good_left_positions[j,:] = three_d_image[v,u,:]

        
    A = np.hstack((good_left_positions_last,np.ones((N,1))))
    b = np.hstack((good_left_positions,np.ones((N,1))))
    
    # Do Least Squares to get the best R and T (isn't that good)    
    # x = np.dot(np.linalg.pinv(A),b)

    # Use a scipy minimizer to find R and T
    x0 = [theta,tz] # Theta (about y), tx, and tz
    res = minimize(obj_func, x0, method='BFGS', args={'A':A,'b':b},
               options={'disp':False})
    theta = 0.5*theta + 0.5*res.x[0]
    tz = 0.5*tz + 0.5*res.x[1]
    ty = 0
    tx = 0

    T = np.array([[np.cos(theta), 0, np.sin(theta), tx],
                  [0, 1, 0, ty],
                  [-np.sin(theta), 0, np.cos(theta), tz],
                  [0, 0, 0, 1]])

    if(not np.isnan(T[0,0])):

        current_T = np.dot(current_T,T)
        
        # print "current_T: \n",current_T

        est_x.append(current_T[0,3])
        est_z.append(current_T[2,3])
        
        pts_x = []
        pts_z = []
        for j in xrange(0,width/10):
            if(three_d_image[height/2-100,j*10,2]>0 and np.linalg.norm(three_d_image[height/2-150,j*10,:])<10):            
                pt = np.dot(current_T[0:3,0:3],three_d_image[height/2-150,j*10,:].T)

                pts_x.append(current_T[0,3]+pt[0])
                pts_z.append(current_T[2,3]+pt[2])
        

        pw.addItem(pg.PlotCurveItem(est_x,est_z,pen=(255,0,0)))
        pw.addItem(pg.ScatterPlotItem(pts_x,pts_z,size=.5,pen=(0,0,255)))
        pw.setAspectLocked(True)
        app.processEvents()                        
        
        
    # movements = np.zeros([N,3,1])
    # for i in xrange(0,N):
    #     movements[i,:,:] = good_left_positions[i,:,:]-good_left_positions_last[i,:,:]
    #     print movements[i,:,:]

        
    # cv2.imshow('disparity',disparity*1.0)        
    # Visualize the temporal matches
    # img4 = cv2.drawMatches(left_img,kp1,left_img_last,kp1_last,temporal_matches[:N], None, flags=2)
    cv2.imshow("Image",left_img)
    # cv2.waitKey(0)

    
