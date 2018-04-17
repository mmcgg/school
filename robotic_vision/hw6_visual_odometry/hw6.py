from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors

import numpy as np
import pygame
import cv2
from copy import deepcopy
import time
from collections import deque

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

def skew(d):
    return np.array([[0., -d[2], d[1]],
                     [d[2], 0., -d[0]],
                     [-d[1], d[0], 0.]])


# Initiate ORB detector
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Initialize Fast detector
fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

app = pg.mkQApp()
pw = pg.plot(title='x odom')
pwy = pg.plot(title='y odom')
pwz = pg.plot(title='z odom')

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 5,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))

p0 = np.zeros([50],dtype=np.float32)

# Setup
pygame.init()
 
# Set the width and height of the screen [width,height]
size = [250, 250]
screen = pygame.display.set_mode(size)
 
pygame.display.set_caption("My Game")

#env = Holodeck.make("UrbanCity")
env = Holodeck.make("EuropeanForest")
#env = Holodeck.make("RedwoodForest")

env.reset()

done=False
count = -1
data_count = 0

# deque to hold true positions
true_x = [None]*10000
true_y = [None]*10000
true_z = [None]*10000
est_x = [None]*10000
est_y = [None]*10000
est_z = [None]*10000
t_vec = [None]*10000

est_pos = np.zeros([3,1])
est_vel = np.zeros([3,1])
    
# Canyon follow
command = np.array([0, 0, 0, 35],dtype=np.float32)

current_R = np.eye(3)

while(done==False):

    state, reward, terminal, _ = env.step(command)
    
    # To access specific sensor data:
    frame = state[Sensors.PRIMARY_PLAYER_CAMERA]
    
    velocity = state[Sensors.VELOCITY_SENSOR]*.01
    velocity[0] = -velocity[0] # Otherwise this frame is left handed

    position = state[Sensors.LOCATION_SENSOR]*.01
    position[0] = -position[0] # Otherwise this frame is left handed

    orientation = state[Sensors.ORIENTATION_SENSOR]
    imu = state[Sensors.IMU_SENSOR]

    body_velocity = np.matmul(orientation,velocity)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame)
    
    if(count==-1):
        new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        est_pos = position
        
        est_x[0] = position[0,0]
        est_y[0] = position[1,0]
        est_z[0] = position[2,0]
        true_x[0] = position[0,0]
        true_y[0] = position[1,0]
        true_z[0] = position[2,0]
        t_vec[0] = data_count
        data_count += 1    
        
    elif(count==3):

        count = 0        
        
        old_gray = new_gray
        new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use optical flow
        # p0 = cv2.goodFeaturesToTrack(old_gray,50,.01,10)
        kp = fast.detect(old_gray,None)
        p0 = np.reshape(np.array([p.pt for p in kp],dtype=np.float32),[-1,1,2])
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

        # Select good points
        good_old = p0[st==1]
        good_new = p1[st==1]

        # draw the tracks
        # for i,(new,old) in enumerate(zip(good_new,good_old)):
            
        #     a,b = new.ravel()
        #     c,d = old.ravel()
        #     mask = cv2.line(mask, (a,b),(c,d), [0,0,255], 2)
    
        # img = cv2.add(frame,mask)
        

        # Use ORB
        # kp1, des1 = orb.detectAndCompute(old_gray,None)
        # kp2, des2 = orb.detectAndCompute(new_gray,None)

        # matches = bf.match(des1,des2)
        # matches = sorted(matches, key = lambda x:x.distance)

        # good_old = np.float32([ kp1[m.queryIdx].pt for m in matches[:50] ]).reshape(-1,1,2)
        # good_new = np.float32([ kp2[m.trainIdx].pt for m in matches[:50] ]).reshape(-1,1,2)

        # draw the matches
        # for i,(new,old) in enumerate(zip(good_new,good_old)):
            
        #     a,b = new.ravel()
        #     c,d = old.ravel()
        #     mask = cv2.line(mask, (a,b),(c,d), [0,0,255], 2)
    
        # img = cv2.add(frame,mask)
        

    

        # # Find the Essential Matrix
        [E,mask] = cv2.findEssentialMat(good_old,good_new,focal = 1.0,method=cv2.RANSAC,prob=.999,threshold=.1)
        
        [R1, R2, t] = cv2.decomposeEssentialMat(E)
        # [junk, R, t, mask] = cv2.recoverPose(E,good_new,good_old)

        if (np.trace(R1)>2.9):
            R = R1
        elif(np.trace(R2)>2.9):
            R = R2
        else:
            R = np.eye(3)

        current_R = R.dot(current_R)
        print(current_R)
        # print("\n",R,"\n",t)        

        # Check to see if essential matrix is good
        # goodness = 0
        # for i in range(len(good_new)):
        #     h_good_new = np.reshape(np.append(good_new[i],1),[1,3])
        #     h_good_old = np.reshape(np.append(good_old[i],1),[3,1])
        #     temp = np.dot(h_good_new,E)
        #     good = np.matmul(temp,h_good_old)
        #     goodness = goodness + abs(good)/len(good_new)


        
        # if(goodness > .5):
        if(0):            
            print("Bad essential matrix")
            pass
        else:

            # Find the Homography Matrix
            # [H,mask] = cv2.findHomography(good_old,good_new,method=cv2.RANSAC)
            # v,s,vt = np.linalg.svd(np.matmul(H.T,H))
            
            # u1 = np.reshape((np.sqrt(1-s[2]**2)*v[0,:] + np.sqrt(s[0]**2-1)*v[2,:])/np.sqrt(s[0]**2-s[2]**2),[3,])
            
            # u2 = np.reshape((np.sqrt(1-s[2]**2)*v[0,:] - np.sqrt(s[0]**2-1)*v[2,:])/np.sqrt(s[0]**2-s[2]**2),[3,])
            
            
            # U1 = np.array([v[:,1],u1,np.matmul(skew(v[:,1]),u1)])
            # U2 = np.array([v[:,1],u2,np.matmul(skew(v[:,1]),u2)])
            
            # W1 = np.array([np.matmul(H,v[:,1]), np.matmul(H,u1), np.matmul(skew(np.matmul(H,v[:,1])),np.matmul(H,u1))])
            # W2 = np.array([np.matmul(H,v[:,1]), np.matmul(H,u2), np.matmul(skew(np.matmul(H,v[:,1])),np.matmul(H,u2))])
            
            # R1 = np.matmul(W1,U1.T)
            # R2 = np.matmul(W2,U2.T)
            
            # t1 = np.matmul((H - R1),np.matmul(skew(v[:,1]),u1))
            # t2 = np.matmul((H - R2),np.matmul(skew(v[:,1]),u2))
            
            # N1 = np.matmul(skew(v[:,1]),u1)
            # N2 = np.matmul(skew(v[:,1]),u2)
            
            # e3 = np.array([[0],
            #                [0],
            #                [1]])
            
            # if(np.matmul(N1.T,e3)[0] > 0):
            #     t = np.reshape(t1,[3,1])
            # elif(np.matmul(N2.T,e3)[0] > 0):
            #     t = np.reshape(t2,[3,1])

            
            # Scale the whole translation vector
            t = t*np.linalg.norm(body_velocity)/np.linalg.norm(t)
            
            
            # Rotate t from camera frame to the quadrotor frame
            t = np.matmul(np.matrix([[0, 0, -1],
                                     [1, 0, 0],
                                     [0, -1, 0]]),t)
            
            # If the t vector is in the wrong direction
            # if((t[0,0]<0 and body_velocity[0,0]>0) or (t[0,0]>0 and body_velocity[0,0]<0)):
            
            #     # Switch the direction of t
            #     t[0,0] = -t[0,0]
            #     # t = -t            
            
            # if((t[1,0]<0 and body_velocity[1,0]>0) or (t[1,0]>0 and body_velocity[1,0]<0)):
            
            #     # Switch the direction of t
            #     t[1,0] = -t[1,0]
            
            # if((t[2,0]<0 and body_velocity[2,0]>0) or (t[2,0]>0 and body_velocity[2,0]<0)):
                
            #     # Switch the direction of t
            #     # t[2,0] = -t[2,0]
            #     t = -t                    
                
            est_vel = 0.5*est_vel + 0.5*t
            
            
            # print("\nR1: ",R1)
            # print("\nR2: ",R2)
            # print("\nest_vel: ",est_vel)
            # print("\nest_pos: ",est_pos)        
            # print("\nworld position: ",position)        
            # print("\nworld velocity: ",velocity)
            # print("\nbody velocity: ",body_velocity)
            
            
            
            # Add this body translation to my world estimate of myself
            # world_t = np.matmul(orientation.T,est_vel*.1)
            world_t = np.matmul(current_R,est_vel*.1)            
            # world_t = est_vel*.1
            world_t[0] = -world_t[0]
            est_pos = est_pos + world_t
            
            est_x[data_count] = est_pos[0,0]
            est_y[data_count] = est_pos[1,0]
            est_z[data_count] = est_pos[2,0]
            true_x[data_count] = position[0,0]
            true_y[data_count] = position[1,0]
            true_z[data_count] = position[2,0]
            t_vec[data_count] = data_count
            data_count += 1    
            

            if(data_count%25==0):
                
                pw.addItem(pg.PlotCurveItem(t_vec[0:data_count],true_x[0:data_count],pen=(255,0,0)))
                pw.addItem(pg.PlotCurveItem(t_vec[0:data_count],est_x[0:data_count],pen=(0,0,255)))
                
                pwy.addItem(pg.PlotCurveItem(t_vec[0:data_count],true_y[0:data_count],pen=(255,0,0)))
                pwy.addItem(pg.PlotCurveItem(t_vec[0:data_count],est_y[0:data_count],pen=(0,0,255)))
                # pwy.addItem(pg.PlotCurveItem(true_x[0:data_count],true_y[0:data_count],pen=(255,0,0)))
                # pwy.addItem(pg.PlotCurveItem(est_x[0:data_count],est_y[0:data_count],pen=(0,0,255)))
                
                pwz.addItem(pg.PlotCurveItem(t_vec[0:data_count],true_z[0:data_count],pen=(255,0,0)))
                pwz.addItem(pg.PlotCurveItem(t_vec[0:data_count],est_z[0:data_count],pen=(0,0,255)))
                app.processEvents()                

            
            # pw.plot(t_vec,true_x,pen=(255,0,0))
            # pw.plot(t_vec,true_y,pen=(0,255,0))
            # pw.plot(t_vec,true_z,pen=(0,0,255))
            
            # pw.plot(t_vec,est_x,pen=(255,100,0))
            # pw.plot(t_vec,est_y,pen=(100,255,0))
            # pw.plot(t_vec,est_z,pen=(0,100,255))        

        

    count += 1
    

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            # User pressed down on a key
 
        elif event.type == pygame.KEYDOWN:
        # Figure out if it was an arrow key. If so
            # adjust speed.
            if event.key == pygame.K_LEFT:
                command[2] = command[2] + .05
            elif event.key == pygame.K_RIGHT:
                command[2] = command[2] - .05                
            elif event.key == pygame.K_c:
                command[0] = .05
            elif event.key == pygame.K_v:
                command[0] = -.05                
            elif event.key == pygame.K_UP:
                command[1] = -.05
            elif event.key == pygame.K_DOWN:
                command[1] = .05
            elif event.key == pygame.K_e:
                command[3] = command[3] + .5  
            elif event.key == pygame.K_d:
                command[3] = command[3] - .5
            elif event.key == pygame.K_q:
                done=True
 
        # User let up on a key
        elif event.type == pygame.KEYUP:
            # If it is an arrow key, reset vector back to zero
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                pass
            elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                command[1] = 0
            elif event.key == pygame.K_c or event.key == pygame.K_v:
                command[0] = 0
