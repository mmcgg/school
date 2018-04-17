import numpy as np

from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors
 
import pygame
import cv2

from copy import deepcopy

import time

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

grid_height = 25
grid_width = 25
p0 = np.zeros([grid_height*grid_width,1,2],dtype=np.float32)

k = 0
for i in range(0,grid_width):
    for j in range(0,grid_height):
        p0[k] = [512/grid_width*.5 + 512/grid_width*i,512/grid_height*.5 + 512/grid_height*j]
        k+=1

# Setup
pygame.init()
 
# Set the width and height of the screen [width,height]
size = [250, 250]
screen = pygame.display.set_mode(size)
 
pygame.display.set_caption("My Game")

#env = Holodeck.make("UrbanCity")
# env = Holodeck.make("EuropeanForest")
env = Holodeck.make("RedwoodForest")

env.reset()

# This command tells the UAV to not roll or pitch, but to constantly yaw left at 10m altitude.
command = np.array([0, 0, 0, .3],dtype=np.float32)
done=False
count = -1
start = time.time()
yaw = 0
right_flow = 0
right_flowy = 0
right_pts = 0
left_flow = 0
left_flowy = 0
left_pts = 0
center_flow = 0
bottom_flow = 0
center_pts = 0
yaw_desired = 1.1
    
# Canyon follow
command = np.array([0, 0, 0, 1],dtype=np.float32)
while(done==False):

    state, reward, terminal, _ = env.step(command)
    
    # To access specific sensor data:
    frame = state[Sensors.PRIMARY_PLAYER_CAMERA]
    velocity = state[Sensors.VELOCITY_SENSOR]
    position = state[Sensors.LOCATION_SENSOR]
    orientation = state[Sensors.ORIENTATION_SENSOR]
    imu = state[Sensors.IMU_SENSOR]

    theta = -np.arcsin(orientation[0,2])
    if(np.isnan(np.arcsin(orientation[1,0]/np.cos(theta))) != 1):
        yawrate = (np.arcsin(orientation[1,0]/np.cos(theta)) - yaw)
        yaw = np.arcsin(orientation[1,0]/np.cos(theta))
    body_velocity = np.dot(orientation,velocity)

    # print("yawrate: ",yawrate)
    # print("vx: ",body_velocity[0])
    # print("vy: ",body_velocity[1])
    # print("vz: ",body_velocity[2])    
    command[2] = 5.0*(yaw_desired-yaw) - 3.0*yawrate

    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame)
    
    if(count==-1):
        new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    elif(count==2):
        
        old_gray = new_gray
        new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
    
        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), [0,0,255], 2)
    
        img = cv2.add(frame,mask)
        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        count = 0

        # Find optical flow in the right half
        this_right_flow = 0
        this_left_flow = 0
        this_right_flowy = 0
        this_left_flowy = 0
        this_center_flow = 0
        this_bottom_flow = 0                
        roll_right_pts = 0
        roll_left_pts = 0
        yaw_right_pts = 0
        yaw_left_pts = 0
        center_pts = 0
        bottom_pts = 0                
        
        for i in range(0,good_old.shape[0]):

            # Roll part of optical flow
            if (good_new[i][0]>256+100 and good_new[i][1]>100 and good_new[i][1]<512-100):
                this_right_flow = this_right_flow + good_new[i][0]-good_old[i][0]
                roll_right_pts = roll_right_pts + 1
            elif (good_new[i][0]<256-100 and good_new[i][1]>100 and good_new[i][1]<512-100):
                this_left_flow = this_left_flow + good_new[i][0]-good_old[i][0]
                roll_left_pts = roll_left_pts + 1
                
            # Yaw part of optical flow
            if (good_new[i][0]>256+0 and good_new[i][1]>0 and good_new[i][1]<512-0):
                this_right_flowy = this_right_flowy + abs(good_new[i][1]-good_old[i][1])
                yaw_right_pts = yaw_right_pts + 1
            elif (good_new[i][0]<256-0 and good_new[i][1]>0 and good_new[i][1]<512-0):
                this_left_flowy = this_left_flowy + abs(good_new[i][1]-good_old[i][1])
                yaw_left_pts = yaw_left_pts + 1

            # Center part of optical flow
            if (good_new[i][0]>256 and good_new[i][0]<256+25 and good_new[i][1]>256-25 and good_new[i][1]<256+50):
                center_flow = np.linalg.norm(good_new[i]-good_old[i])
                center_pts = center_pts + 1

            # Bottom part of optical flow
            if (good_new[i][1]<256-200):
                bottom_flow = np.linalg.norm(good_new[i]-good_old[i])
                bottom_pts = bottom_pts + 1
                
                
        right_flow = 0.5*right_flow + 0.5*this_right_flow/roll_right_pts + 0.01*body_velocity[1] - 1.0*yawrate
        left_flow = 0.5*left_flow + 0.5*this_left_flow/roll_left_pts + 0.01*body_velocity[1] - 1.0*yawrate

        right_flowy = 0.5*right_flowy + 0.5*this_right_flowy/yaw_right_pts
        left_flowy = 0.5*left_flowy + 0.5*this_left_flowy/yaw_left_pts

        if(center_pts>0):
            center_flow = 0.5*center_flow + 0.5*this_center_flow/center_pts

        bottom_flow = 0.5*bottom_flow + 0.5*this_bottom_flow/bottom_pts            

        
        

        
    if(time.time()-start>10.0):
        command[1] = -.15
        # print("\nright + left: ",right_flow+left_flow)
        # print("right: ",right_flow)
        # print("left: ",left_flow)
        # print("righty: ",this_right_flowy)
        # print("lefty: ",this_left_flowy)


        # Canyon following (roll)
        if(abs(right_flow+left_flow)>1.0):
            command[0] = 0.05*(right_flow+left_flow)
        else:
            command[0] = 0
            
        if command[0] > .2:
            command[0] = .2
        elif command[0] < -.2:
            command[0] = -.2
            
        print("Roll Command: ",command[0])

        # Obstacle avoidance (yaw)
        command[2] = 5.0*(yaw_desired-yaw) - 1.0*yawrate
        if(abs(-left_flowy+right_flowy)>1.0):
            command[2] = .2*(-left_flowy+right_flowy)
            
        if command[2] > 5:
            command[2] = 5
        elif command[2] < -5:
            command[2] = -5
            
        print("Yaw command: ",command[2])

        # Obstacle avoidance (center)
        if(body_velocity[0]>3.0 and center_flow>10):
            command[1] = 1.5
            print("STOP!!!!!!")

        # Height maintenance (bottom)
        command[3] = 0.01*(.005*body_velocity[0]/bottom_flow) + 0.99*command[3]

        print("Velocity over flow: ",body_velocity[0]/bottom_flow)

        # print("Height command: ",command[3])

        print("\nyaw: ",yaw)        
        
    count += 1
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            # User pressed down on a key
 
        elif event.type == pygame.KEYDOWN:
        # Figure out if it was an arrow key. If so
            if event.key == pygame.K_e:
                command[3] = command[3] + 1                
            elif event.key == pygame.K_d:
                command[3] = command[3] - 1
            elif event.key == pygame.K_q:
                done=True
 
        # User let up on a key
        elif event.type == pygame.KEYUP:
            # If it is an arrow key, reset vector back to zero
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                pass
            elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                command[1] = 0

