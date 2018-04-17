from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors

import pygame
import numpy as np
import cv2
from matplotlib import pyplot as plt

mymap = np.ones([500,500,3])*255
my_x = 0.0
my_y = 0.0
my_z = 0.0

# Initiate ORB detector
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

current_R = np.eye(3)
current_t = np.zeros([3,1])

true_R = np.eye(3)
true_t = np.zeros([3,1])

def map_points(map_pts):
    for i in range(0,len(map_pts)):
        x = map_pts[i][0]
        z = map_pts[i][2]

        mymap[250+5*int(x),250+5*int(z)] -= 1

def get_commands():
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
                command[1] = -.1
            elif event.key == pygame.K_DOWN:
                command[1] = .1
            elif event.key == pygame.K_e:
                command[3] = command[3] + .5  
            elif event.key == pygame.K_d:
                command[3] = command[3] - .5
            elif event.key == pygame.K_z:
                cv2.imwrite("img_"+str(img_count)+".jpg",frame)
                img_count = img_count + 1
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


# Setup Holodeck
pygame.init()
screen = pygame.display.set_mode([250,250])
pygame.display.set_caption("My Game")
env = Holodeck.make("UrbanCity")
env.reset()
command = np.array([0, 0, 0, 5],dtype=np.float32)

done=False
first_frame=True
while(done==False):

    if(first_frame==True):
        state, reward, terminal, _ = env.step(command)
        frame = state[Sensors.PRIMARY_PLAYER_CAMERA]
        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        first_frame=False


    for i in range(0,30):
        get_commands()
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

    new_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    img1 = old_gray
    img2 = new_gray

    # Find the keypoints in the images and compute the descriptors
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    
    N = int(len(matches)*.1)
    
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

    P1 = np.hstack((current_R,current_t))

    P2 = np.hstack((R,t))

    three_d_pts = cv2.triangulatePoints(P1,P2,good_new,good_old)
    print(three_d_pts)

    map_pts = []
    for i in range(0,three_d_pts.shape[1]):
        three_d_pts[:,i] = three_d_pts[:,i]/three_d_pts[3,i]

        if(three_d_pts[1,i]>4 and three_d_pts[1,i]<6):
            map_pts.append(three_d_pts[:,i])
            # print(three_d_pts[:,i])


    map_points(map_pts)
    
    # current_R = np.dot(current_R,R)
    # print(current_R)
    
    # current_t += t*np.linalg.norm(body_velocity)
    # print(current_t)

    # plt.ion()
    # plt.imshow(img3)
    # plt.show()
    # plt.pause(1)

    plt.ion()    
    plt.imshow(mymap)
    plt.show()
    plt.pause(1)
    
    
    old_gray = new_gray
    


