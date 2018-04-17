import numpy as np

from Holodeck import Holodeck, Agents
from Holodeck.Environments import HolodeckEnvironment
from Holodeck.Sensors import Sensors
 
import pygame
import cv2

from collections import deque
import matplotlib.pyplot as plt
import scipy.io as sio
from copy import deepcopy

positions = deque()
orientations = deque()
velocities = deque()

# Setup
pygame.init()
 
# Set the width and height of the screen [width,height]
size = [700, 500]
screen = pygame.display.set_mode(size)
 
pygame.display.set_caption("My Game")

env = Holodeck.make("UrbanCity")

env.reset()

# This command tells the UAV to not roll or pitch, but to constantly yaw left at 10m altitude.
command = np.array([0, 0, 0, 20],dtype=np.float32)
done=False
while(done==False):
    state, reward, terminal, _ = env.step(command)
    
    # To access specific sensor data:
    frame = state[Sensors.PRIMARY_PLAYER_CAMERA]
    velocity = state[Sensors.VELOCITY_SENSOR]
    position = state[Sensors.LOCATION_SENSOR]
    orientation = state[Sensors.ORIENTATION_SENSOR]
    imu = state[Sensors.IMU_SENSOR]

    positions.append(deepcopy(position))
    orientations.append(deepcopy(orientation))
    velocities.append(deepcopy(velocity))

        
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv2.Canny(frame, 150, 300)

    # Display the canny edge detection image
    cv2.imshow('Canny', edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        

    #frame = gray

    kernel = np.array([[1,0,-1],
                       [1,0,-1],
                       [1,0,-1]])
    dst_hor1 = cv2.filter2D(frame,-1,kernel)
    dst_hor2 = cv2.filter2D(frame,-1,kernel*-1.0)        
    
    kernel = np.array([[1,1,1],
                       [0,0,0],
                       [-1,-1,-1]])
    dst_vert1 = cv2.filter2D(frame,-1,kernel)
    dst_vert2 = cv2.filter2D(frame,-1,kernel*-1.0)        
    
    dst = dst_hor1 + dst_hor2 + dst_vert1 + dst_vert2
    
    #ret,dst = cv2.threshold(dst,100,255,cv2.THRESH_BINARY)
    
    # Display the custom kernel filtered image
    cv2.imshow('Custom Filter', dst)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
            # User pressed down on a key
 
        elif event.type == pygame.KEYDOWN:
        # Figure out if it was an arrow key. If so
            # adjust speed.
            if event.key == pygame.K_LEFT:
                command[2] = command[2] + .25
            elif event.key == pygame.K_RIGHT:
                command[2] = command[2] - .25                
            elif event.key == pygame.K_UP:
                command[1] = -1
            elif event.key == pygame.K_DOWN:
                command[1] = 1
            elif event.key == pygame.K_e:
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




print("squeezing")
positions = np.squeeze(np.array(positions))
orientations = np.squeeze(np.array(orientations))
velocities = np.squeeze(np.array(velocities))
print(positions.shape)
print(orientations.shape)
print(velocities.shape)

print("dictionary-izing")
data = {'positions':positions,
        'orientations':orientations,
        'velocities':velocities}

print("saving")
sio.savemat('data', data)

print("plotting")
plt.ion()
plt.figure(1)
plt.plot(positions)
plt.title("Position")
plt.legend(["x","y","z"])
plt.show()

# plt.figure(2)
# plt.plot(orientations)
# plt.show()

plt.ioff()
plt.figure(3)
plt.plot(velocities)
plt.title("Velocity")
plt.legend(["x","y","z"])
plt.show()

