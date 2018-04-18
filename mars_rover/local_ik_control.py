#!/usr/bin/env python

import HalKinematics as hkin
import numpy as np

class ArmControl():
    def __init__(self):
        self.kin = hkin()
        self.delta_x = [0,0,0,0,0,0]
        
        #Subscriber for joystick commands
        rospy.Subscriber("/joy",joystick_msg, self.joystick_callback, tcp_nodelay=True)
        #Publishers for Pololu commands

        
    def joystick_callback(self,msg):
        #Establish which direction the user wants the EE to move in cartesian space
        if joystick_forward:
            self.delta_x = [0,1,0,0,0,0]
        elif joystick_backward:
            self.delta_x = [0,-1,0,0,0,0]
        elif joystick_up:
            self.delta_x = [0,0,1,0,0,0]
        elif joystick_down:
            self.delta_x = [0,0,-1,0,0,0]
        elif joystick_left:
            self.delta_x = [-1,0,0,0,0,0]
        elif joystick_right:
            self.delta_x = [1,0,0,0,0,0]
            
        #Establish the roll pitch and yaw change desired
        elif tilt_joystick_forward:
            self.delta_x = [0,0,0,1,0,0]
        elif tilt_joystick_backward:
            self.delta_x = [0,0,0,-1,0,0]
        elif tilt_joystick_up:
            self.delta_x = [0,0,0,0,1,0]
        elif tilt_joystick_down:
            self.delta_x = [0,0,0,0,-1,0]
        elif tilt_joystick_left:
            self.delta_x = [0,0,0,0,0,1]
        elif tilt_joystick_right:
            self.delta_x = [0,0,0,0,0,-1]
            
        #If the user doesn't want to move    
        else:
            delta_x = [0,0,0,0,0,0]
        
    def local_IK(self):
        #Find the change in joint angles that moves the EE in delta x direction
        J = self.kin.jacobian06(self.kin.jangles)
        J_t = np.pinv(J)
        delta_q = np.dot(J_t,self.delta_x)*.001

        #Command the new joint angles
        q = self.kin.jangles+delta_q

        #Publish the new commands to the pololus


if __name__=='__main__':
    control = ArmControl()
    rate = rospy.Rate(100)
    
    while not rospy.is_shutdown():
        control.local_IK()
        rate.sleep()

