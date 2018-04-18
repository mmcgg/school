#!/usr/bin/env python
import numpy as np
import hal_arm_kinematics as hal_arm_kinematics

class HalKinematics(hal_arm_kinematics):
    def __init__(self):
        self.dh = [1,2,3,4,
                   4,5,6,7,
                   5,4,6,4]

        self.q0 = 0
        self.q1 = 0
        self.q2 = 0
        self.q3 = 0
        self.q4 = 0
        self.q5 = 0
        self.jangles = [self.q0,self.q1,self.q2,self.q3,self.q4,self.q5]
        
        rospy.init_node('Hal_FK')

        rospy.Subscriber("/joint_angles",joint_angles_msg, self.joint_angles_callback, tcp_nodelay=True)

    def joint_angles_callback(self,msg):
        data = msg
        self.q0 = msg[0]
        self.q1 = msg[1]
        self.q2 = msg[2]
        self.q3 = msg[3]
        self.q4 = msg[4]
        self.q5 = msg[5]
        self.jangles = [self.q0,self.q1,self.q2,self.q3,self.q4,self.q5]

    def get_joint_angles(self):
        jt_angles = [self.q0,self.q1,self.q2,self.q3,self.q4,self.q5]
        return jt_angles
            

            
