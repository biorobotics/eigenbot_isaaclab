#!/usr/bin/env python3

import numpy as np
import rospy
import rosnode
from sensor_msgs.msg import JointState


def wrap_to_pi(angle):
    return np.remainder(angle + np.pi,  np.pi*2) - np.pi


class EigenbotJointPub():
    def __init__(self):
        # self.initialized = False
        self.initialized = True
        self.num_joints = 3
        self.joint_positions = [float('nan')]*self.num_joints
        self.initial_joint_positions = np.zeros(self.num_joints)
        self.rate = rospy.Rate(10)
        # self.joint_cmd_pub = rospy.Publisher('/eigenbot/joint_cmd', JointState, queue_size=1)
        self.joint_cmd_pub = rospy.Publisher('joint_states', JointState, queue_size=1)
        self.joint_fb_sub = rospy.Subscriber('/eigenbot/joint_fb', JointState, self.joint_fb_callback)
    

    def main_loop(self):
        # parameters for alternating tripad
        amplitudes = np.tile(np.array([np.pi/8, 3*np.pi/16, np.pi/4])[:,None], (1,6))
        # const_offsets = np.array([[0,0,0,0,0,0], 
        #     [1,1,1,1,1,1], [0,0,0,0,0,0]])*amplitudes
        const_offsets = np.array([
            [-1,1,-1,1,-1,1],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]
        ])
        phase_offsets = np.array([
            [np.pi/2,-np.pi/2,np.pi/2,-np.pi/2,np.pi/2,-np.pi/2],
            [0,np.pi,0,np.pi,0,np.pi],
            [0,np.pi,0,np.pi,0,np.pi]
        ])
        # for test without 90deg module in hexapod
        # const_offsets[1,:] -= np.pi/4
        # const_offsets[2,:] += np.pi/4
        t = 0
        dt = np.pi/15
        while not rospy.is_shutdown():
            if not self.initialized:
                self.rate.sleep()
                continue

            # Create Joint State
            joint_state = JointState()
            joint_state.header.stamp = rospy.Time.now()
            
            for i in range(self.num_joints):
                joint_state.name.append('bendy_input_M{}_S{}'.format(i+1,i+1))

            # Assume we are using positional control
            joint_state.velocity = [float('nan')]*self.num_joints
            joint_state.effort = [float('nan')]*self.num_joints
            joint_state.position  = np.copy(self.initial_joint_positions)

            # Set joint positions
            for i in range(self.num_joints):
                joint_i = i//6
                leg_i = i%6
                joint_state.position[i] = amplitudes[joint_i,leg_i]*np.sin(t + phase_offsets[joint_i,leg_i]) # + const_offsets[joint_i, leg_i]
                if joint_i >= 1:
                    joint_state.position[i] = max(0, joint_state.position[i])
                joint_state.position[i] += self.initial_joint_positions[i]
            # print(joint_state.position[6]*180/np.pi)
            # self.joint_positions = [pos + 0.01 for pos in self.joint_positions]
            self.joint_cmd_pub.publish(joint_state)

            t += dt
            self.rate.sleep()


    def joint_fb_callback(self, msg):
        if not self.initialized:
            # Initialize current joint positions from feedback
            self.initial_joint_positions = np.array(msg.position)
            self.initialized = True


if __name__ == '__main__':
    rospy.init_node('eigenbot_joint_pub')
    eigenbot_joint_pub = EigenbotJointPub()
    eigenbot_joint_pub.main_loop()    
