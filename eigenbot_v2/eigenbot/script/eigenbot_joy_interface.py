#!/usr/bin/env python3

'''
eigenbot_joy_interface.py

Subscribes to joystick commands and produces a
desired velocity command
'''

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy


class EigenbotJoyInterface():
    def __init__(self):
        # Publishers and subscribers
        self._joy_sub = rospy.Subscriber('/joy', Joy, self._joy_callback)
        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Parameters
        # Default parameters are for Logitech F710 controller
        self._axis_linear = rospy.get_param('axis_linear', 5)
        self._axis_angular = rospy.get_param('axis_angular', 4)
        self._linear_vel_scale = rospy.get_param('linear_vel_scale', 1.0)
        self._angular_vel_scale = rospy.get_param('angular_vel_scale', 1.0)
    
        self._rate = rospy.Rate(10)
        self._joy_msg = Joy()


    def main_loop(self):
        while not rospy.is_shutdown():
            cmd_vel = Twist()
            if len(self._joy_msg.axes) > self._axis_linear:
                cmd_vel.linear.x = self._joy_msg.axes[self._axis_linear]*self._linear_vel_scale
            if len(self._joy_msg.axes) > self._axis_angular:
                cmd_vel.angular.z = self._joy_msg.axes[self._axis_angular]*self._angular_vel_scale
            self._cmd_vel_pub.publish(cmd_vel)
            self._rate.sleep()


    def _joy_callback(self, msg):
        self._joy_msg = msg


if __name__ == '__main__':
    rospy.init_node('eigenbot_joy_interface')
    eigenbot_joy_interface = EigenbotJoyInterface()
    eigenbot_joy_interface.main_loop()
