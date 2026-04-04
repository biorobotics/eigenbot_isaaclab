#!/usr/bin/env python3

"""
@author: jwhitman@cmu.edu
Julian Whitman
This ros node listens for eigenbot/topology messages, then takes the first one,
then decodes it, prints out the assembly info, and then assembles and launches it.
note: make sure to run chmod +x description_listener.py first to make it executable
"""
# see http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29

import os
import sys

import rospkg
import rospy
from std_msgs.msg import String

_PACKAGE_DIR = rospkg.RosPack().get_path("eigenbot")
_SCRIPT_DIR = os.path.join(_PACKAGE_DIR, "script")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from topology_message_tools import parse_topology
from eigenbot_joystick_ros import joint_state_talker

last_data = []


def callback(data):
    global last_data
    if last_data != data.data:
        last_data = data.data
        if len(data.data) > 0:
            module_types_str, graph_edges, module_ids = parse_topology(data.data)
            print('------ parsed to:')
            print('module_types_str = ')
            print(module_types_str)
            print('graph_edges = ')
            print(graph_edges)
            print('module_ids = ')
            print(module_ids)
            print('---------')

            joint_state_talker(module_types_str, graph_edges, module_ids)



def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    # rospy.init_node('description_listener', anonymous=True)
    rospy.init_node('description_listener')
    data = rospy.wait_for_message("/eigenbot/topology", String)
    if len(data.data) > 0:
        module_types_str, graph_edges, module_ids = parse_topology(data.data)
        joint_state_talker(module_types_str, graph_edges, module_ids)

if __name__ == '__main__':
    listener()
