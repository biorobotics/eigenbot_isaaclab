#!/usr/bin/env python

"""
@author: jwhitman@cmu.edu
Julian Whitman
This ros node listens for eigenbot/topology messages, then takes the first one,
then decodes it, prints out the assembly info, and then assembles and launches it.
note: make sure to run chmod +x description_listener.py first to make it executable
"""
# see http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29

import rospy
import message_parsing
import description_assembler
from std_msgs.msg import String
last_data = []

def callback(data):
    global last_data
    if not(last_data==data.data): # only log new data. If it's the same, ignore it.
	    rospy.loginfo(rospy.get_caller_id() + " I heard %s", data.data)
     	    if len(data.data)>0:
        	    module_ID, module_attachments, module_ID_serials = message_parsing.parse_message_str(data.data)
        	    print module_ID
        	    print module_ID_serials
        	    print module_attachments
        	    description_assembler.description_assemble(module_ID, module_attachments,module_ID_serials)
        	    rospy.signal_shutdown("Wrote assembly, shutting this node down")



def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('description_listener', anonymous=True)

    rospy.Subscriber("eigenbot/topology", String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
