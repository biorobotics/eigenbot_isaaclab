#!/usr/bin/env python
# see http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
# note: make sure to run chmod +x listener.py first to make it executable
import rospy
from std_msgs.msg import String
last_data = []


def callback(data):
    global last_data
    if not(last_data==data.data): # only log new data. If it's the same, ignore it.
	    rospy.loginfo(rospy.get_caller_id() + " I heard %s", data.data)
	    last_data = data.data

def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    rospy.Subscriber("chatter", String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
