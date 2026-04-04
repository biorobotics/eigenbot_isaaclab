#!/usr/bin/env python
# license removed for brevity
import rospy
from std_msgs.msg import String

def talker():
    pub = rospy.Publisher('eigenbot/topology', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(1) # hz
    while not rospy.is_shutdown():
        time_now = rospy.get_rostime().secs  
        #hello_str = "hello world %s" % str(time_now)[:-1]      
	   #hello_str = "hello world %s" % rospy.get_time()
        message_str = ".05S2,0,00,02.02S1,0,04,00.04S2,0,03,00.03S1,0,00,00"
        rospy.loginfo(message_str)
        pub.publish(message_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass


#Created on Sun May 12 10:48:59 2019
#by Julian Whitman
#
#This file publishes fake assembly info for testing purposes.

# based on  http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
# note: make sure to run chmod +x talker.py first to make it executable
# license removed for brevity
