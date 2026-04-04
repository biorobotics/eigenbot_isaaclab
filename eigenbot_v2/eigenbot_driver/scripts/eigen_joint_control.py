#!/usr/bin/env python3
#
# Software License Agreement (BSD License)
#
# Copyright (c) 2020, Charles Hart, Carengie Mellon University
# chart at cmu dot edu
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.
# * Neither the names of the authors nor the names of their
# affiliated organizations may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import rospy
from numpy import nan, isnan
from sensor_msgs.msg import JointState
from std_msgs.msg import String
import copy

VERSION="0.5"

class eigen_joint_control():
    def __init__(self):
        rospy.init_node('eigen_joint_control')
        self.command_pub = rospy.Publisher('eigenbot/in', String, queue_size=100)
        self.feedback_pub = rospy.Publisher('eigenbot/joint_fb', JointState,
                queue_size=100)
        self.feedback = {}
        self.last_feedback = {}

        rospy.Subscriber('eigenbot/out', String, self.eigenFeedbackCB)
        rospy.Subscriber('eigenbot/joint_cmd', JointState, self.jointStateCB)

        freq = rospy.get_param('~rate', 10)
        self.rate = rospy.Rate(freq)

        #Print a message indicating a successful initialization
        print("Eigen Joint Control v{} running, with feedback at {}Hz."
                .format(VERSION, freq))

    # iterate through accumulated feedback values, publish, then empty
    def publishFeedback(self):
        fb = JointState()
        for node_id in self.last_feedback:
            fb.name.append(node_id)
            if "position" in self.feedback[node_id]:
                fb.position.append(self.feedback[node_id]["position"])
            else:
                fb.position.append(nan)
            if "velocity" in self.feedback[node_id]:
                fb.velocity.append(self.feedback[node_id]["velocity"])
            else:
                fb.velocity.append(nan)
            if "effort" in self.feedback[node_id]:
                fb.effort.append(self.feedback[node_id]["effort"])
            else:
                fb.effort.append(nan)
        self.feedback_pub.publish(fb)
        #self.feedback = {}

    # accumulate feedback messages so they can be published at a constant rate
    def eigenFeedbackCB(self, msg):
        if len(msg.data) > 3 and msg.data[0] == '.' and \
              (msg.data[3] == 'L' or msg.data[3] == 'V' or msg.data[3] == 'I'):
            node_id = msg.data[1:3]
            fb_type = msg.data[3]
            if node_id not in self.feedback:
                self.feedback[node_id] = {}
            for x in [('L', "position"), ('V', "velocity"), ('I', "effort")]:
                if fb_type == x[0]:
                    try:
                        node_fb = float(msg.data.split(x[0])[1].split(':')[0])
                        self.feedback[node_id][x[1]] = node_fb
                    except ValueError as e:
                        rospy.logerr(e)
            self.last_feedback = copy.deepcopy(self.feedback)

    # jointStateCB:
    # every time a joint state message is received, format and send the command
    def jointStateCB(self, msg):
        ns = msg.name
        ps = msg.position
        vs = msg.velocity
        es = msg.effort
        if len(ns) != len(ps) or len(ns) != len(vs) or len(ns) != len(es):
          rospy.logwarn("bad joint state message")
        else:
          for i in range(len(ns)):
            if isnan(ps[i]):
              rospy.logdebug("disabling position control of {}".format(ns[i]))
              ### TODO: Will implement when supported by firmware!!!
            else:
              #rospy.loginfo("Set position of {} to {}".format(ns[i], ps[i]))
              self.command_pub.publish("{}P{:.3f}".format(ns[i], ps[i]))

            if isnan(vs[i]):
              rospy.logdebug("disabling velocity control of {}".format(ns[i]))
              ### TODO: Will implement when supported by firmware!!!
            else:
              rospy.logdebug("Set velocity of {} to {}".format(ns[i], vs[i]))
              #self.command_pub.publish("{}S{:.3f}".format(ns[i], vs[i]))

            if isnan(es[i]):
              rospy.logdebug("disabling effort control of {}".format(ns[i]))
              ### TODO: Will implement when supported by firmware!!!
            else:
              #rospy.logdebug("Set effort of {} to {}".format(ns[i], es[i]))
              self.command_pub.publish("{}T{:.3f}".format(ns[i], es[i]))
          self.command_pub.publish("FFR")

    #run:
    # The main loop for this node. While the node is not shutdown, it will
    # publish joint state command messages as they are received, and
    # publish joint state feedback messages at the rate param specified.
    def run(self):
        while not rospy.is_shutdown():
            if self.feedback:
                self.publishFeedback()
            # request Position feedback from all modules (Velocity and Effort removed until supportd by modules)
            self.command_pub.publish("FFQ51")
            self.rate.sleep()

if __name__ == "__main__":
    controller = eigen_joint_control()
    controller.run()

#eof
