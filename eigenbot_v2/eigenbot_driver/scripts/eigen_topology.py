#!/usr/bin/env python3
#
# Software License Agreement (BSD License)
#
# Copyright (c) 2019, Charles Hart
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

import json
import rospy
from std_msgs.msg import String
from time import time, sleep

VERSION="0.3"


def ebus2json(ebus_str):
    jsonobj = {"id":"None", "type":"None", "orientation":"None", "children":["None"]}
    if isTopo(ebus_str): # topology status
        jsonobj["id"] = ebus_str[1:3]
        jsonobj["type"] = ebus_str[4:ebus_str.index(',')]
        jsonobj["orientation"] = ebus_str.split(',')[1]
        jsonobj["children"] = ebus_str.split(":")[0].split(',')[2:]
    return jsonobj

def isTopo(ebus_str):
    return len(ebus_str) > 3 and (ebus_str[0] == '.' and ebus_str[3] == 'S') and ':' in ebus_str

class eigen_topology():
    def __init__(self):
        rospy.init_node('eigen_topology')
        
        self.topoPub = rospy.Publisher('eigenbot/topology', String, queue_size=1)
        self.serialPub = rospy.Publisher('eigenbot/in', String, queue_size=1)
        rospy.Subscriber('eigenbot/out', String, self.eigenBusCB)
    
        q_rate = rospy.get_param('~query_rate', 3) # Hz
        self.rate = rospy.Rate(q_rate)
    
        self.topology = {}
        self.expiration = rospy.get_param('~expiration', 3) # sec
    
        print("EigenBot Topology Mapper Tool v{} running.".format(VERSION))
        print("Ready to query eigenbot modules on {}, listen on {}, & report on {}."
              .format('eigenbot/in', 'eigenbot/out', 'eigenbot/topology'))
        
        d_wait = rospy.get_param('~wait', 0) # sec
        for i in range(d_wait, 0, -1):
            print("{}...".format(i))
            sleep(1)

    def eigenBusCB(self, msg):
        if isTopo(msg.data):
            h = hash(msg.data)
            t = time()
            if not h in self.topology:
                self.topology[h] = [ebus2json(msg.data), t]
                rospy.loginfo("Eigenmodule {} connected!"
                        .format(self.topology[h][0].get("id")))
            else:
                self.topology[h][1] = t # time module was last seen
    
    def run(self):
        while not rospy.is_shutdown():
            topo_string = ""
            for item in self.topology.copy():
                dt = time() - self.topology[item][1]
                if dt > self.expiration:
                    p = self.topology.pop(item)
                    rospy.loginfo("Eigenmodule {} disconnected."
                            .format(p[0].get("id")))
                else:
                    #print(item, self.topology[item][0])
                    topo_string += json.dumps(self.topology[item][0]) + ","
            # publish the topology
            if len(topo_string) > 0:
                self.topoPub.publish(topo_string)
            # request topology status 
            self.serialPub.publish(String("FFO\n")) 
            
            self.rate.sleep()


if __name__ == "__main__":
    topo = eigen_topology()
    topo.run()

#eof
