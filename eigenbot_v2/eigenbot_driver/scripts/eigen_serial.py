#!/usr/bin/env python3
#
# Software License Agreement (BSD License)
#
# Copyright (c) 2019, 2020 Charles Hart
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

import crc8
import rospy
from std_msgs.msg import String
import serial
from time import sleep

VERSION="0.3"

class eigenbot_driver():
    def __init__(self):
        rospy.init_node('eigenbot_serial_driver')

        #Setup the serial port using the specified parameters
        eigenPort = rospy.get_param('~port', '/dev/ttyACM0')
        eigenBaud = rospy.get_param('~baud', 500000)
        eigenIn = rospy.get_param('~in', 'eigenbot/in')
        eigenOut = rospy.get_param('~out', 'eigenbot/out')

        connection = None
        while connection is None and not rospy.is_shutdown():
            try:
                self.eigenSerial = serial.Serial(port=eigenPort, baudrate=eigenBaud, timeout=1.0)
                connection = "Yes!"
            except serial.serialutil.SerialException as e:
                rospy.logerr(e)
                sleep(3)

        self.eigenOutPub = rospy.Publisher(eigenOut, String, queue_size=1)
        rospy.Subscriber(eigenIn, String, self.eigenCmdCB, queue_size=1000)

        #Print a message indicating a successful initialization
        print("EigenBot Driver v{} running.".format(VERSION))
        print("Connected to {} port at {} baud.".format(eigenPort, eigenBaud))
        print("Ready to accept commands on {}, responding on {}."
            .format(eigenIn, eigenOut))

    #eigenCmdCB:
    # A callback that handles writing incoming ROS messages to the serial port
    # Guarantees that each message will end with a checksum and newline
    def eigenCmdCB(self, msg):
        #print("got cmd msg \"{}\"".format(msg))
        data = bytes(msg.data, 'ascii')
        # eigenbus packets need a payload, a colon, a crc, and a newline
        payload, *crc = data.strip(b'\n').split(b':')
        #print("payload: {} crc: {}".format(payload, crc))
        hash = crc8.crc8(initial_start = 0xff)
        hash.update(payload)
        packet = "{}:{:02X}\n".format(payload.decode('ascii'), ord(hash.digest()))
        #print("sending {}".format(packet))
        self.eigenSerial.write(packet.encode('ascii'))

    #run:
    # The main loop for this node. While the node is not shutdown, it will
    # publish all of the received eigenbus messages as ROS messages.
    def run(self):
        msgs = 1
        good_msgs = 1
        last_print = rospy.Time.now().secs
        while not rospy.is_shutdown():
            try:
                line = self.eigenSerial.readline().decode('ascii')
            except UnicodeDecodeError:
                continue
            msgs += 1
            if len(line) > 0:
                #If we receive a valid message, publish it for the other nodes
                if ':' in line:
                    hash = crc8.crc8(initial_start = 0xff)
                    payload, *crc = bytes(line, 'ascii').strip(b'\n').split(b':')
                    hash.update(payload)
                    check = bytes("{:02X}".format(ord(hash.digest())), 'ascii')
                    if crc is not None:
                        if crc[0] == check:
                            good_msgs += 1
                            self.eigenOutPub.publish(line)
                        else:
                            rospy.logerr("CRC fail")
                    else:
                        rospy.logerr("CRC missing")
                    #print("{}: n{}, p{}, c{}, c{}, l{}"
                    #        .format(rospy.Time.now(), len(line), payload.decode('ascii'), \
                    #                crc[0], check, line))
                else:
                    rospy.logerr("CRC incomplete")
                fresh_check = 100.0*(good_msgs/msgs)
                if rospy.Time.now().secs > last_print:
                    rospy.logwarn(">>> {:0.5}% of RX'd packets have good CRC.".format(fresh_check))
                    last_print = rospy.Time.now().secs


if __name__ == "__main__":
    driver = eigenbot_driver()
    driver.run()

#eof
