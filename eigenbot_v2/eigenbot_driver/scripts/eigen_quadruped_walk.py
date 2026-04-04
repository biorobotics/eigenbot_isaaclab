import time
from math import sin, pi
import numpy as np
#import matplotlib.pyplot as plt
import rospy
from sensor_msgs.msg import JointState

OUTPUT = "ROS"
test_mod = 'ff'

def wave(): #period, amplitude, y_offset, x_offset=0):
  d_wait = 0
  d_test = 65.1

  period = 3
  amplitude = pi/12.0
  offset = 0

  period2 = 5
  amplitude2 = pi/9.0
  offset2 = pi*1.0/8.0

  period3 = 10
  amplitude3 = pi/8.0
  offset3 = pi*1.0/8.0

  t_start = time.time()
  rospy.init_node('scripted_motion')
  rate = rospy.Rate(20)
  pub = rospy.Publisher('eigenbot/joint_cmd', JointState, queue_size=1)

  kill = False

  while not kill and not rospy.is_shutdown():
      t_now = time.time()
      t_elapsed = t_now - t_start
      
      if t_elapsed < d_wait:
          set_pos = offset
          set_pos2 = offset2
          set_pos3 = offset3
      elif t_elapsed < d_wait + d_test:
          set_pos = offset + amplitude * sin((2.0*pi*t_elapsed)/period)
          set_pos2 = offset2 + amplitude2 * sin((2.0*pi*t_elapsed)/period2)
          set_pos3 = offset3 + amplitude3 * sin((2.0*pi*t_elapsed)/period3)
      else:
          set_pos = offset
          set_pos2 = offset2
          set_pos3 = offset3
          kill = True

      if OUTPUT == "ROS":
          js = JointState()
          # eigenbot js.name = ['0A', '0B', '0C']
          js.name = [test_mod] #, '00', '00']
          # o6
          #js.name = ['07', '0A', '13']
          js.position = [set_pos] #, set_pos2, set_pos3]
          js.velocity = [float('nan')] #, float('nan'), float('nan')]
          js.effort = [float('nan')] #, float('nan'), float('nan')]
          pub.publish(js)
      
      #time.sleep(0.005)
      if OUTPUT == "PLT":
          plt.scatter(t_now, set_pos, c='red', marker='o')
          plt.scatter(t_now, set_pos2, c='green', marker='o')
          plt.scatter(t_now, set_pos3, c='blue', marker='o')
          plt.draw()
          plt.pause(0.005)
      if OUTPUT == "ROS":
          rate.sleep()

if __name__ == "__main__":
  wave()
  #time.sleep(3)
      
