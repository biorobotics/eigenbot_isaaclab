#!/usr/bin/env python
#http://docs.ros.org/diamondback/api/kinematics_msgs/html/__GetPositionIK_8py_source.html

import sys
import os
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from math import pi
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from moveit_commander.conversions import pose_to_list
import time

from get_ik import GetIK
#from moveit_python_tools.get_ik import GetIK
from geometry_msgs.msg import PoseStamped

def main():
  try:

    ## First initialize `moveit_commander`_ and a `rospy`_ node:
    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node('move_group_as_ik_test',anonymous=True)

    ## Instantiate a `RobotCommander`_ object. This object is the outer-level interface to
    ## the robot:
    robot = moveit_commander.RobotCommander()

    ## Instantiate a `PlanningSceneInterface`_ object.  This object is an interface
    ## to the world surrounding the robot:
    scene = moveit_commander.PlanningSceneInterface()

    ## Instantiate a `MoveGroupCommander`_ object.  This object is an interface
    ## to one group of joints.  In this case the group is the joints in the Panda
    ## arm so we set ``group_name = panda_arm``. If you are using a different robot,
    ## you should change this value to the name of your robot arm planning group.
    ## This interface can be used to plan and execute motions on the Panda:
    group_name = "modular_arm"
    group = moveit_commander.MoveGroupCommander(group_name)

    ## We create a `DisplayTrajectory`_ publisher which is used later to publish
    ## trajectories for RViz to visualize:
#    display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
#                                                   moveit_msgs.msg.DisplayTrajectory,
#                                                   queue_size=20)


    ## Getting Basic Information
    ## ^^^^^^^^^^^^^^^^^^^^^^^^^
    # We can get the name of the reference frame for this robot:
    planning_frame = group.get_planning_frame()
    print "============ Reference frame: %s" % planning_frame

    # We can also print the name of the end-effector link for this group:
    eef_link = group.get_end_effector_link()
    print "============ End effector: %s" % eef_link

    # We can get a list of all the groups in the robot:
    group_names = robot.get_group_names()
    print "============ Robot Groups:", robot.get_group_names()

    # Sometimes for debugging it is useful to print the entire state of the
    # robot:
    print "============ Printing robot state"
    print robot.get_current_state()
    print ""

    print("getting workspace pose")
    wpose = group.get_current_pose().pose

    print "restarting joint state publisher"
    os.system("rosnode kill /joint_state_publisher")
    joint_state_publisher = rospy.Publisher('joint_states',
                                                   JointState,
                                                   queue_size=1)
                                                   


   



    ## Planning to a Pose Goal
    ## ^^^^^^^^^^^^^^^^^^^^^^^
    ## We can plan a motion for this group to a desired pose for the
    ## end-effector:    pose_goal = geometry_msgs.msg.Pose()
  #  pose_goal = geometry_msgs.msg.Pose()
  #  pose_goal.orientation.w = 1.0
  #  pose_goal.position.x = -wpose.position.x
  #  pose_goal.position.y = wpose.position.y
  #  pose_goal.position.z = wpose.position.z
  #  group.set_pose_target(pose_goal)

#
##    ## Now, we call the planner to compute the plan and execute it.
#    plan = group.plan()
#    print ('------------ plan ---------------')
#    print(plan)
#
#
#    ## Displaying a Trajectory
#    ## ^^^^^^^^^^^^^^^^^^^^^^^
#    ## You can ask RViz to visualize a plan (aka trajectory) for you. But the
#    ## group.plan() method does this automatically so this is not that useful
#    ## here (it just displays the same trajectory again):
#    ##
#    ## A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
#    ## We populate the trajectory_start with our current robot state to copy over
#    ## any AttachedCollisionObjects and add our plan to the trajectory.
#    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
#    display_trajectory.trajectory_start = robot.get_current_state()
#    display_trajectory.trajectory.append(plan)
#    # Publish
#    display_trajectory_publisher.publish(display_trajectory);




    gik = GetIK(group_name, ik_timeout=0.1, ik_attempts=5, avoid_collisions=True)
#    gik = GetIK(group_name, ik_timeout=1.0, ik_attempts=5, avoid_collisions=False)
    ps = PoseStamped()
    ps.header.frame_id = eef_link
    ps.pose.position.x = wpose.position.x
    ps.pose.position.y = wpose.position.y
    ps.pose.position.z = wpose.position.z
    #ps.pose.orientation.w = 1.0
    t1= rospy.get_time()
    ik_out = gik.get_ik(ps)
    t2 = rospy.get_time()
    print " ============solved ik in " + str(t2-t1) + " seconds"
    print ik_out

# note to self: ik_out isa GetPositionIKResponse
#ik_out..solution isa moveit_msgs/RobotState    
    

    ## Displaying a Trajectory
    ## ^^^^^^^^^^^^^^^^^^^^^^^
    ## You can ask RViz to visualize a plan (aka trajectory) for you. But the
    ## group.plan() method does this automatically so this is not that useful
    ## here (it just displays the same trajectory again):
    ##
    ## A `DisplayTrajectory`_ msg has two primary fields, trajectory_start and trajectory.
    ## We populate the trajectory_start with our current robot state to copy over
    ## any AttachedCollisionObjects and add our plan to the trajectory.
#    display_trajectory = moveit_msgs.msg.DisplayTrajectory()
##    display_trajectory.trajectory_start = robot.get_current_state()
#    display_trajectory.trajectory_start = ik_out.solution
#    
#    plan_ik = moveit_msgs.msg.RobotTrajectory()
#    
#    for i in range(100):
#        plan_ik.append(ik_out.solution)
#    display_trajectory.trajectory.append(plan_ik)
    
#    display_trajectory.trajectory.append(plan)
#    # Publish
#    display_trajectory_publisher.publish(display_trajectory);

    while not rospy.is_shutdown():
        k = raw_input('Enter your input:')
        print("saw: " + k )
        if k == "a":

            ps.pose.position.x = ps.pose.position.x - 0.1
            ik_out = gik.get_ik(ps)
            print ik_out

        elif k == "d":

            ps.pose.position.x = ps.pose.position.x + 0.1
            ik_out = gik.get_ik(ps)
            print ik_out
        elif k == "w":

            ps.pose.position.z = ps.pose.position.z - 0.1
            ik_out = gik.get_ik(ps)
            print ik_out

        elif k == "s":

            ps.pose.position.z = ps.pose.position.z + 0.1
            ik_out = gik.get_ik(ps)
            print ik_out
        print([ps.pose.position.x, ps.pose.position.y, ps.pose.position.z])
        joint_state_publisher.publish(ik_out.solution.joint_state)

  except rospy.ROSInterruptException:
    return
  except KeyboardInterrupt:
    return

if __name__ == '__main__':
  main()


