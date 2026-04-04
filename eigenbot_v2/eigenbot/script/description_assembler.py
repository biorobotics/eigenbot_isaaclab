# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2 13:59:15 2019

@author: jwhitman@cmu.edu
Julian Whitman
This file reads in modules from xml and then writes an xacro for ros use.

Then it launches rviz after a keypress (to save from having to type the command)


TODO: reads and import the same files multiple times, could check for repeats.

"""
import xml.etree.ElementTree as ET
import subprocess
import sys
import os

def description_assemble(module_ID, module_attachments, module_ID_serials):
#if True:

  # uses https://docs.python.org/2/library/xml.etree.elementtree.html


  # Pretend that we got the following data from the modules:
  #module_ID = ['Module_Type_1', 'Module_Type_2'] # these should match the file names


  # module_attachments is a somewhat complicated data structure, but the idea is this:
  #      [ [things attached to module 1], [things attached to module 2], ... ]
  #     where  [things attached to module n] = [[port number, mount number, child module attached], [port number, mount number, child module attached]]
  #     This should be enough information to define a tree. Each module passes its own name, and a list of the modules it sees on its output ports. 


  # preallocations:
  input_ports = []
  fnames = []
  # first RB is free floating
  parent_rigid_bodies = ['fix_base'] 
  mount_xyz = ['0 0 0'] 
  mount_rpy = ['0 0 0'] 


  treeList = list()
  for module_num in range(len(module_ID)):
      module_name= module_ID[module_num]
      tree = ET.parse('../description/' + module_name + '.xml')
      treeList.append(tree)
      root = tree.getroot()
      
      # find the file name field
      fname = root.find('filename').text
      fnames.append(fname)
      print("file: " + fname)
      
      # find the input port field
      input_port = root.find('input_port').text
      input_ports.append(input_port)
      print("input: " + input_port)
      
      ports = root.find('output_ports')
      #    for child in ports:
      #        print(child.tag, child.attrib, child.text


      M_num_parent = 'S' + str(module_ID_serials[module_num]) +'M' + str(module_num)

      # go through each port attachment and break it into parent-child relationships
      for attachment in module_attachments[module_num]:
        if len(attachment)>0:
          active_port = attachment[0]
          active_mount = attachment[1]
          child_module = attachment[2]
          port = ports[active_port] # from xml ports listing
          parent_rigid_body = port.find('parent').text
          parent_rigid_bodies.append(parent_rigid_body + M_num_parent)
          mounts = port.findall('mount')
          mount = mounts[active_mount]
          xyz = mount.find('xyz').text
          rpy = mount.find('rpy').text
          mount_xyz.append(xyz)
          mount_rpy.append(rpy)
          print("attach " + child_module + " on " + parent_rigid_body + " at (" + xyz + '), (' + rpy + ")\n")

  # keep the last module's first output port
  last_output = ports[0].find('parent').text + M_num_parent

  # after reading in all the modules and ports, we can now write the file
  f = open('../urdf/autoXACRO.xacro', 'w')
  f.write('<?xml version="1.0"?> \n')
  f.write('<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="modular_robot_compiled">  \n\n')
  f.write('<xacro:property name="M_PI" value="3.1415926535897931" /> \n')
  f.write('<link name="fix_base"></link> \n')
  for module_num in range(len(module_ID)):
      
      f.write('<!--' + 'Include files for, then joint to attach, module type ' + module_ID[module_num] + '-->\n')
      M_num = 'S' + str(module_ID_serials[module_num]) +'M' + str(module_num)
      
      # write file name to include
      fname = fnames[module_num]
      f.write('<xacro:include filename="../urdf/' + fname + '"/>\n')
      
      # invoke the file with the module number ID as name
      dot_index = fname.find('.') # returns first instance of the dot
      xacro_name = fname[0:dot_index] 
      f.write('<xacro:' + xacro_name + ' module_label="' + M_num + '"/>\n\n')
      
      # write joints between the module and its child
      f.write('<joint name="module_' + str(module_num) + '_attachment" type="fixed">' + "\n")
      f.write('<origin' + "\n")
      f.write('xyz="' + str(mount_xyz[module_num]) + '"\n')
      f.write('rpy="' + str(mount_rpy[module_num]) + '"\n')
      f.write('/>' + "\n")
      f.write('<parent link="' + parent_rigid_bodies[module_num] + '"/>' + "\n")
      f.write('<child link="' + input_ports[module_num] + M_num + '"/>' + "\n")
      f.write('</joint>' + "\n")
      f.write('\n')

  f.write('</robot>' + "\n")
  f.close()



  # run some commands to make the package. 


#  catkin_path = subprocess.check_output(['catkin', 'locate']) # get path of ROS workspace
  catkin_path = '/home/cobracommander/catkin_ws'
  cwd = os.getcwd()
  catkin_path = cwd[:cwd.rfind('_ws') + len('_ws')]
  #catkin_path = catkin_path[0:-1] # remove the \n from end of string
  print("catkin path:" + catkin_path)

  # compile xacro to urdf
  urdf_path = catkin_path +'/src/eigenbot/eigenbot/urdf/'
  p = subprocess.Popen('rosrun xacro xacro --inorder -o ' + urdf_path + 'autoXACRO.urdf ' + urdf_path + 'autoXACRO.xacro', shell=True, cwd="/")
  p.wait()

  # next read the newly created urdf to extract all the joint names, so that they can be used for moveit config files
  urdf_tree = ET.parse('../urdf/autoXACRO.urdf')
  urdf_root = urdf_tree.getroot()
  joint_list = urdf_root.findall('joint')
  moving_joint_list= []
  joint_parent_list = []
  joint_child_list = []
  print("Joints found:")
  for joint in joint_list:
    joint_parent_list.append(joint.find('parent').get('link'))
    joint_child_list.append( joint.find('child').get('link') )
    if joint.get('type')=='continuous' or joint.get('type')=='revolute' or joint.get('type')=='prismatic':
      print(joint.get('name'))
      moving_joint_list.append(joint.get('name'))
    
    

  # write some extra files for moveit: still in beta
  #modular_robot_compiled.srdf
  f = open('../../eigenbot_moveit_config/config/modular_robot_compiled.srdf', 'w+')
  f.write('<?xml version="1.0" ?>\n')
  f.write('<!--This does not replace URDF, and is not an extension of URDF.\n')
  f.write('    This is a format for representing semantic information about the robot structure.\n')
  f.write('    A URDF file must exist for this robot as well, where the joints and the links that are referenced are defined-->\n')
  f.write('<robot name="modular_robot_compiled">\n')
  f.write('  <group name="modular_arm">\n')
  f.write('        <chain base_link="fix_base" tip_link="' + last_output + '" />\n')
  f.write('    </group>\n')
  f.write('<!--END EFFECTOR: Purpose: Represent information about an end effector.-->\n')
  f.write('    <end_effector name="arm_end_effector" parent_link="' + last_output + '" group="modular_arm" />\n')
  f.write('    <!--VIRTUAL JOINT: Purpose: this element defines a virtual joint between a robot link and an external frame of reference (considered fixed with respect to the robot)--> \n')
  f.write('    <virtual_joint name="virtual_fix_base_joint" type="fixed" parent_frame="virtual_fix_base_frame" child_link="fix_base" /> \n')
  f.write('    <!--DISABLE COLLISIONS: By default it is assumed that any link of the robot could potentially come into collision with any other link in the robot. This tag disables collision checking between a specified pair of links. -->\n')
  #for module_num in range(len(module_ID)):
  #    M_num = 'M' + str(module_num)
  #    f.write('<disable_collisions link1="' + parent_rigid_bodies[module_num] + '" link2="' + input_ports[module_num] + M_num + '" reason="Fixed joint"/>' + "\n")
  #for module_num in range(len(module_ID)-1):
  #    M_num = 'M' + str(module_num)
  #    M_num_plus_one = 'M' + str(module_num+1)
  #    f.write('<disable_collisions link1="' + parent_rigid_bodies[module_num] + '" link2="' + parent_rigid_bodies[module_num+1] + '" reason="Adjacent outputs"/>' + "\n")
  #    f.write('<disable_collisions link1="' + input_ports[module_num] + M_num + '" link2="' + input_ports[module_num] + M_num_plus_one + '" reason="Adjacent inputs"/>' + "\n")      
  #    f.write('<disable_collisions link1="' + parent_rigid_bodies[module_num] + '" link2="' + parent_rigid_bodies[module_num+1] + '" reason="Adjacent outputs"/>' + "\n")
  #    f.write('<disable_collisions link1="' + input_ports[module_num] + M_num_plus_one + '" link2="' + parent_rigid_bodies[module_num] + '" reason="same module"/>' + "\n")      # this is not really correct, just hacking it out.
  # --- TODO: find appropriate collision to disable. ----
  for joint_num in range(len(joint_list)):
      f.write('<disable_collisions link1="' + joint_parent_list[joint_num] + '" link2="' + joint_child_list[joint_num] + '" reason="Joint connection"/>' + "\n")


   # other disable collisions may be possible. Might need to disable collisions for links connected by a joint.
  f.write('</robot>');
  f.close()



  # write file joint_limits.yaml
  print("Writing joint_limits.yaml...")
  f = open('../../eigenbot_moveit_config/config/joint_limits.yaml', 'w+')
  f.write('# ---- This file written by description_assembler.py----\n')
  f.write('# joint_limits.yaml allows the dynamics properties specified in the URDF to be overwritten or augmented as needed \n')
  f.write('# Specific joint properties can be changed with the keys [max_position, min_position, max_velocity, max_acceleration] \n')
  f.write('# Joint limits can be turned off with [has_velocity_limits, has_acceleration_limits] \n')
  f.write('joint_limits:\n')
  for moving_joint in moving_joint_list:
    f.write('  ' + moving_joint + ':\n')
    f.write('    has_velocity_limits: true\n')
    f.write('    max_velocity: 2\n')
    f.write('    has_acceleration_limits: false\n')
    f.write('    max_acceleration: 0\n')
  f.close()


  # write file fake_controllers.yaml
  print("Writing fake_controllers.yaml...")
  f = open('../../eigenbot_moveit_config/config/fake_controllers.yaml', 'w+')
  f.write('controller_list:\n')
  f.write('  - name: fake_modular_arm_controller\n')
  f.write('    joints:\n')
  for moving_joint in moving_joint_list:
    f.write('      - ' + moving_joint + '\n')
  f.close()



  # write file ros_controllers.yaml
  print("Writing ros_controllers.yaml...")
  # blocks of text saved in thest txt files to be written in.

  f=open('ros_controllers_yaml_text1.txt')  
  f2=open('../../eigenbot_moveit_config/config/ros_controllers.yaml','w+')
  for x in f.readlines():
      f2.write(x)
  f.close()


  for moving_joint in moving_joint_list:
    f2.write('      - '+moving_joint+'\n')

  # blocks of text saved in thest txt files to be written in.
  f=open('ros_controllers_yaml_text2.txt')  
  for x in f.readlines():
      f2.write(x)
  f.close()

  for moving_joint in moving_joint_list:
    f2.write('      - '+moving_joint+'\n')
  f2.write('    gains:\n')
  for moving_joint in moving_joint_list:
    f2.write('      '+moving_joint+':\n')
    f2.write('        p: 100\n')
    f2.write('        d: 1\n')
    f2.write('        i: 1\n')
    f2.write('        i_clamp: 1\n')
  f2.close()


#  # make it all: it turns out this is unecessary if all we are changing is the urdfs, srdfs, etc. No need.
#  #print("running catkin_make"
#  #p= subprocess.Popen('catkin_make', shell=True, cwd=catkin_path)
#  #p.wait()
#  #print('catkin_make success'
#
##  raw_input("Press enter to continue...")
#  input("Press enter to continue...")
#
#  # Launch moveit
#  #print("launching moveit in rviz"
#  #p= subprocess.Popen('roslaunch eigenbot_moveit_config demo.launch', cwd=catkin_path) #  shell=True,
#
#  # launch rviz demo: This process can cause issues with hidden roscores so be careful if enabling it.
#  print("launching rviz demo")
#  p= subprocess.Popen('roslaunch eigenbot rviz_test.launch', shell=True, cwd=catkin_path)
#  #p.wait()
#

if __name__== "__main__":
#  module_ID = ['Static_base_module', # make sure there is a comma serparating the values!
#            'Torsional_module',
#            'Static_elbow_module',
#            'Torsional_module',
#            'Static_elbow_module',
#            'Torsional_module',
#            'Static_elbow_module',
#            'Torsional_module',
#            'Static_elbow_module',
#            'Torsional_module',
#            'Static_elbow_module']
#  module_attachments = [  [[0,0,'Torsional_module']], 
#                      [[0,2,'Static_elbow_module']], 
#                      [[0,2,'Torsional_module']], 
#                      [[0,2,'Static_elbow_module']], 
#                      [[0,2,'Torsional_module']], 
#                      [[0,2,'Static_elbow_module']], 
#                      [[0,2,'Torsional_module']], 
#                      [[0,2,'Static_elbow_module']], 
#                      [[0,2,'Torsional_module']], 
#                      [[0,2,'Static_elbow_module']], 
#                      []]

#  module_ID = ['Hebi_X5_9_module', # make sure there is a comma serparating the values!
#            'Hebi_tube_150_0',
#            'Hebi_X5_9_module',
#            'Hebi_tube_150_1',
#            'Hebi_X5_9_module',
#            'Hebi_tube_150_2',
#            'Hebi_X5_9_module',
#            'Hebi_tube_150_3']
#  module_ID_serials = ['', '', '', '', '', '','','', ''] 
#  module_attachments = [  
#                      [[0,0,'Hebi_tube_150_0']], 
#                      [[0,0,'Hebi_X5_9_module']], 
#                      [[0,0,'Hebi_tube_150_1']], 
#                      [[0,0,'Hebi_X5_9_module']], 
#                      [[0,0,'Hebi_tube_150_2']], 
#                      [[0,0,'Hebi_X5_9_module']], 
#                      [[0,0,'Hebi_tube_150_3']], 
#                      []]

    module_ID = ['Static_base_module', # make sure there is a comma serparating the values!
            'Bendy_module',
            'Gripper_module']
    module_ID_serials = ['', '', '', '', '', '','','', ''] 
    module_attachments = [  
                      [[0,0,'Bendy_module']], 
                      [[0,0,'Gripper_module']], 
                      []]
    description_assemble(module_ID, module_attachments, module_ID_serials)



