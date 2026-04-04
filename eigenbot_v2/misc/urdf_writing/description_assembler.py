# -*- coding: utf-8 -*-
"""
Created on Tue Apr 2 13:59:15 2019

@author: jwhitman@cmu.edu
Julian Whitman
This file reads in modules from xml and then writes an xacro for ros use.

TODO: reads and import the same files multiple times, could check for repeats.

"""


import xml.etree.ElementTree as ET
# uses https://docs.python.org/2/library/xml.etree.elementtree.html

# Pretend that we got the following data from the modules:
#module_ID = ['Module_Type_1', 'Module_Type_2'] # these should match the file names
module_ID =          [ 'Module_Type_1',         'Module_Type_2',           'Module_Type_1', 'Bendy_module'] # these should match the file names
module_attachments = [ [[1,0,'Module_Type_2']], [[0, 1,'Module_Type_1']], [[0, 0,'Bendy_module']],  [] ] # where [] indicates there is nothing attached to its ports
# This is a somewhat complicated data structure, but the idea is this:
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
    tree = ET.parse(module_name + '.xml')
    treeList.append(tree)
    root = tree.getroot()
    
    # find the file name field
    fname = root.find('filename').text
    fnames.append(fname)
    print "file: " + fname
    
    # find the input port field
    input_port = root.find('input_port').text
    input_ports.append(input_port)
    print "input: " + input_port
    
    ports = root.find('output_ports')
#    for child in ports:
#        print child.tag, child.attrib, child.text


    M_num_parent = 'M' + str(module_num)

    # go through each port attachment and break it into parent-child relationships
    for attachment in module_attachments[module_num]:
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
        print "attach " + child_module + " on " + parent_rigid_body + " at (" + xyz + '), (' + rpy + ")\n"



# after reading in all the modules and ports, we can now write the file
f = open('autoXACRO.xacro', 'w')
f.write('<?xml version="1.0"?> \n')
f.write('<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="modular_robot_compiled">  \n\n')
f.write('<xacro:property name="M_PI" value="3.1415926535897931" /> \n')
f.write('<link name="fix_base"></link> \n')
for module_num in range(len(module_ID)):
    
    f.write('<!--' + 'Include files for, then joint to attach, module type ' + module_ID[module_num] + '-->\n')
    M_num = 'M' + str(module_num)
    
    # write file name to include
    fname = fnames[module_num]
    f.write('<xacro:include filename="./' + fname + '"/>\n')
    
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

