# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:11:01 2019

@author: Julian Whitman
"""

#https://docs.python.org/2.7/library/xml.etree.elementtree.html
# https://stackabuse.com/reading-and-writing-xml-files-in-python/
import xml.etree.ElementTree as ET
tree = ET.parse('test.xml')
root = tree.getroot()
#root = ET.fromstring(country_data_as_string)



# create the file structure
data = ET.Element('data')  
items = ET.SubElement(data, 'items')  # SubElement(parent, tag, attrib={}, **extra)
item1 = ET.SubElement(items, 'item')  
item2 = ET.SubElement(items, 'item')  
item1.set('name','item1')  
item2.set('name','item2')  
item1.text = 'item1abc'  
item2.text = 'item2abc'

# create a new XML file with the results
mydata = ET.tostring(data)  
myfile = open("items2.xml", "w")  
myfile.write(mydata)  
myfile.close()




        
#    active_ports  = module_ports[module_num]
#    active_mounts = port_mounts[module_num]
#    for port_num in range(len(active_ports)):
#        active_port = active_ports[port_num]
#        active_mounts = active_mounts[port_num]
#        # select the right port from the list and get its mount
#        port = ports[active_port]
#        parent_rigid_body = port.find('parent').text
#        parent_rigid_bodies.append(parent_rigid_body)
#        mount = mounts[active_mounts]
#        xyz = mount.find('xyz').text
#        rpy = mount.find('rpy').text
#        mount_xyz.append(xyz)
#        mount_rpy.append(rpy)
#        print "attach on " + parent_rigid_body + " at" + xyz + rpy + "\n"