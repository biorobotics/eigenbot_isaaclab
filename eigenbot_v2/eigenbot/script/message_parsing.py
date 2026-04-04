# -*- coding: utf-8 -*-
"""
Created on Sun May 12 10:48:59 2019

@author: jwhitman
This method takes in a string containing topology information about the sequence of modules.
The message string reports what the name, type, mount index, and neighbors are for each module.
These messages are broken up by a single period and the messages can be concatenated in any order, since the script should be able to sort it out as long as no info is missing.
I assume the robot will start with a static base module.


S: .[src_addr]S[node_type (see EEPROM address map for values],[connector orientation 1-8 or 0 if error detecting],[J6 connector neighbor],[J11 connector neighbor]\n
Topology status as comma separated list of ports.  Also includes connector orientation with 0 for unknown and 1-8 for the 8 different orientations of the connector.
Ex. .1AS4,2,31,2B\n Node 0x1A is reporting that it’s connector orientation is position 4(45deg clocking), node_type=2(twist module), node 0x31 is connected to port 1, and node 0x2B is connected to port 2
Ex. .1AS4,2,00,2B\n Node 0x1A is reporting that it’s connector orientation is position 4(45deg clocking), node_type=2(twist module), nothing is connected to port 1, and node 0x2B is connected to port 2

The base has an RS485 adapter connected to it that just needs to be plugged into the computer you are going to use.  The serial port should be opened at 115200 baud, 8N1, no flow control.  The lab power supply should be connected and set to 24V (green wire is ground).  You should not hotplug the modules.

So right now the node addresses are:
02: 90 deg (node type 1)
03: 90 deg (node type 1)
04: twist (node type 2)
05: twist (node type 2)

So for example if you have chain where 5 is connected to the base, then 2, then 4, then 3 you will see the follow response to a FFO command (broadcast request for orientation status)
H05
.05S2,n,02,00
.02S1,n,04,00
.04S2,n,03,00
.03S1,n,00,00
where n represents the orientation of the connector from 1-8 (or 0 if error occurs)

The results should always be in order from base to tip because of propagation delay.  There is a chance that one of the twist modules has the downstream cable connected to the other port (messing with it trying to debug a wiring issue).  In that case you would get .05S2,n,00,02
Also the two twist modules have different speed servos in them so one will be a little faster than the other (but the encoder counts/revolution is the same)
It is possible that timestamp messages may be mixed with the S messages.  You could also manually build up using the H message to establish the base node and then doing individual O requests to specific addresses.  Either way you could still have a timestamp message mixed with responses.


------
How to decode messages:
- break by periods
- break by commas
- build tree (list for now since trees require some kind of recursion or extra work)
- return the parsed topology info
"""

def parse_message_str(message_str):
#message_str = ".05S2,0,02,00.02S1,1,04,00.04S2,2,03,00.03S1,3,00,00"
#message_str = ".05S2,0,00,02.02S1,1,04,00.04S2,2,03,00.03S1,3,00,00"



#if True:
    period_inds = [n for n in xrange(len(message_str)) if message_str.find('.', n) == n] # (message_str == '.')
    message_segments = []
    message_seg_contents = []
    for i in range(len(period_inds)-1):
	    message_segments.append(message_str[period_inds[i]+1:period_inds[i+1]])
    message_segments.append(message_str[period_inds[-1]+1:])
    for message_segment in message_segments:
	    comma_inds = [n for n in xrange(len(message_segment)) if message_segment.find(',', n) == n] # (message_str == '.,)
	    if len(comma_inds)>1 and message_segment.find('S')>0: #rough check that its the right message type
		    # the number after S is its type
		    name_type = message_segment[:comma_inds[0]]
		    ind_S = name_type.find('S')
		    my_name = name_type[:ind_S] # this module's name
		    my_type = name_type[(ind_S+1):] # this modules type
		    my_mount_index = message_segment[comma_inds[0]+1:comma_inds[1]] # this modules mount index
		    my_neighbors = [] # arbitrary number of connected neighbors
		    for i in range(1,len(comma_inds)):
                      my_neighbors.append(message_segment[comma_inds[i]+1:comma_inds[i]+3])
		    if int(my_neighbors[0])==0:
                      my_neighbors = [my_neighbors[1], my_neighbors[0]]
		    message_seg_contents.append([my_name, my_type, my_mount_index, my_neighbors])
		#    message_seg_contents = [my_name, my_type, my_mount_index, my_neighbors_list]


    module_types = []
    module_serials=  []
    for i in range(len(message_seg_contents)):
	    my_type = int(message_seg_contents[i][1]) # NOTE: if we get into more than 9 module types, will need to convert from hex
	    #Type decoder: add new types here 
	    #(1=passive_90deg, 2=twist, 3=bend, 4=passive_tube)
	    # TODO: not terribly extensible, but quick and dirty. Fix later.
	    if my_type==1:
		type_name = "Static_elbow_module"
	    elif my_type==2:
		type_name = "Torsional_module"
	    elif my_type==3:
		type_name = "Bendy_module"
	    elif my_type==4:
		type_name = "Static_link_module"
	    elif my_type==5:
		type_name = "Gripper_module"
	    else:
		type_name = "Unknown"
	    module_types.append(type_name)
	    module_serials.append(int(message_seg_contents[i][0]))

	# next convert this to a set of module_ID and module attachment in chain order. 
	# TODO: Not sure if this will work for trees... do it later
	# assume they are in the right order for now
	    
	# for each module, find whats on which of its ports, and what inds theyre on
    module_attachments = []
	# assume they all start with static base even if that doesnt send back a message
    module_ID = ["Static_base_module"]    
    module_ID_serials = [0]
    
    # figure out which module is first by looking for which one does not appear as a neighbor
    found_serial = [-1]*len(module_types)
    for i in range(len(module_types)):
        serial_now = module_serials[i]
        for j in range(len(message_seg_contents)): # go through each and tag found modules
            neighbors = message_seg_contents[j][3]
            for neighbor in neighbors:
                if int(neighbor)==serial_now:
                    found_serial[i] = j
    # the index where found_serial
    modules_no_parent = [i for i, x in enumerate(found_serial) if x == -1]
    if len(modules_no_parent)>1:
        print "ERROR: more than one module has no parent"
    root_module = modules_no_parent[0]
    module_ID.append(module_types[root_module])
    module_ID_serials.append(module_serials[root_module])

    module_attachments.append([[0, int(message_seg_contents[root_module][2])-1, module_types[root_module]]])

# This is messy because I am trying to allow for the order of the segments to come in arbitrarily.
    
	# add the remaining modules # TODO: This is going to break for sure for actual trees. need real tree management, not like this.
    current_ind = root_module
    for i in range(len(message_seg_contents)):
        module_attachments_i = []
        for j in range(len(message_seg_contents[i][3])):
            neighbor = message_seg_contents[current_ind][3][j]
            if int(neighbor)>0:
                neighbor_ind = module_serials.index(int(neighbor)) 
                neighbot_mount = int(message_seg_contents[neighbor_ind][2])
                neighbor_port = j
                module_attachments_i.append([neighbor_port, neighbot_mount-1, module_types[neighbor_ind]])
                module_ID.append(module_types[neighbor_ind])
                module_ID_serials.append(module_serials[neighbor_ind])
                current_ind = neighbor_ind
            else:
                module_attachments_i.append([])
        module_attachments.append(module_attachments_i)
#    print module_ID
#    print module_attachments 
    return module_ID, module_attachments, module_ID_serials

if __name__== "__main__":
    message_str = ".05S2,0,00,02.02S1,1,04,00.04S2,2,03,00.03S1,3,00,00"
        # for testing: scramble the message
    inds = period_inds + [len(message_str)]
    segments = []
    for i in range(len(inds)-1):
        segments.append(message_str[inds[i]:inds[i+1]])
    import random
    segments_scrambled = segments[:]
    random.shuffle(segments_scrambled)
    message_str = ''.join(segments_scrambled)

    module_ID, module_attachments, module_ID_serials = parse_message_str(message_str)
    print module_ID
    print module_ID_serials
    print module_attachments

