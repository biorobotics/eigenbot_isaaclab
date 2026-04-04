# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:40:27 2019

@author: jwhitman
"""

import random
from anytree import Node, RenderTree

# message =[my_ID, my_type, parent_ID, child1(ID,port,index), child2(ID,port,index)...]
message1 = [1,'type1', 0, (2, 0, 0), (3,1,1) ]
message2 = [2,'type2', 1, (4, 0, 0) ]
message3 = [3,'type1', 1, (5, 0, 0) ]
message4 = [4,'type2', 2 ]
message5 = [5,'type2', 3 ]
messages = [message1, message2, message3, message4, message5]

random.shuffle(messages)

## note which is the root
root_node = Node('base')

## move along and build tree from there
building = True
current_parent = root_node
#while building:
for m in messages:
    if m[2]==root_node:
        node_now = Node(m[1], parent=current_parent)




        

#udo = Node("Udo")
#marc = Node("Marc", parent=udo)
#lian = Node("Lian", parent=marc)
#dan = Node("Dan", parent=udo)
#jet = Node("Jet", parent=dan)
#jan = Node("Jan", parent=dan)
#joe = Node("Joe", parent=dan)
#
#print(udo)
#Node('/Udo')
#print(joe)
#Node('/Udo/Dan/Joe')
#
#for pre, fill, node in RenderTree(udo):
#    print("%s%s" % (pre, node.name))
# print(dan.children)

        
