#!/usr/bin/env python3

import json
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import String


# Module types
WHEEL_MODULE_TYPE = 0x01
BENDY_MODULE_TYPE = 0x03
FOOT_MODULE_TYPE = 0x0A
STATIC_90_BEND_MODULE_TYPE = 0x0D
EIGENBODY_MODULE_TYPE = 0x0E

# Limb types
WHEEL_LIMB_TYPE = 1
LEG_LIMB_TYPE = 2


class EigenbotJointController():
    def __init__(self):
        # Joint info
        self.joint_fb_initialized = False
        self.joint_positions = []
        self.joint_names = []

        # Topology data
        self.topology_str = ''
        self.eigenbody_id = '99'  # TODO modify default value
        self.topology = {}
        self.topology_initialized = False
        self.n_chassis_ports = 6
        self.limb_info = [None]*self.n_chassis_ports
        self.chassis_ports = [0, 1, 3, 7, 5, 4] # Indices of chassis ports, empirically determined (from top left of EigenBot chassis, going clockwise)

        # Control data
        self.forward_cmd = 0
        self.turn_cmd = 0
        self.max_wheel_velocity = 2

        # Publishers and subscribers
        self.joint_fb_sub = rospy.Subscriber('/eigenbot/joint_fb', JointState, self.joint_fb_callback)
        self.topology_sub = rospy.Subscriber('/eigenbot/topology', String, self.topology_callback)
        self.cmd_vel_sub = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback)
        self.joint_cmd_pub = rospy.Publisher('/eigenbot/joint_cmd', JointState, queue_size=1)

        self.rate = rospy.Rate(20)

        # Gait parameters
        self.dt = 1.0/20 * np.pi
        self.const_offsets = np.array([
            [-1,0,1,-1,0,1],
            [-1,-1,-1,-1,-1,-1],
            [1,1,1,1,1,1]
        ])*np.pi/4
        self.phase_offsets = np.array([
            [np.pi/2,-np.pi/2,np.pi/2,-np.pi/2,np.pi/2,-np.pi/2],
            [0,np.pi,0,np.pi,0,np.pi],
            [0,np.pi,0,np.pi,0,np.pi]
        ])
        self.amplitudes = np.tile(np.array([np.pi/8, 2*np.pi/16, np.pi/8])[:,None], (1,6))
        self.t = 0

    def main_loop(self):
        while not rospy.is_shutdown():
            # Wait for joint feedback and topology to be initialized
            if not self.joint_fb_initialized or not self.topology_initialized:
                self.rate.sleep()
                continue

            # Create Joint State
            joint_state = JointState()
            joint_state.header.stamp = rospy.Time.now()

            joint_state.name = []
            joint_state.position = []

            # Check if a velocity command was received
            if abs(self.forward_cmd) > 0 or abs(self.turn_cmd) > 0:
                self.t += self.dt
            else:
                self.t = 0

            # Iterate over limbs
            for i in range(len(self.limb_info)):
                limb_type, joint_id = self.limb_info[i]
                if limb_type == WHEEL_LIMB_TYPE:
                    # TODO wheels have not yet been tested
                    if i<3:
                        cmd = (-self.forward_cmd + self.turn_cmd)
                    else:
                        cmd = (self.forward_cmd + self.turn_cmd)
                    wheel_joint_id = joint_id
                    while (self.topology[wheel_joint_id]['children'][0] in self.topology):
                        wheel_joint_id = self.topology[wheel_joint_id]['children'][0]
                    joint_index = list(self.topology).index(wheel_joint_id)
                    self.joint_positions[joint_index] += cmd*self.max_wheel_velocity*self.dt
                    joint_state.name.append(wheel_joint_id)
                    joint_state.position.append(self.joint_positions[joint_index])

                elif limb_type == LEG_LIMB_TYPE:
                    # Retrieve tripod gait parameters
                    leg_angles_i = self.amplitudes[:,i]*np.sin(self.t + self.phase_offsets[:,i])
                    leg_angles_i[1:] = np.clip(leg_angles_i[1:], -np.inf,0) # convert up-down motion to up-flat motion
                    leg_angles_i += self.const_offsets[:,i]

                    bendy_joint_idx = 0
                    leg_joint_id = joint_id
                    while (leg_joint_id is not None and leg_joint_id in self.topology): # and len(self.topology[leg_joint_id]['children'])): # TODO confirm why first condition is needed on robot and second condition is needed in sim
                        if self.topology[leg_joint_id]['type'] == BENDY_MODULE_TYPE:
                            joint_index = list(self.topology).index(leg_joint_id)
                            # TODO handle orientation of bendy modules in here
                            joint_angle = self.amplitudes[bendy_joint_idx,i]*np.sin(self.t + self.phase_offsets[bendy_joint_idx,i])
                            joint_min = -np.pi/16
                            if bendy_joint_idx == 1:
                                joint_angle = max(joint_angle, joint_min)
                                joint_angle *= -1
                            elif bendy_joint_idx == 2:
                                joint_angle = max(joint_angle, joint_min)
                            else:
                                joint_angle *= 0.5*np.sign(self.forward_cmd) + (0.5*(np.sign(self.turn_cmd) if i >= 3 else -0.5*np.sign(self.turn_cmd)))
                            if (i == 1 or i == 4) and bendy_joint_idx == 0 and self.t < np.pi/2:
                                joint_angle = 0
                            joint_angle += self.const_offsets[bendy_joint_idx,i]
                            joint_state.position.append(joint_angle)
                            joint_state.name.append(leg_joint_id)
                            bendy_joint_idx += 1
                        leg_joint_id = self.topology[leg_joint_id]['children'][0] if len(self.topology[leg_joint_id]['children']) else None
            joint_state.velocity = [float('nan')]*len(joint_state.position)
            joint_state.effort = [float('nan')]*len(joint_state.position)
            self.joint_cmd_pub.publish(joint_state)

            self.rate.sleep()

    
    def joint_fb_callback(self, msg):
        if self.topology_initialized and not self.joint_fb_initialized:
            # TODO: figure out why detached modules persist in joint_fb message
            actual_joints_idx = [i for i,name in enumerate(msg.name) if name in self.joint_names]

            # Initialize current joint positions from feedback
            self.joint_positions = np.array([msg.position[i] for i in actual_joints_idx])
            self.joint_fb_initialized = True
    

    def topology_callback(self,msg):
        json_str = '[' + msg.data[:-1] + ']'
        if json_str == self.topology_str:
            # No need to update topology
            return
        print('Topology Received!')

        self.topology_str = json_str
        topology_data = json.loads(json_str)

        # Reformat topology string into a dictionary indexed by module ID
        # TODO clear self.eigenbody_id and add warning if body not found
        topology_by_id = {}
        for data in topology_data:
            module_id = str(data['id'])
            module_type = int(data['type'], 16)
            try:
                module_orientation = int(data['orientation'], 16)
            except:
                rospy.logwarn('Unable to set orientation: {}'.format(data['orientation']))
                return
            module_children =  [str(child_id) for child_id in data['children']]
            if module_type == EIGENBODY_MODULE_TYPE:
                self.eigenbody_id = module_id
            topology_by_id[module_id] = {
                'type': module_type,
                'orientation': module_orientation,
                'children': module_children
            }
        self.topology = topology_by_id
        print(self.topology)

        # Get limb types
        # TODO verify this works with Eigenbody children.
        # Assumes that # of children >= n_chassis_ports
        for i in range(self.n_chassis_ports):
            child_id = self.topology[self.eigenbody_id]['children'][self.chassis_ports[i]]
            if child_id not in self.topology:
                # TODO may not need this warning - what is expected behavior when there is no limb there?
                rospy.logwarn('Unable to find child: {}'.format(child_id))
                self.limb_info[i] = [None,None]
            else:
                # Find "leaf node" in topology for this limb
                parent_id = child_id
                prev_child_id = child_id
                limb_topology = []
                while child_id in self.topology and len(self.topology[child_id]['children']): # TODO confirm why first condition is needed when not running in sim
                    limb_topology.append(child_id)
                    prev_child_id = child_id
                    child_id = self.topology[child_id]['children'][0]
                limb_topology.append(child_id)
                child_id = prev_child_id
                if self.topology[child_id]['type'] == WHEEL_MODULE_TYPE:
                    self.limb_info[i] = [WHEEL_LIMB_TYPE, parent_id]
                else:
                    self.limb_info[i] = [LEG_LIMB_TYPE, parent_id]

        # TODO delete joint_names here, needs to be appended in order to properly preserve order ignoring eigenbody type
        self.joint_names = [name for name in self.topology if self.topology[name]['type'] != EIGENBODY_MODULE_TYPE]
        self.joint_fb_initialized = False  # Set to False to re-initialize initial positions
        self.topology_initialized = True
    

    def cmd_vel_callback(self, msg):
        self.forward_cmd = msg.linear.x
        # Restrict to only straight line motion or point turns
        if abs(self.forward_cmd) < 1e-3:
            self.turn_cmd = msg.angular.z



if __name__ == '__main__':
    rospy.init_node('eigenbot_joint_controller')
    eigenbot_joint_controller = EigenbotJointController()
    eigenbot_joint_controller.main_loop()    
