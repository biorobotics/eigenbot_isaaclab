#!/usr/bin/env python3
"""Assemble a modular Eigenbot xacro from graph-style topology data."""

import os
import subprocess
import xml.etree.ElementTree as ET


def _paths():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    pkg_dir = os.path.dirname(script_dir)
    repo_dir = os.path.dirname(pkg_dir)
    return {
        "script_dir": script_dir,
        "pkg_dir": pkg_dir,
        "description_dir": os.path.join(pkg_dir, "description"),
        "urdf_dir": os.path.join(pkg_dir, "urdf"),
        "moveit_config_dir": os.path.join(repo_dir, "eigenbot_moveit_config", "config"),
    }


def description_assemble(graph_nodes, graph_edges, graph_nodes_serials):
    print('Assembling with graph_nodes_serials')
    print(graph_nodes_serials)

    paths = _paths()

    input_ports = []
    fnames = []
    output_ports = []
    module_labels = []

    for module_num, module_name in enumerate(graph_nodes):
        tree = ET.parse(os.path.join(paths["description_dir"], module_name + '.xml'))
        root = tree.getroot()

        fname = root.find('filename').text
        input_port = root.find('input_port').text
        ports = root.find('output_ports')
        module_label = 'M' + str(module_num) + '_S' + str(graph_nodes_serials[module_num])

        fnames.append(fname)
        input_ports.append(input_port)
        output_ports.append(ports)
        module_labels.append(module_label)

        print("file: " + fname + ", input port: " + input_port + ', Label: ' + module_label)

    parent_rigid_bodies = []
    child_rigid_bodies = []
    mount_xyz = []
    mount_rpy = []
    for edge in graph_edges:
        from_node, to_node, active_port, active_mount = edge
        port = output_ports[from_node][active_port]

        parent_rigid_body = port.find('parent').text + module_labels[from_node]
        child_rigid_body = input_ports[to_node] + module_labels[to_node]
        mount = port.findall('mount')[active_mount]

        parent_rigid_bodies.append(parent_rigid_body)
        child_rigid_bodies.append(child_rigid_body)
        mount_xyz.append(mount.find('xyz').text)
        mount_rpy.append(mount.find('rpy').text)
        print(
            "attach " + str(child_rigid_body) + " on "
            + str(parent_rigid_body) + " at (" + mount_xyz[-1] + '), (' + mount_rpy[-1] + ")"
        )

    xacro_path = os.path.join(paths["urdf_dir"], 'autoXACRO.xacro')
    with open(xacro_path, 'w', encoding='utf-8') as handle:
        handle.write('<?xml version="1.0"?>\n')
        handle.write('<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="modular_robot_compiled">\n\n')
        handle.write('<xacro:property name="M_PI" value="3.1415926535897931" />\n')
        for module_num, module_name in enumerate(graph_nodes):
            handle.write('<!-- Include and instantiate module type ' + module_name + ' -->\n')
            fname = fnames[module_num]
            handle.write('<xacro:include filename="../urdf/' + fname + '"/>\n')
            xacro_name = fname[:fname.find('.')]
            handle.write('<xacro:' + xacro_name + ' module_label="' + module_labels[module_num] + '"/>\n\n')

        for edge_num in range(len(graph_edges)):
            handle.write('<joint name="connection_' + str(edge_num) + '_attachment" type="fixed">\n')
            handle.write('  <origin\n')
            handle.write('    xyz="' + mount_xyz[edge_num] + '"\n')
            handle.write('    rpy="' + mount_rpy[edge_num] + '"\n')
            handle.write('  />\n')
            handle.write('  <parent link="' + parent_rigid_bodies[edge_num] + '"/>\n')
            handle.write('  <child link="' + child_rigid_bodies[edge_num] + '"/>\n')
            handle.write('</joint>\n\n')

        handle.write('</robot>\n')

    urdf_path = os.path.join(paths["urdf_dir"], 'autoXACRO.urdf')
    subprocess.Popen(
        ['rosrun', 'xacro', 'xacro', '--inorder', '-o', urdf_path, xacro_path],
        cwd='/',
    ).wait()

    urdf_tree = ET.parse(urdf_path)
    joint_list = urdf_tree.getroot().findall('joint')
    moving_joint_list = []
    for joint in joint_list:
        if joint.get('type') in ('continuous', 'revolute', 'prismatic'):
            moving_joint_list.append(joint.get('name'))

    print("Joints found: " + str(moving_joint_list))
