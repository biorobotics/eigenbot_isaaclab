import json
import re
import numpy as np


def parse_topology(string_in):
    print('---- Raw data: ----')
    print(string_in)
    print('--------')

    json_starts = [m.start() for m in re.finditer('{', string_in)]
    json_ends = [m.start() for m in re.finditer('}', string_in)]
    json_in = []
    for i in range(len(json_starts)):
        module_json = string_in[json_starts[i]:json_ends[i] + 1]
        json_in.append(json.loads(module_json))

    module_ids = []
    module_types = []
    module_orientations = []
    module_children_ids = []
    for module in json_in:
        if module['id'] in module_ids:
            continue
        module_ids.append(module['id'])
        module_types.append(int(module['type'], 16))
        module_orientations.append(int(module['orientation']) - 1)
        module_children_ids.append(module['children'])

    for index in range(len(module_orientations)):
        if module_orientations[index] < 0:
            module_orientations[index] = 0

    print('module_ids ' + str(len(module_ids)) + ':', module_ids)
    print('module_types' + str(len(module_types)) + ':', module_types)
    print('module_orientations ' + str(len(module_orientations)) + ':', module_orientations)
    print('module_children_ids ' + str(len(module_children_ids)) + ':', module_children_ids)

    module_attachments = []
    graph_edges = []
    for parent_index, children_ids in enumerate(module_children_ids):
        print('Module ' + str(parent_index) + ' children serials: ' + str(children_ids))
        module_attachments_now = []
        for port_num, child_id in enumerate(children_ids):
            if child_id == 'FF':
                continue
            index_found = module_ids.index(child_id) if child_id in module_ids else None
            print(
                'Found module index ' + str(index_found)
                + ' (id: ' + str(module_ids[index_found])
                + ' orn: ' + str(module_orientations[index_found])
                + ') on port ' + str(port_num)
            )
            graph_edges.append([parent_index, index_found, port_num, module_orientations[index_found]])
            module_attachments_now.append([port_num, module_orientations[index_found], index_found])
        module_attachments.append(module_attachments_now)

    node_type_enumeration = [
        'Null',
        'Wheel_module',
        'Torsional_module',
        'Bendy_module',
        'Gripper_foot',
        'Gripper_module',
        'O=6 module',
        'Battery',
        'Eigenbody',
        'TeeSplitter',
        'Foot_module',
        'Static straight',
        'Static_45_deg',
        'Static_90deg_module',
        'Eigenbody',
    ]

    module_types_str = [node_type_enumeration[module_type] for module_type in module_types]
    print(module_types_str)
    print('edges:')
    print(graph_edges)

    return module_types_str, graph_edges, module_ids


def compare_topologies(graph_edges1, module_ids1, graph_edges2, module_ids2):
    same_graph = True
    if not (len(module_ids1) == len(module_ids2) and len(graph_edges1) == len(graph_edges2)):
        same_graph = False

    for module_id in module_ids1:
        if module_id not in module_ids2:
            same_graph = False

    if same_graph:
        edges_found = [False] * len(graph_edges1)
        for i in range(len(graph_edges1)):
            edge1 = graph_edges1[i]
            parent1, child1, port1, orn1 = edge1
            parent_serial1 = module_ids1[parent1]
            child_serial1 = module_ids1[child1]
            for edge2 in graph_edges2:
                parent2, child2, port2, orn2 = edge2
                parent_serial2 = module_ids2[parent2]
                child_serial2 = module_ids2[child2]
                if (
                    parent_serial1 == parent_serial2
                    and child_serial1 == child_serial2
                    and port1 == port2
                    and orn1 == orn2
                ):
                    edges_found[i] = True
        if not np.all(edges_found):
            same_graph = False

    return same_graph
