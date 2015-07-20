import numpy as np
from itertools import izip


def incident_edge_label(graph_list=None, output_attribute='type', separator='', level=1):
    """
    level: int
    level=1 considers all incident edges
    level=2 considers all edges incident on the neighbors
    """
    for g in graph_list:
        # iterate over nodes
        for n, d in g.nodes_iter(data=True):
            # for all neighbors
            edge_labels = []
            if level == 1:
                edge_labels += [ed.get('label', 'N/A') for u, v, ed in g.edges_iter(n, data=True)]
            elif level == 2:
                neighbors = g.neighbors(n)
                for nn in neighbors:
                    # extract list of edge labels
                    edge_labels += [ed.get('label', 'N/A') for u, v, ed in g.edges_iter(nn, data=True)]
            else:
                raise Exception('Unknown level: %s' % level)
            # consider the sorted serialization of all labels as a type
            vertex_type = separator.join(sorted(edge_labels))
            g.node[n][output_attribute] = vertex_type
        yield g


def incident_node_label(graph_list=None, output_attribute='type', separator='', level=1):
    """
    level: int
    level=1 considers all incident nodes
    level=2 considers all nodes incident on the neighbors
    """
    for g in graph_list:
        # iterate over nodes
        for n, d in g.nodes_iter(data=True):
            # for all neighbors
            node_labels = []
            if level == 1:
                node_labels += [g.node[u].get('label', 'N/A') for u in g.neighbors(n)]
            elif level == 2:
                neighbors = g.neighbors(n)
                for nn in neighbors:
                    # extract list of labels
                    node_labels += [g.node[u].get('label', 'N/A') for u in g.neighbors(nn)]
            else:
                raise Exception('Unknown level: %s' % level)
            # consider the sorted serialization of all labels as a type
            vertex_type = separator.join(sorted(node_labels))
            g.node[n][output_attribute] = vertex_type
        yield g


def translate(graph_list=None,
              input_attribute='label',
              output_attribute='label',
              label_map=dict(),
              default=' '):
    original_attribute = input_attribute + '_original'
    for g in graph_list:
        # iterate over nodes
        for n, d in g.nodes_iter(data=True):
            key = d.get(input_attribute, default)
            g.node[n][original_attribute] = key
            mapped_attribute = label_map.get(key, default)
            g.node[n][output_attribute] = mapped_attribute
        yield g


def colorize(graph_list=None, output_attribute='level', labels=['A', 'U', 'C', 'G'], mode=None):
    values = np.linspace(0.0, 1.0, num=len(labels))
    color_dict = dict(zip(labels, values))
    for g in graph_list:
        # iterate over nodes
        for n, d in g.nodes_iter(data=True):
            if mode == "3D":
                g.node[n][output_attribute] = color_dict.get(d['text_label'], 0)
            else:
                g.node[n][output_attribute] = color_dict.get(d['label'], 0)
        yield g


def colorize_binary(graph_list=None, output_attribute='color_value', input_attribute='weight', level=0):
    for g in graph_list:
        # iterate over nodes
        for n, d in g.nodes_iter(data=True):
            val = d.get(input_attribute, 0)
            if val <= level:
                color_value = 0
            else:
                color_value = 1
            g.node[n][output_attribute] = color_value
        yield g


def discretize(graph_list=None, output_attribute='value', input_attribute='weight', interval=0.1):
    for g in graph_list:
        # iterate over nodes
        for n, d in g.nodes_iter(data=True):
            val = d.get(input_attribute, 0)
            g.node[n][output_attribute] = int(val / interval)
        yield g


def trapezoidal_reweighting(graph_list=None,
                            high_weight=1.0,
                            low_weight=0.1,
                            high_weight_window_start=0,
                            high_weight_window_end=1,
                            low_weight_window_start=0,
                            low_weight_window_end=1):
    """
    Piece wise linear weight function between two levels with specified start end positions.
    high   ___
    low __/   \__
    """
# assert high_ weight > low_weight
    if high_weight < low_weight:
        raise Exception('high_weight (%f) must be higher than low_weight (%f)' % (high_weight, low_weight))

    # assert low_weight boundaries includes high_weight boundaries
    if high_weight_window_start > low_weight_window_end:
        raise Exception('high_weight_window_start (%d) must be lower than low_weight_window_end (%d)' %
                        (high_weight_window_start, low_weight_window_end))
    if high_weight_window_start < low_weight_window_start:
        raise Exception('high_weight_window_start (%d) must be higher than low_weight_window_start (%d)' %
                        (high_weight_window_start, low_weight_window_start))
    if high_weight_window_end < low_weight_window_start:
        raise Exception('high_weight_window_end (%d) must be higher than low_weight_window_start (%d)' %
                        (high_weight_window_end, low_weight_window_start))
    if high_weight_window_end > low_weight_window_end:
        raise Exception('high_weight_window_end (%d) must be higher than low_weight_window_end (%d)' %
                        (high_weight_window_end, low_weight_window_end))

    for g in graph_list:
        # iterate over nodes
        for n, d in g.nodes_iter(data=True):
            if 'position' not in d:
                # assert nodes must have position attribute
                raise Exception('Nodes must have "position" attribute')
            # given the 'position' attribute of node assign weight according to
            # piece wise linear weight function between two levels
            pos = d['position']
            if pos < low_weight_window_start:
                """
                   ___
                __/   \__
                |
                """
                g.node[n]["weight"] = low_weight
            elif pos >= low_weight_window_start and pos < high_weight_window_start:
                """
                   ___
                __/   \__
                  |
                """
                g.node[n]["weight"] = (high_weight - low_weight) / (high_weight_window_start - low_weight_window_start) * \
                    (pos - low_weight_window_start) + low_weight
            elif pos >= high_weight_window_start and pos < high_weight_window_end:
                """
                   ___
                __/   \__
                    |
                """
                g.node[n]["weight"] = high_weight
            elif pos >= high_weight_window_end and pos < low_weight_window_end:
                """
                   ___
                __/   \__
                      |
                """
                g.node[n]["weight"] = high_weight - \
                    (high_weight - low_weight) / (low_weight_window_end - high_weight_window_end) * \
                    (pos - high_weight_window_end)
            else:
                """
                   ___
                __/   \__
                        |
                """
                g.node[n]["weight"] = low_weight
        yield g


def reweight(graph_list, weight_vector_list):
    """Assigns a value to the weight attribute of each node in each graph according to
    the information supplied in the list of vectors."""

    for g, w in izip(graph_list, weight_vector_list):
        # iterate over nodes
        for n, d in g.nodes_iter(data=True):
            if 'position' not in d:
                # assert nodes must have position attribute
                raise Exception('Nodes must have "position" attribute')
            # given the 'position' attribute of node assign the weight accordingly
            pos = d['position']
            g.node[n]["weight"] = w[pos]
        yield g
