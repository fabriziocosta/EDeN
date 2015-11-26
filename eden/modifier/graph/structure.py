import networkx as nx
from collections import Counter, namedtuple


def delete_edge_type(graphs, edge_type_key='basepair'):
    for graph in graphs:
        for edge_src, edge_dest, edge_dat in graph.edges_iter(data=True):
            if edge_dat.get('type', None) == edge_type_key:
                graph.remove_edge(edge_src, edge_dest)
        yield graph


def make_edge_type_into_nesting(graphs, edge_type_key='basepair'):
    for graph in graphs:
        for edge_src, edge_dest, edge_dat in graph.edges_iter(data=True):
            if edge_dat.get('type', None) == edge_type_key:
                edge_dat['nesting'] = True
                graph.add_edge(edge_src, edge_dest, edge_dat)
        yield graph


def edge_contraction(graph=None, node_attribute=None):
    g = graph.copy()
    # add a 'contracted' attribute in each node
    for n, d in g.nodes_iter(data=True):
        g.node[n]['contracted'] = set()
        # add the node itself to its contraction list
        g.node[n]['contracted'].add(n)
    # iterate until contractions are possible, marked by flag: change_has_occured
    # Note: the order of the contraction operations is irrelevant
    while True:
        change_has_occured = False
        for n, d in g.nodes_iter(data=True):
            g.node[n]['label'] = g.node[n][node_attribute]
            if node_attribute in d and 'position' in d:
                neighbors = g.neighbors(n)
                if len(neighbors) > 0:
                    # identify neighbors that have a greater 'position' attribute and that have
                    # the same node_attribute
                    greater_position_neighbors = [v for v in neighbors if 'position' in g.node[v] and
                                                  node_attribute in g.node[v] and
                                                  g.node[v][node_attribute] == d[node_attribute] and
                                                  g.node[v]['position'] > d['position']]
                    if len(greater_position_neighbors) > 0:
                        # contract all neighbors
                        # replicate all edges with n as endpoint instead of v
                        # i.e. move the endpoint of all edges ending in v to n
                        cntr_edge_set = g.edges(greater_position_neighbors, data=True)
                        new_edges = map(lambda x: (n, x[1], x[2]), cntr_edge_set)
                        # we are going to remove the greater pos neighbors , so we better make sure not to
                        # loose their contracted sets.
                        gpn_contracted = set([removed_node for greater_position_node in
                                              greater_position_neighbors for removed_node in g.node[
                                                  greater_position_node]['contracted']])

                        # remove nodes
                        g.remove_nodes_from(greater_position_neighbors)
                        # remove edges
                        g.remove_edges_from(cntr_edge_set)
                        # add edges only if endpoint nodes still exist and they are not self loops
                        new_valid_edges = [e for e in new_edges if e[1] in g.nodes() and e[1] != n]
                        g.add_edges_from(new_valid_edges)
                        # store neighbor ids in the contracted list
                        g.node[n]['contracted'].update(gpn_contracted)
                        change_has_occured = True
                        break
        if change_has_occured is False:
            break
    return g


def contraction_histogram(input_attribute=None, graph=None, id_nodes=None):
    labels = [graph.node[v].get(input_attribute, 'N/A') for v in id_nodes]
    dict_label = dict(Counter(labels).most_common())
    sparse_vec = {str(key): value for key, value in dict_label.iteritems()}
    return sparse_vec


def contraction_sum(input_attribute=None, graph=None, id_nodes=None):
    vals = [float(graph.node[v].get(input_attribute, 1)) for v in id_nodes]
    return sum(vals)


def contraction_average(input_attribute=None, graph=None, id_nodes=None):
    vals = [float(graph.node[v].get(input_attribute, 0)) for v in id_nodes]
    return sum(vals) / float(len(vals))


def contraction_categorical(input_attribute=None, graph=None, id_nodes=None, separator='.'):
    vals = sorted([str(graph.node[v].get(input_attribute, 'N/A')) for v in id_nodes])
    return separator.join(vals)


def contraction_set_categorical(input_attribute=None, graph=None, id_nodes=None, separator='.'):
    vals = sorted(set([str(graph.node[v].get(input_attribute, 'N/A')) for v in id_nodes]))
    return separator.join(vals)


contraction_modifer_map = {'histogram': contraction_histogram,
                           'sum': contraction_sum,
                           'average': contraction_average,
                           'categorical': contraction_categorical,
                           'set_categorical': contraction_set_categorical}
contraction_modifier = namedtuple('contraction_modifier', 'attribute_in attribute_out reduction')
label_modifier = contraction_modifier(attribute_in='type', attribute_out='label', reduction='set_categorical')
weight_modifier = contraction_modifier(attribute_in='weight', attribute_out='weight', reduction='sum')
modifiers = [label_modifier, weight_modifier]


def serialize_modifiers(modifiers):
    lines = ""
    for modifier in modifiers:
        line = "attribute_in:%s attribute_out:%s reduction:%s" % (modifier.attribute_in,
                                                                  modifier.attribute_out,
                                                                  modifier.reduction)
        lines += line + "\n"
    return lines


def contraction(graphs=None, contraction_attribute='label', nesting=False, modifiers=modifiers, **options):
    '''
    modifiers: list of named tuples, each containing the keys: attribute_in, attribute_out and reduction.
    "attribute_in" identifies the node attribute that is extracted from all contracted nodes.
    "attribute_out" identifies the node attribute that is written in the resulting graph.
    "reduction" is one of the following reduction operations:
    1. histogram,
    2. sum,
    3. average,
    4. categorical,
    5. set_categorical.
    "histogram" returns a sparse vector with numerical hashed keys,
    "sum" and "average" cast the values into floats before computing the sum and average respectively,
    "categorical" returns the concatenation string of the lexicographically sorted list of input attributes,
    "set_categorical" returns the concatenation string of the lexicographically sorted set of input
    attributes.
    '''
    for g in graphs:
        # check for 'position' attribute and add it if not present
        for i, (n, d) in enumerate(g.nodes_iter(data=True)):
            if d.get('position', None) is None:
                g.node[n]['position'] = i
        # compute contraction
        g_contracted = edge_contraction(graph=g, node_attribute=contraction_attribute)
        info = g_contracted.graph.get('info', '')
        g_contracted.graph['info'] = info + '\n' + serialize_modifiers(modifiers)
        for n, d in g_contracted.nodes_iter(data=True):
            # get list of contracted node ids
            contracted = d.get('contracted', None)
            if contracted is None:
                raise Exception('Empty contraction list for: id %d data: %s' % (n, d))
            for modifier in modifiers:
                modifier_func = contraction_modifer_map[modifier.reduction]
                g_contracted.node[n][modifier.attribute_out] = modifier_func(
                    input_attribute=modifier.attribute_in, graph=g, id_nodes=contracted)
        if nesting:  # add nesting edges between the constraction graph and the original graph
            g_nested = nx.disjoint_union(g, g_contracted)
            # rewire contracted graph to the original graph
            for n, d in g_nested.nodes_iter(data=True):
                contracted = d.get('contracted', None)
                if contracted:
                    for m in contracted:
                        g_nested.add_edge(n, m, label='.', len=1, nesting=True)
            yield g_nested
        else:
            yield g_contracted
