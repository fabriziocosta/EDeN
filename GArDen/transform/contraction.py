#!/usr/bin/env python
"""Provides contraction."""

from sklearn.base import BaseEstimator, TransformerMixin
import networkx as nx
from collections import Counter, namedtuple, defaultdict
from eden.util import _serialize_list

import logging
logger = logging.getLogger(__name__)


def contraction_histogram(input_attribute=None, graph=None, id_nodes=None):
    """contraction_histogram."""
    labels = [graph.node[v].get(input_attribute, 'N/A') for v in id_nodes]
    dict_label = dict(Counter(labels).most_common())
    sparse_vec = {str(key): value for key, value in dict_label.iteritems()}
    return sparse_vec


def contraction_sum(input_attribute=None, graph=None, id_nodes=None):
    """contraction_sum."""
    vals = [float(graph.node[v].get(input_attribute, 1)) for v in id_nodes]
    return sum(vals)


def contraction_average(input_attribute=None, graph=None, id_nodes=None):
    """contraction_average."""
    vals = [float(graph.node[v].get(input_attribute, 1)) for v in id_nodes]
    return sum(vals) / float(len(vals))


def contraction_categorical(input_attribute=None,
                            graph=None,
                            id_nodes=None,
                            separator='.'):
    """contraction_categorical."""
    vals = sorted([_serialize_list(graph.node[v].get(input_attribute, 'N/A'))
                   for v in id_nodes])
    return _serialize_list(vals, separator=separator)


def contraction_set_categorical(input_attribute=None,
                                graph=None,
                                id_nodes=None,
                                separator='.'):
    """contraction_set_categorical."""
    vals = sorted(set(
        [_serialize_list(graph.node[v].get(input_attribute, 'N/A'))
         for v in id_nodes]))
    return _serialize_list(vals, separator=separator)


def serialize_modifiers(modifiers):
    """serialize_modifiers."""
    lines = ""
    if modifiers:
        for modifier in modifiers:
            line = "attribute_in:%s attribute_out:%s reduction:%s" % \
                (modifier.attribute_in,
                 modifier.attribute_out,
                 modifier.reduction)
            lines += line + "\n"
    return lines

contraction_modifer_map = {'histogram': contraction_histogram,
                           'sum': contraction_sum,
                           'average': contraction_average,
                           'categorical': contraction_categorical,
                           'set_categorical': contraction_set_categorical}
contraction_modifier = namedtuple('contraction_modifier',
                                  'attribute_in attribute_out reduction')
label_modifier = contraction_modifier(attribute_in='type',
                                      attribute_out='label',
                                      reduction='set_categorical')
weight_modifier = contraction_modifier(attribute_in='weight',
                                       attribute_out='weight',
                                       reduction='sum')
# modifiers = [label_modifier, weight_modifier]

# ------------------------------------------------------------------------------


class Contract(BaseEstimator, TransformerMixin):
    """Contract."""

    def __init__(self,
                 contraction_attribute='label',
                 nesting=False,
                 original_edges_to_nesting=False,
                 weight_scaling_factor=1,
                 modifiers=None):
        """Constructor."""
        self.contraction_attribute = contraction_attribute
        self.nesting = nesting
        self.original_edges_to_nesting = original_edges_to_nesting
        self.weight_scaling_factor = weight_scaling_factor
        self.modifiers = modifiers

    def transform(self, graphs=None):
        """transform."""
        try:
            for g in graphs:
                for graph_contracted in self._transform(g):
                    yield graph_contracted
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _transform(self, g):
        # check for 'position' attribute and add it if not present
        for i, (n, d) in enumerate(g.nodes_iter(data=True)):
            if d.get('position', None) is None:
                g.node[n]['position'] = i
        # compute contraction
        g_contracted = self._edge_contraction(graph=g)
        info = g_contracted.graph.get('info', '')
        g_contracted.graph['info'] = info + '\n' + \
            serialize_modifiers(self.modifiers)
        for n, d in g_contracted.nodes_iter(data=True):
            # get list of contracted node ids
            contracted = d.get('contracted', None)
            if contracted is None:
                raise Exception(
                    'Empty contraction list for: id %d data: %s' %
                    (n, d))
            for modifier in self.modifiers:
                modifier_func = contraction_modifer_map[modifier.reduction]
                g_contracted.node[n][modifier.attribute_out] = modifier_func(
                    input_attribute=modifier.attribute_in,
                    graph=g,
                    id_nodes=contracted)
            # rescale the weight of the contracted nodes
            if self.weight_scaling_factor != 1:
                w = d.get('weight', 1)
                w = w * self.weight_scaling_factor
                g_contracted.node[n]['weight'] = w
        # add nesting edges between the contraction graph and original graph
        if self.nesting:
            if self.original_edges_to_nesting:
                for u, v in g.edges():
                    g.edge[u][v]['nesting'] = True
            g_nested = nx.disjoint_union(g, g_contracted)
            # rewire contracted graph to the original graph
            for n, d in g_nested.nodes_iter(data=True):
                contracted = d.get('contracted', None)
                if contracted:
                    for m in contracted:
                        g_nested.add_edge(n, m,
                                          label='.', len=.1, nesting=True)
            yield g_nested
        else:
            yield g_contracted

    def _edge_contraction(self, graph=None):
        g = graph.copy()
        # add a 'contracted' attribute in each node
        for n, d in g.nodes_iter(data=True):
            g.node[n]['contracted'] = set()
            # add the node itself to its contraction list
            g.node[n]['contracted'].add(n)
        # iterate until contractions are possible, marked by flag:
        # change_has_occured
        # Note: the order of the contraction operations is irrelevant
        while True:
            change_has_occured = False
            for n, d in g.nodes_iter(data=True):
                g.node[n]['label'] = g.node[n][self.contraction_attribute]
                if self.contraction_attribute in d and 'position' in d:
                    neighbors = g.neighbors(n)
                    if len(neighbors) > 0:
                        # identify neighbors that have a greater 'position'
                        # attribute and that have the same
                        # self.contraction_attribute
                        greater_position_neighbors = [v for v in neighbors if 'position' in g.node[v] and
                                                      self.contraction_attribute in g.node[v] and
                                                      g.node[v][self.contraction_attribute] == d[self.contraction_attribute] and
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

# ------------------------------------------------------------------------------


class Minor(BaseEstimator, TransformerMixin):
    """Minor."""

    def __init__(self,
                 part_id='part_id',
                 part_name='part_name',
                 nesting=False,
                 weight_scaling_factor=1,
                 modifiers=None):
        """Constructor."""
        self.part_id = part_id
        self.part_name = part_name
        self.nesting = nesting
        self.weight_scaling_factor = weight_scaling_factor
        self.modifiers = modifiers

    def transform(self, graphs=None):
        """Transform."""
        try:
            for graph in graphs:
                for minor_graph in self._transform(graph):
                    yield minor_graph
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _transform(self, graph):
        # contract all nodes that have the same value for the part_id
        minor_graph = self._minor(graph)
        info = minor_graph.graph.get('info', '')
        minor_graph.graph['info'] = info + '\n' + \
            serialize_modifiers(self.modifiers)
        for n, d in minor_graph.nodes_iter(data=True):
            # get list of contracted node ids
            contracted = d.get('contracted', None)
            if contracted is None:
                raise Exception('Empty contraction list for: id %d data: %s' %
                                (n, d))
            for modifier in self.modifiers:
                modifier_func = contraction_modifer_map[modifier.reduction]
                minor_graph.node[n][modifier.attribute_out] = modifier_func(
                    input_attribute=modifier.attribute_in,
                    graph=graph, id_nodes=contracted)
            # rescale the weight of the contracted nodes
            if self.weight_scaling_factor != 1:
                w = d.get('weight', 1)
                w = w * self.weight_scaling_factor
                minor_graph.node[n]['weight'] = w

        # build nesting graph
        if self.nesting:
            # add nesting edges between the minor graph and the original graph
            g_nested = nx.disjoint_union(graph, minor_graph)
            g_nested.graph.update(graph.graph)
            # rewire contracted graph to the original graph
            for n, d in g_nested.nodes_iter(data=True):
                contracted = d.get('contracted', None)
                if contracted:
                    for m in contracted:
                        g_nested.add_edge(n, m,
                                          label='.', len=.1, nesting=True)
            yield g_nested
        else:
            yield minor_graph

    def _minor(self, graph):
        # contract all nodes that have the same value for the part_id
        # store the contracted nodes original ids for later reference

        # find all values for the part_id and create a dict with
        # key=part_id and value=node ids
        part_name_dict = dict()
        part_id_dict = defaultdict(list)
        for u in graph.nodes_iter():
            if self.part_id in graph.node[u]:
                part_ids = graph.node[u][self.part_id]
                part_names = graph.node[u][self.part_name]
                for part_id, part_name in zip(part_ids, part_names):
                    part_id_dict[part_id].append(u)
                    part_name_dict[part_id] = part_name
        m_graph = nx.Graph()
        # copy attributes
        m_graph.graph = graph.graph
        # allocate a node per part_id
        for part_id in part_id_dict:
            m_graph.add_node(part_id,
                             is_contracted=True,
                             contracted=part_id_dict[part_id])
            m_graph.node[part_id][self.part_id] = part_id
            m_graph.node[part_id][self.part_name] = part_name_dict[part_id]
            m_graph.node[part_id]['label'] = part_name_dict[part_id]
        # create an edge between two part_id nodes if there existed such an
        # edge between two nodes that had that part_id
        for u, v in graph.edges():
            if self.part_id in graph.node[u] and self.part_id in graph.node[v]:
                p_id_us = graph.node[u][self.part_id]
                p_id_vs = graph.node[v][self.part_id]
                for p_id_u in p_id_us:
                    for p_id_v in p_id_vs:
                        if p_id_u != p_id_v:
                            if (p_id_u, p_id_v) not in m_graph.edges():
                                m_graph.add_edge(p_id_u, p_id_v)
                                m_graph.edge[p_id_u][p_id_v]['label'] = 'part_of'
        return m_graph
