#!/usr/bin/env python
"""Provides modification of node attributes."""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from itertools import combinations_with_replacement

import logging
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------


class ReplaceWithAllCombinations(BaseEstimator, TransformerMixin):
    """ReplaceWithAllCombinations."""

    def __init__(self,
                 label_list=['A', 'C', 'G', 'U'],
                 attribute='selected'):
        """Construct."""
        self.label_list = label_list
        self.attribute = attribute

    def transform(self, graphs):
        """transform."""
        try:
            for graph in graphs:
                transformed_graphs = self._replace_with_all_combinations(graph)
                for transformed_graph in transformed_graphs:
                    yield transformed_graph
            pass
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _replace_with_all_combinations(self, graph):
        # find nodes with attribute
        kmer_pos = []
        ref_kmer = []
        for u in graph.nodes():
            if graph.node[u].get(self.attribute, False):
                kmer_pos.append(u)
                ref_kmer.append(graph.node[u]['label'])
        ref_kmer = tuple(ref_kmer)
        kmers = combinations_with_replacement(self.label_list, len(kmer_pos))
        for kmer in kmers:
            if kmer != ref_kmer:
                new_graph = graph.copy()
                new_graph.graph['ref_kmer'] = ref_kmer
                new_graph.graph['kmer_pos'] = kmer_pos
                new_graph.graph['kmer'] = kmer
                new_graph.graph['header'] += '_' + ''.join(kmer) +\
                    '-' + ''.join(ref_kmer)
                for i in range(len(kmer)):
                    pos = kmer_pos[i]
                    new_label = kmer[i]
                    new_graph.node[pos]['label'] = new_label
                yield new_graph


# ------------------------------------------------------------------------------


class MarkKTop(BaseEstimator, TransformerMixin):
    """MarkKTop."""

    def __init__(self,
                 mark_attribute='selected',
                 attribute='importance',
                 ktop=1,
                 reverse=True):
        """Construct."""
        self.mark_attribute = mark_attribute
        self.attribute = attribute
        self.ktop = ktop
        self.reverse = reverse

    def transform(self, graphs):
        """transform."""
        try:
            for graph in graphs:
                transformed_graph = self._mark(graph)
                yield transformed_graph
            pass
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _mark(self, graph):
        # iterate over nodes and sort values of attribute
        values = []
        for n in graph.nodes():
            value = float(graph.node[n].get(self.attribute, 0))
            values.append(value)
        sorted_values = sorted(values, reverse=self.reverse)

        if self.ktop < len(sorted_values):
            threshold = sorted_values[self.ktop]
        else:
            threshold = sorted_values[-1]
        # iterate over nodes and mark
        for n in graph.nodes():
            value = float(graph.node[n].get(self.attribute, 0))
            if value > threshold:
                graph.node[n][self.mark_attribute] = True
            else:
                graph.node[n][self.mark_attribute] = False
        return graph


# ------------------------------------------------------------------------------

class ColorNode(BaseEstimator, TransformerMixin):
    """ColorNode."""

    def __init__(self, input_attribute='label',
                 output_attribute='level',
                 labels=['A', 'U', 'C', 'G']):
        """Construct."""
        self.input_attribute = input_attribute
        self.output_attribute = output_attribute
        self.labels = labels
        self.color_dict = None

    def transform(self, graphs):
        """transform."""
        values = np.linspace(0.0, 1.0, num=len(self.labels))
        self.color_dict = dict(zip(self.labels, values))

        try:
            for graph in graphs:
                yield self._colorize(graph=graph)
            pass
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _colorize(self, graph=None):
        # iterate over nodes
        for n, d in graph.nodes_iter(data=True):
            graph.node[n][self.output_attribute] = \
                self.color_dict.get(d[self.input_attribute], 0)
        return graph

# ------------------------------------------------------------------------------


class AddGraphAttributeValue(BaseEstimator, TransformerMixin):
    """AddGraphAttributeValue."""

    def __init__(self, attribute=None, value=None):
        """Construct."""
        self.attribute = attribute
        self.value = value

    def transform(self, graphs):
        """TODO."""
        try:
            for graph in graphs:
                graph.graph[self.attribute] = self.value
                yield graph
            pass
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

# ------------------------------------------------------------------------------


class AddNodeAttributeValue(BaseEstimator, TransformerMixin):
    """AddNodeAttributeValue."""

    def __init__(self, attribute=None, value=None):
        """Construct."""
        self.attribute = attribute
        self.value = value

    def transform(self, graphs):
        """TODO."""
        try:
            for graph in graphs:
                # iterate over nodes
                for n in graph.nodes():
                    graph.node[n][self.attribute] = self.value
                yield graph
            pass
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)


# ------------------------------------------------------------------------------


class MultiplicativeReweightDictionary(BaseEstimator, TransformerMixin):
    """MultiplicativeReweightDictionary.

    Multiply weights according to attribute,value pairs matching.
    The weight_dict is structured as a dict of attribute strings associated
    to a dictionary of values types associated to a weight number.
    """

    def __init__(self, weight_dict=None):
        """Construct."""
        self.weight_dict = weight_dict

    def transform(self, graphs):
        """Transform."""
        try:
            for graph in graphs:
                # iterate over nodes
                for n, d in graph.nodes_iter(data=True):
                    for attribute in self.weight_dict:
                        if attribute in d:
                            if d[attribute] in self.weight_dict[attribute]:
                                graph.node[n]['weight'] = \
                                    graph.node[n]['weight'] * \
                                    self.weight_dict[attribute][d[attribute]]
                yield graph
            pass
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)


# ------------------------------------------------------------------------------

class WeightWithIntervals(BaseEstimator, TransformerMixin):
    """WeightWithIntervals.

    Assign weights according to a list of triplets: each triplet defines
    the start, end position and the (uniform) weight of the region; the
    order of the triplets matters: later triplets override the weight
    specification of previous triplets. The special triplet (-1,-1,w)
    assign a default weight to all nodes.

    If listof_start_end_weight_list is available then each element in
    listof_start_end_weight_list specifies the start_end_weight_list for a
    single graph.
    """

    def __init__(self,
                 attribute='weight',
                 start_end_weight_list=None):
        """Construct."""
        self.attribute = attribute
        self.start_end_weight_list = start_end_weight_list

    def transform(self, graphs):
        """Transform."""
        try:
            for graph in graphs:
                graph = self._start_end_weight_reweight(graph)
                yield graph
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _start_end_weight_reweight(self, graph):
        for start, end, weight in self.start_end_weight_list:
            # iterate over nodes
            for n, d in graph.nodes_iter(data=True):
                if 'position' not in d:
                    # assert nodes must have position attribute
                    raise Exception('Nodes must have "position" attribute')
                # given the 'position' attribute of node assign the weight
                # accordingly
                pos = d['position']
                if pos >= start and pos < end:
                    graph.node[n][self.attribute] = weight
                if start == -1 and end == -1:
                    graph.node[n][self.attribute] = weight
        return graph

# ------------------------------------------------------------------------------


class RelabelWithLabelOfIncidentEdges(BaseEstimator, TransformerMixin):
    """RelabelWithLabelOfIncidentEdges.

    Delete an edge if its dictionary has a key equal to 'attribute' and the
    'condition' is true between 'value' and the value associated to
    key=attribute.
    """

    def __init__(self, output_attribute='type', separator='', distance=1):
        """"Construct.

        Parameters
        ----------
        graphs : iterator over path graphs of RNA sequences

        output_attribute : string
            The key of the node dictionary where to write the result.

        separator : string
            The string used to separate the sorted concatenation of labels.

        distance : integer (default 1)
            The neighborhood radius explored.
        """
        self.output_attribute = output_attribute
        self.separator = separator
        self.distance = distance

    def transform(self, graphs):
        """Transform."""
        try:
            for graph in graphs:
                # iterate over nodes
                for n, d in graph.nodes_iter(data=True):
                    # for all neighbors
                    edge_labels = []
                    if self.distance == 1:
                        edge_labels += [graph.edge[u][v].get('label', '-')
                                        for u, v in graph.edges_iter(n)]
                    elif self.distance == 2:
                        neighbors = graph.neighbors(n)
                        for nn in neighbors:
                            # extract list of edge labels
                            edge_labels += [graph.edge[u][v].get('label', '-')
                                            for u, v in graph.edges_iter(nn)]
                    else:
                        raise Exception('Unknown distance: %s' % self.distance)
                    # consider the sorted serialization of all labels as a type
                    vertex_type = self.separator.join(sorted(edge_labels))
                    graph.node[n][self.output_attribute] = vertex_type
                yield graph
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)
