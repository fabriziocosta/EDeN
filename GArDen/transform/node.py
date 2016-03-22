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
                    '-' + ''.join(ref_kmer) + '_' + \
                    ':'.join([str(val) for val in kmer_pos])
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
                 exclude_attribute='exclude',
                 ktop=1,
                 reverse=True):
        """Construct."""
        self.mark_attribute = mark_attribute
        self.exclude_attribute = exclude_attribute
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
            exclude = graph.node[n].get(self.exclude_attribute, False)
            if exclude is False:
                values.append(value)
        sorted_values = sorted(values, reverse=self.reverse)

        if self.ktop < len(sorted_values):
            threshold = sorted_values[self.ktop]
        else:
            threshold = sorted_values[-1]
        # iterate over nodes and mark
        for n in graph.nodes():
            value = float(graph.node[n].get(self.attribute, 0))
            exclude = graph.node[n].get(self.exclude_attribute, False)
            if value > threshold and exclude is False:
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

class MarkWithIntervals(BaseEstimator, TransformerMixin):
    """MarkWithIntervals.

    Assign attribute-value pair according to a list of quadruples:
    each quadruple defines
    the start, end position and the region; the
    order of the quadruples matters: later quadruples override the
    specification of previous quadruples. The special quadruple (-1,-1,att,val)
    assign the attribute 'att' with value 'val' to all nodes.
    """

    def __init__(self, quadruples=None):
        """Construct."""
        self.quadruples = quadruples

    def transform(self, graphs):
        """Transform."""
        try:
            for graph in graphs:
                graph = self._quadruple_assignment(graph)
                yield graph
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _quadruple_assignment(self, graph):
        for start, end, attribute, value in self.quadruples:
            # iterate over nodes
            for n, d in graph.nodes_iter(data=True):
                if 'position' not in d:
                    # assert nodes must have position attribute
                    raise Exception('Nodes must have "position" attribute')
                # given the 'position' attribute of node assign the attribute
                # value pair accordingly
                pos = d['position']
                if pos >= start and pos < end:
                    graph.node[n][attribute] = value
                elif start == -1 and end == -1:
                    graph.node[n][attribute] = value
        return graph

# ------------------------------------------------------------------------------


class WeightSymmetricTrapezoidal(BaseEstimator, TransformerMixin):
    """Symmetric piecewise linear weight function between two levels. Size of
    the high weights region is is given as the radius around the center
    position. The dropdown between high and low levels is given by the distance
    of the positions with high and low levels.

    By default, the center of the sequence is set as the center of the
    trapezoid. The center position can also be set by providing a dictionary of
    graph ids and center positions. In that case the weighting will abort if
    this annotation is not set for all graphs.

                |rrrrr      - radius_high
                |
    high    __center__
           /          \
    low __/            \__

                      ddd   - distance_high2low

    weighting with center set to sequence center
    >>> from GArDen.convert.sequence import SeqToPathGraph
    >>> graphs = SeqToPathGraph().transform([("ID0", "ACGUACGUAC")])
    >>> wst = WeightSymmetricTrapezoidal(
    ...     high_weight=1,
    ...     low_weight=0,
    ...     radius_high=1,
    ...     distance_high2low=2)
    >>> graphs = wst.transform(graphs)
    >>> [ x["weight"] for x in graphs.next().node.values() ]
    [0, 0, 0, 0.5, 1, 1, 1, 0.5, 0, 0]

    weighting with centers set according to dictionary
    >>> from GArDen.convert.sequence import SeqToPathGraph
    >>> graphs = SeqToPathGraph().transform([("ID0", "ACGUACGUAC")])
    >>> wst = WeightSymmetricTrapezoidal(
    ...     high_weight=1,
    ...     low_weight=0,
    ...     radius_high=1,
    ...     distance_high2low=2,
    ...     center_dict={"ID0" : 4})
    >>> graphs = wst.transform(graphs)
    >>> [ x["weight"] for x in graphs.next().node.values() ]
    [0, 0, 0.5, 1, 1, 1, 0.5, 0, 0, 0]

    """

    def __init__(self,
                 high_weight=1,
                 low_weight=0.1,
                 radius_high=10,
                 distance_high2low=10,
                 attribute='weight',
                 center_dict=None):
        """"Construct.

        Parameters
        ----------
        high_weight : float (default: 1)
            Weight assigned to the nodes at the top of the trapezoid.

        low_weight : float (default: 0.1)
            Weight assigned to the nodes outside of the trapezoid.

        radius_high : integer (default: 10)
            Radius of the top of the trapezoid.

        distance_high2low : integer (default: 10)
            Interpolate from high_weight to low_weight over this many nodes.

        attribute : string (default: 'weight')
            Node attribute to assign the weights to.

        center_dict: dictionary (default: None)
            This dictionary specifies the center positions of all graphs. If
            unset, the actual center is used.
        """
        self.high_weight = high_weight
        self.low_weight = low_weight
        self.radius_high = radius_high
        self.distance_high2low = distance_high2low
        self.attribute = attribute
        self.center_dict = center_dict

    def transform(self, graphs):
        """Transform.

        Parameters
        ----------
        graphs : iterator over path graphs of RNA sequences
        """
        try:
            for graph in graphs:
                graph = self._symmetric_trapezoidal_reweighting(graph)
                yield graph
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _symmetric_trapezoidal_reweighting(self, graph):
        if self.center_dict is None:
            center_position = int(len(graph) / float(2))
        else:
            try:
                center_position = self.center_dict[graph.graph["id"]]
            except KeyError:
                raise Exception(
                    "Center annotation not set for graph '{}'".format(graph.graph["id"]))

        # determine absolute positions from distances
        interpolate_up_start = center_position - \
            self.radius_high - self.distance_high2low
        interpolate_up_end = center_position - self.radius_high
        interpolate_down_start = center_position + self.radius_high
        interpolate_down_end = center_position + \
            self.radius_high + self.distance_high2low

        # iterate over nodes
        for n, d in graph.nodes_iter(data=True):
            if 'position' not in d:
                # assert nodes must have position attribute
                raise Exception('Nodes must have "position" attribute')
            # given the 'position' attribute of node assign weight according to
            # piece wise linear weight function between two levels
            pos = d['position']
            graph.node[n][self.attribute] = _linear_trapezoidal_weight(
                pos,
                high_weight=self.high_weight,
                low_weight=self.low_weight,
                interpolate_up_start=interpolate_up_start,
                interpolate_up_end=interpolate_up_end,
                interpolate_down_start=interpolate_down_start,
                interpolate_down_end=interpolate_down_end)
        return graph

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


def _linear_trapezoidal_weight(pos,
                               high_weight=None,
                               low_weight=None,
                               interpolate_up_start=None,
                               interpolate_up_end=None,
                               interpolate_down_start=None,
                               interpolate_down_end=None):
    """
    Piecewise linear weight function between two levels with specified start end
    positions. This function linearly interpolates between positions
    interpolate_up_start and interpolate_up_end and between positions
    interpolate_down_start and interpolate_down_end where start and end points get
    high and low weights.

    This function does not check for parameter sanity.

              interpolate_up_end
              |
              | interpolate_down_start
              | |
    high      ___
             /   \
    low   __/     \__
           |       |
           |       interpolate_down_end
           |
           interpolate_up_start

    >>> map(lambda pos: _linear_trapezoidal_weight(pos=pos,
    ...                            high_weight=1,
    ...                            low_weight=0,
    ...                            interpolate_up_start=1,
    ...                            interpolate_up_end=3,
    ...                            interpolate_down_start=5,
    ...                            interpolate_down_end=7),
    ...                 [0,1,2,3,4,5,6,7])
    [0, 0, 0.5, 1, 1, 1, 0.5, 0]
    """
    if pos <= interpolate_up_start:
        """
           ___
        __/   \__
        |
        """
        return low_weight
    elif pos > interpolate_up_start and pos < interpolate_up_end:
        """
           ___
        __/   \__
          |
        """
        return (high_weight - low_weight) / float(interpolate_up_end - interpolate_up_start) * \
            (pos - interpolate_up_start) + low_weight
    elif pos >= interpolate_up_end and pos <= interpolate_down_start:
        """
           ___
        __/   \__
            |
        """
        return high_weight
    elif pos > interpolate_down_start and pos < interpolate_down_end:
        """
           ___
        __/   \__
              |
        """
        return high_weight - \
            (high_weight - low_weight) / float(interpolate_down_end - interpolate_down_start) * \
            (pos - interpolate_down_start)
    elif pos >= interpolate_down_end:
        """
           ___
        __/   \__
                |
        """
        return low_weight
