#!/usr/bin/env python
"""Provides ways to split graphs."""

from sklearn.base import BaseEstimator, TransformerMixin
import networkx as nx

import logging
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------


class ThresholdedConnectedComponents(BaseEstimator, TransformerMixin):
    """ThresholdedConnectedComponents."""

    def __init__(self, attribute='importance', threshold=0, min_size=3,
                 less_then=True, more_than=True):
        """Construct."""
        self.attribute = attribute
        self.threshold = threshold
        self.min_size = min_size
        self.less_then = less_then
        self.more_than = more_than
        self.counter = 0

    def transform(self, graphs):
        """Transform."""
        try:
            self.counter = 0
            for graph in graphs:
                ccomponents = self._extract_ccomponents(
                    graph,
                    threshold=self.threshold,
                    min_size=self.min_size)
                yield ccomponents
            pass
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _extract_ccomponents(self, graph, threshold=0, min_size=2):
        # remove all vertices that have a score less then threshold
        cc_list = []

        if self.less_then:
            less_component_graph = graph.copy()
            for v, d in less_component_graph.nodes_iter(data=True):
                if d.get(self.attribute, False):
                    if d[self.attribute] < threshold:
                        less_component_graph.remove_node(v)
            for cc in nx.connected_component_subgraphs(less_component_graph):
                if len(cc) >= min_size:
                    cc_list.append(cc)

        # remove all vertices that have a score more then threshold
        if self.more_than:
            more_component_graph = graph.copy()
            for v, d in more_component_graph.nodes_iter(data=True):
                if d.get(self.attribute, False):
                    if d[self.attribute] >= threshold:
                        more_component_graph.remove_node(v)

            for cc in nx.connected_component_subgraphs(more_component_graph):
                if len(cc) >= min_size:
                    cc_list.append(cc)
        return cc_list

# ------------------------------------------------------------------------------


class SplitConnectedComponents(BaseEstimator, TransformerMixin):
    """SplitConnectedComponents."""

    def __init__(self, attribute='ref'):
        """Construct."""
        self.attribute = attribute
        self.counter = 0

    def transform(self, graphs):
        """Transform."""
        try:
            self.counter = 0
            for graph in graphs:
                graphs_list = self._split(graph)
                yield graphs_list
            pass
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _split(self, composed_graph):
        # extract the list of the connected components
        graphs_list = list(nx.connected_component_subgraphs(composed_graph))
        # each graph receives a unique identifier
        graphs_list_out = []
        for graph in graphs_list:
            graph = nx.convert_node_labels_to_integers(graph)
            graph.graph[self.attribute] = self.counter
            graphs_list_out.append(graph)
        self.counter += 1
        return graphs_list_out
