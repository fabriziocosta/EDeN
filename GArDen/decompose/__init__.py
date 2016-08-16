#!/usr/bin/env python
"""Provides ways to split graphs."""

from sklearn.base import BaseEstimator, TransformerMixin
import networkx as nx

import logging

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------


class ThresholdedConnectedComponents(BaseEstimator, TransformerMixin):
    """ThresholdedConnectedComponents."""

    def __init__(self, attribute='importance', threshold=0, min_size=3, max_size=20,
                 shrink_graphs=False,
                 less_then=True, more_than=True):
        """Construct."""
        self.attribute = attribute
        self.threshold = threshold
        self.min_size = min_size
        self.less_then = less_then
        self.more_than = more_than
        self.max_size = max_size
        self.shrink_graphs = shrink_graphs
        self.counter = 0  # this guy looks like hes doing nothing?

    def transform(self, graphs):
        """Transform."""
        try:
            self.counter = 0
            for graph in graphs:
                ccomponents = self._extract_ccomponents(
                    graph,
                    threshold=self.threshold,
                    min_size=self.min_size,
                    max_size=self.max_size)
                yield ccomponents
            pass
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _extract_ccomponents(self, graph, threshold=0, min_size=2, max_size=20):
        # remove all vertices that have a score less then threshold
        cc_list = []

        if self.less_then:
            less_component_graph = graph.copy()
            for v, d in less_component_graph.nodes_iter(data=True):
                if d.get(self.attribute, False):
                    if d[self.attribute] < threshold:
                        less_component_graph.remove_node(v)
            for cc in nx.connected_component_subgraphs(less_component_graph):
                if len(cc) >= min_size and len(cc) <= max_size:
                    cc_list.append(cc)
                if len(cc) > max_size and self.shrink_graphs:
                    cc_list += list(self.enforce_max_size(cc, min_size, max_size))

        # remove all vertices that have a score more then threshold
        if self.more_than:
            more_component_graph = graph.copy()
            for v, d in more_component_graph.nodes_iter(data=True):
                if d.get(self.attribute, False):
                    if d[self.attribute] >= threshold:
                        more_component_graph.remove_node(v)

            for cc in nx.connected_component_subgraphs(more_component_graph):
                if len(cc) >= min_size and len(cc) <= max_size:
                    cc_list.append(cc)

                if len(cc) > max_size and self.shrink_graphs:
                    cc_list += list(self.enforce_max_size(cc, min_size, max_size, choose_cut_node=max))

        return cc_list

    def enforce_max_size(self, graph, min_size, max_size, choose_cut_node=min):
        # checklist contains graphs that are too large.
        checklist = [graph]
        while checklist:
            # remove lowest scoring node:
            graph = checklist.pop()
            scores = [(d[self.attribute], n) for n, d in graph.nodes(data=True)]
            graph.remove_node(choose_cut_node(scores)[1])
            # check the resulting components
            for g in nx.connected_component_subgraphs(graph):
                if len(g) > max_size:
                    checklist.append(g)
                elif len(g) >= min_size:
                    yield g


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
