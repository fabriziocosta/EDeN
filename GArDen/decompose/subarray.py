#!/usr/bin/env python
"""Provides ways to split graphs."""

from sklearn.base import BaseEstimator, TransformerMixin
import networkx as nx

import logging
logger = logging.getLogger(__name__)


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
