#!/usr/bin/env python
"""Provides editing of nodes and edges based on attributes."""

from sklearn.base import BaseEstimator, TransformerMixin

import logging
logger = logging.getLogger(__name__)


class DeleteEdge(BaseEstimator, TransformerMixin):
    """
    Delete edges.

    Delete an edge if its dictionary has a key equal to 'attribute' and
    the 'condition' is true between 'value' and the value associated to
    key=attribute.
    """

    def __init__(self, attribute=None, value=None):
        """Constructor."""
        self.attribute = attribute
        self.value = value

    def transform(self, graphs):
        """Transform."""
        try:
            for graph in graphs:
                graph = self._delete_edges(graph)
                yield graph
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _delete_edges(self, graph):
        for u, v in graph.edges():
            if self.attribute in graph.edge[u][v] and \
                    graph.edge[u][v].get(self.attribute, False) == self.value:
                graph.remove_edge(u, v)
        return graph


class DeleteNode(BaseEstimator, TransformerMixin):
    """Delete node.

    Delete a node if its dictionary has a key equal to 'attribute' and
    the 'condition' is true between 'value' and the value associated to
    key=attribute.

    Parameters
    ----------
    graphs : iterator over path graphs of RNA sequences

    attribute : string
        The key of the node dictionary.
    value : string
        The value associated to the attribute.
    """

    def __init__(self, attribute_value_dict=None, mode='AND'):
        """Constructor."""
        self.attribute_value_dict = attribute_value_dict
        self.mode = mode

    def transform(self, graphs):
        """Transform."""
        try:
            for graph in graphs:
                for n, node_dict in graph.nodes_iter(data=True):
                    if self.mode == 'AND':
                        trigger = True
                        for attribute in self.attribute_value_dict:
                            value = self.attribute_value_dict[attribute]
                            if attribute not in node_dict or \
                                    node_dict[attribute] != value:
                                trigger = False
                    elif self.mode == 'OR':
                        trigger = False
                        for attribute in self.attribute_value_dict:
                            value = self.attribute_value_dict[attribute]
                            if attribute in node_dict and \
                                    node_dict[attribute] == value:
                                trigger = True
                    if trigger is True:
                        graph.remove_node(n)
                yield graph
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)
