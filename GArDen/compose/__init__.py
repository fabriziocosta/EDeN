#!/usr/bin/env python
"""Provides ways to join distinct graphs."""
from GArDen.transform.contraction import Minor

from sklearn.base import BaseEstimator, TransformerMixin
import networkx as nx

import logging
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------


class Flatten(BaseEstimator, TransformerMixin):
    """DisjointUnion."""

    def __init__(self):
        """Construct."""
        pass

    def transform(self, graphs_list):
        """transform."""
        try:
            for graphs in graphs_list:
                for graph in graphs:
                    yield graph
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)


# ------------------------------------------------------------------------------


class DisjointUnion(BaseEstimator, TransformerMixin):
    """DisjointUnion."""

    def __init__(self):
        """Construct."""
        pass

    def transform(self, graphs_list):
        """transform."""
        try:
            for graphs in graphs_list:
                transformed_graph = self._disjoint_union(graphs)
                yield transformed_graph
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _disjoint_union(self, graphs):
        # make the disjoint union of all graphs
        graph_global = nx.Graph()
        for graph in graphs:
            graph_global = nx.disjoint_union(graph_global, graph)
        return graph_global

# ------------------------------------------------------------------------------


class Union(BaseEstimator, TransformerMixin):
    """Union."""

    def __init__(self, attribute='position'):
        """Construct."""
        self.attribute = attribute

    def transform(self, graphs_list):
        """transform."""
        try:
            minor = Minor()
            graphs = self._union_list(graphs_list)
            return minor.transform(graphs)
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _union_list(self, graphs_list):
        for graphs in graphs_list:
            transformed_graph = self._union(graphs)
            yield transformed_graph

    def _union(self, graphs):
        graph_global = nx.Graph()
        for graph in graphs:
            graph_global = nx.disjoint_union(graph_global, graph)
        for n in graph_global.nodes():
            if self.attribute in graph_global.node[n]:
                graph_global.node[n]['part_id'] = \
                    [graph_global.node[n][self.attribute]]
                graph_global.node[n]['part_name'] = \
                    [graph_global.node[n]['label']]
        return graph_global
