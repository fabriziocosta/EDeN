#!/usr/bin/env python
"""Provides annotation of minimal cycles."""

from sklearn.base import BaseEstimator, ClassifierMixin
import networkx as nx
import logging
logger = logging.getLogger(__name__)


class TrimToLargestComponent(BaseEstimator, ClassifierMixin):
    """TrimToLargestComponent."""

    def transform(self, graphs):
        """transform."""
        try:
            for graph in graphs:
                yield self._transform(graph)
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _transform(self, graph):
        components = sorted(nx.connected_components(graph),
                            key=len, reverse=True)
        # compute the largest component
        max_component = components[0]
        return graph.subgraph(max_component)
