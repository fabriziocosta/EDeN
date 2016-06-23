#!/usr/bin/env python
"""Provides ways to split graphs finding max subarray."""

from sklearn.base import BaseEstimator, TransformerMixin
import networkx as nx

from eden.util.iterated_maximum_subarray import compute_max_subarrays
import logging
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------


class MaxSubArray(BaseEstimator, TransformerMixin):
    """MaxSubarray."""

    def __init__(self, min_subarray_size=None, max_subarray_size=None):
        """Construct."""
        self.min_subarray_size = min_subarray_size
        self.max_subarray_size = max_subarray_size

    def transform(self, graphs):
        """Transform."""
        try:
            self.counter = 0
            for graph in graphs:
                graphs_list = self._split(graph)
                if graphs_list:
                    yield graphs_list
            pass
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _split(self, graph):
        graphs_list_out = []
        subarrays = compute_max_subarrays(
            graph=graph,
            min_subarray_size=self.min_subarray_size,
            max_subarray_size=self.max_subarray_size,
            output='non_minimal', margin=1)
        for subarray in subarrays:
            ids = range(subarray['begin'], subarray['end'])
            subgraph = nx.Graph(graph.subgraph(ids))
            graphs_list_out.append(subgraph)
        return graphs_list_out
