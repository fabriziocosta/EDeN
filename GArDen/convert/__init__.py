#!/usr/bin/env python
"""Provides conversion from strings."""

from sklearn.base import BaseEstimator, TransformerMixin
import networkx as nx

import logging
logger = logging.getLogger(__name__)


def seq_to_networkx(seq, constr=None):
    """Convert sequence tuples to networkx graphs."""
    graph = nx.Graph()
    for id, character in enumerate(seq):
        graph.add_node(id, label=character, position=id)
        if id > 0:
            graph.add_edge(id - 1, id, label='-')
    assert(len(graph) > 0), 'ERROR: generated empty graph.\
    Perhaps wrong format?'
    graph.graph['sequence'] = seq
    return graph


# ------------------------------------------------------------------------------

class SeqToPathGraph(BaseEstimator, TransformerMixin):
    """Transform seq lists into path graphs."""

    def transform(self, seqs):
        """transform."""
        try:
            for seq in seqs:
                yield seq_to_networkx(seq)
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)
