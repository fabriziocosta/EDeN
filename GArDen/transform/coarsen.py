#!/usr/bin/env python
"""Provides coarsening."""

from sklearn.base import BaseEstimator, TransformerMixin
import networkx as nx
from GArDen.interfaces import transform
from GArDen.transform.contraction import Contract, contraction_modifier

import logging
logger = logging.getLogger(__name__)


def annotate_degree(graph):
    """Annotate_degree."""
    for n in graph.nodes():
        graph.node[n]['degree'] = len(graph.neighbors(n))


# ------------------------------------------------------------------------------


class KCoreDecomposer(BaseEstimator, TransformerMixin):
    """KCoreDecomposer."""

    def __init__(self,
                 k_list=[]):
        """Constructor."""
        self.k_list = k_list

    def transform(self, graphs=None):
        """transform."""
        try:
            for g in graphs:
                yield self._iterative_kcore_decomposition(g, ks=self.k_list)
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _effective_degree(self, graph, u):
        return sum([1 for v in graph.neighbors(u)
                    if not graph.edge[u][v].get('nesting', False)])

    def _is_high_degree_node(self, graph, u, th=3):
        return self._effective_degree(graph, u) >= th

    def _kcore_decompose(self, graph, k=3):
        high_degree_nodes = [u for u in graph.nodes()
                             if self._is_high_degree_node(graph, u, th=k)]
        low_degree_nodes = [u for u in graph.nodes()
                            if not self._is_high_degree_node(graph, u, th=k)]
        graph_high_degree = graph.copy()
        graph_high_degree.remove_nodes_from(high_degree_nodes)
        graph_low_degree = graph.copy()
        graph_low_degree.remove_nodes_from(low_degree_nodes)
        graph_union = nx.union(graph_high_degree, graph_low_degree)
        edges = graph.edges()
        for e in edges:
            if e not in graph_union.edges():
                graph_union.add_edge(e[0], e[1],
                                     label='=', nesting=True, level=0)
        return graph_union

    def _increase_level(self, graph):
        for u, v in graph.edges():
            if graph.edge[u][v].get('nesting', False) is not False:
                if graph.edge[u][v].get('level', False) is not False:
                    graph.edge[u][v]['level'] += 1

    def _iterative_kcore_decomposition(self, graph, ks=[]):
        g = graph.copy()
        annotate_degree(g)
        for k in ks:
            g = self._kcore_decompose(g, k=k)
            self._increase_level(g)
        return g

# ------------------------------------------------------------------------------


class CliqueDecomposer(BaseEstimator, TransformerMixin):
    """KCoreDecomposer."""

    def __init__(self,
                 min_clique_size=3,
                 nesting=True,
                 original_edges_to_nesting=True):
        """Constructor."""
        self.min_clique_size = min_clique_size
        self.nesting = nesting
        self.original_edges_to_nesting = original_edges_to_nesting

    def transform(self, orig_graphs=None):
        """transform."""
        try:
            graphs = self._transform(orig_graphs)
            # reduce all 'label' attributes of contracted nodes to a
            # histogram to be written in the 'label' attribute of the
            # resulting graph
            label_modifier = contraction_modifier(attribute_in='label',
                                                  attribute_out='label',
                                                  reduction='categorical')
            # reduce all 'weight' attributes of contracted nodes using
            # a sum to be written in the 'weight' attribute of the
            # resulting graph
            weight_modifier = contraction_modifier(attribute_in='weight',
                                                   attribute_out='weight',
                                                   reduction='sum')
            modifiers = [label_modifier, weight_modifier]
            s = self.original_edges_to_nesting
            priors = dict(nesting=self.nesting,
                          weight_scaling_factor=1,
                          original_edges_to_nesting=s)
            ca = 'max_clique_hash'
            graphs = transform(graphs,
                               program=Contract(modifiers=modifiers,
                                                contraction_attribute=ca),
                               parameters_priors=priors)
            return graphs

        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _transform(self, graphs=None):
        for graph in graphs:
            g = graph.copy()
            annotate_degree(g)
            self._annotate_max_clique(g, min_clique_size=self.min_clique_size,
                                      attribute='degree')
            yield g

    def _annotate_cliques(self, graph, min_clique_size=3):
        cliques = list(nx.find_cliques(graph))
        for u in graph.nodes():
            cliques_list = nx.cliques_containing_node(
                graph, nodes=u, cliques=cliques)
            trimmed_cliques_list = [c for c in cliques_list
                                    if len(c) >= min_clique_size]
            if len(trimmed_cliques_list) > 0:
                graph.node[u]['cliques'] = trimmed_cliques_list

    def _clique_weight(self, clique, attribute, graph):
        if len(clique) == 0:
            return 0
        return sum([graph.node[id][attribute] for id in clique])

    def _max_clique(self, graph, u, attribute):
        scores = [(self._clique_weight(clique, attribute, graph), clique)
                  for clique in graph.node[u].get('cliques', [])]
        if scores:
            score, clique = max(scores)
            return clique
        else:
            return [u]

    def _annotate_max_clique(self, graph, min_clique_size=3, attribute=None):
        self._annotate_cliques(graph, min_clique_size=min_clique_size)
        for u in graph.nodes():
            graph.node[u]['max_clique'] = self._max_clique(graph, u, 'degree')
            h = str(hash(tuple(graph.node[u]['max_clique'])))
            graph.node[u]['max_clique_hash'] = h
# ------------------------------------------------------------------------------
