#!/usr/bin/env python
"""Provides link prediction utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
import numpy as np
import random
from eden.util import timeit
from eden.ml.estimator_utils import paired_shuffle
import logging

logger = logging.getLogger()


def _bfs(graph, start, max_depth):
    visited, queue, dist = set(), [start], dict()
    dist[start] = 0
    while queue:
        vertex = queue.pop(0)
        if dist[vertex] <= max_depth:
            if vertex not in visited:
                visited.add(vertex)
                next_nodes = [u for u in graph.neighbors(vertex)
                              if u not in visited]
                next_dist = dist[vertex] + 1
                for u in next_nodes:
                    dist[u] = next_dist
                queue.extend(next_nodes)
    return visited


def _make_neighborhood_pair(graph, endpoint_1, endpoint_2, radius):
    n1 = _bfs(graph, endpoint_1, radius)
    n2 = _bfs(graph, endpoint_2, radius)
    neighborhood_nodes = n1 | n2
    neighborhood_pair = nx.Graph(graph.subgraph(neighborhood_nodes))
    # add a new node connected to the two endpoints
    edge_node_id = len(graph)
    neighborhood_pair.add_node(edge_node_id, label='#')
    neighborhood_pair.add_edge(endpoint_1, edge_node_id, label="#")
    neighborhood_pair.add_edge(edge_node_id, endpoint_2, label="#")
    if neighborhood_pair.has_edge(endpoint_1, endpoint_2):
        neighborhood_pair.remove_edge(endpoint_1, endpoint_2)
    neighborhood_pair.graph['roots'] = (endpoint_1, endpoint_2)
    return neighborhood_pair


def _make_subgraph_set(graph, radius, endpoints):
    for u, v in endpoints:
        yield _make_neighborhood_pair(graph, u, v, radius)


@timeit
def make_train_test_set(graph, radius,
                        test_proportion=.3, ratio_neg_to_pos=10):
    """make_train_test_set."""
    pos = [(u, v) for u, v in graph.edges()]
    neg = [(u, v) for u, v in nx.non_edges(graph)]
    random.shuffle(pos)
    random.shuffle(neg)
    pos_dim = len(pos)
    neg_dim = len(neg)
    max_n_neg = min(pos_dim * ratio_neg_to_pos, neg_dim)
    neg = neg[:max_n_neg]
    neg_dim = len(neg)
    tr_pos = pos[:-int(pos_dim * test_proportion)]
    te_pos = pos[-int(pos_dim * test_proportion):]
    tr_neg = neg[:-int(neg_dim * test_proportion)]
    te_neg = neg[-int(neg_dim * test_proportion):]

    # remove edges
    tr_graph = graph.copy()
    tr_graph.remove_edges_from(te_pos)
    tr_pos_graphs = list(_make_subgraph_set(tr_graph, radius, tr_pos))
    tr_neg_graphs = list(_make_subgraph_set(tr_graph, radius, tr_neg))
    te_pos_graphs = list(_make_subgraph_set(tr_graph, radius, te_pos))
    te_neg_graphs = list(_make_subgraph_set(tr_graph, radius, te_neg))

    tr_graphs = tr_pos_graphs + tr_neg_graphs
    te_graphs = te_pos_graphs + te_neg_graphs
    tr_targets = [1] * len(tr_pos_graphs) + [0] * len(tr_neg_graphs)

    te_targets = [1] * len(te_pos_graphs) + [0] * len(te_neg_graphs)
    tr_graphs, tr_targets = paired_shuffle(tr_graphs, tr_targets)
    te_graphs, te_targets = paired_shuffle(te_graphs, te_targets)

    return (tr_graphs, np.array(tr_targets)), (te_graphs, np.array(te_targets))
