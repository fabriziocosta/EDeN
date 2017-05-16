#!/usr/bin/env python
"""Provides link prediction utilities."""

import networkx as nx
import numpy as np
import random
from eden.estimator import paired_shuffle
from eden.display import draw_graph
import matplotlib.pyplot as plt


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
    neighborhood_pair.add_edge(endpoint_1, endpoint_2, label="#")
    neighborhood_pair.graph['roots'] = (endpoint_1, endpoint_2)
    return neighborhood_pair


def _make_positive_set(graph, radius):
    for u, v in graph.edges():
        yield _make_neighborhood_pair(graph, u, v, radius)


def _make_negative_set(graph, radius):
    for u, v in nx.non_edges(graph):
        yield _make_neighborhood_pair(graph, u, v, radius)


def make_train_test_set(graph, radius, test_proportion=.3):
    """make_train_test_set."""
    pos = list(_make_positive_set(graph, radius))
    neg = list(_make_negative_set(graph, radius))
    random.shuffle(pos)
    random.shuffle(neg)
    pos_dim = len(pos)
    neg_dim = len(neg)
    tr_pos_graphs = pos[:-int(pos_dim * test_proportion)]
    te_pos_graphs = pos[-int(pos_dim * test_proportion):]
    tr_neg_graphs = neg[:-int(neg_dim * test_proportion)]
    te_neg_graphs = neg[-int(neg_dim * test_proportion):]
    tr_graphs = tr_pos_graphs + tr_neg_graphs
    te_graphs = te_pos_graphs + te_neg_graphs
    tr_targets = [1] * len(tr_pos_graphs) + [0] * len(tr_neg_graphs)

    te_targets = [1] * len(te_pos_graphs) + [0] * len(te_neg_graphs)
    tr_graphs, tr_targets = paired_shuffle(tr_graphs, tr_targets)
    te_graphs, te_targets = paired_shuffle(te_graphs, te_targets)

    return (tr_graphs, np.array(tr_targets)), (te_graphs, np.array(te_targets))


def filter_if_degree_smaller_then(g, th=1):
    """filter_if_degree_smaller_then."""
    subset = [u for u in g.nodes() if len(g.neighbors(u)) >= th]
    return nx.Graph(g.subgraph(subset))


def filter_if_degree_greater_then(g, th=1):
    """filter_if_degree_greater_then."""
    subset = [u for u in g.nodes() if len(g.neighbors(u)) < th]
    return nx.Graph(g.subgraph(subset))


def show_graph(g, vertex_color='typeof'):
    """show_graph."""
    print('num nodes=%d' % len(g))
    print('num edges=%d' % len(g.edges()))
    print('num non edges=%d' % len(list(nx.non_edges(g))))
    max_degree = max([len(g.neighbors(u)) for u in g.nodes()])
    print('max degree=%d' % max_degree)

    draw_graph(g, size=15, colormap='Paired',
               vertex_color=vertex_color, vertex_label=None,
               vertex_size=200, edge_label=None)

    degrees = [len(g.neighbors(u)) for u in g.nodes()]
    plt.hist(degrees, len(set(degrees)) - 1, alpha=0.75)
    plt.grid()
    plt.show()


def display_edge_predictions(g, tr_graphs, tr_targets,
                             te_graphs, te_targets, preds):
    """display_edge_predictions."""
    tr_roots = [gg.graph['roots'] for gg in tr_graphs]
    graph = g.copy()
    for (u, v), t in zip(tr_roots, tr_targets):
        if t == 1:
            graph.edge[u][v]['color'] = 'gray'

    te_roots = [gg.graph['roots'] for gg in te_graphs]

    for (u, v), p, t in zip(te_roots, preds, te_targets):
        # for all predicted edges
        if p == 1:
            # true positive
            if t == 1:
                graph.edge[u][v]['color'] = 'limegreen'
            # false positive
            if t == 0:
                graph.add_edge(u, v)
                graph.edge[u][v]['nesting'] = True
                graph.edge[u][v]['color'] = 'lightgray'
        if p == 0:
            # false negative
            if t == 1:
                graph.edge[u][v]['color'] = 'crimson'

    draw_graph(graph, size=15, colormap='Paired', vertex_color='typeof',
               vertex_size=100, vertex_label=None, edge_label=None,
               edge_color='color', edge_alpha=1,
               ignore_for_layout='nesting', dark_edge_alpha=.4,
               dark_edge_color='color')
    return graph
