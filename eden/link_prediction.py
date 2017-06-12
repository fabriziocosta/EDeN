#!/usr/bin/env python
"""Provides link prediction utilities."""

import networkx as nx
import numpy as np
import random
from eden.util import timeit
from eden.estimator import paired_shuffle
from eden.display import draw_graph
import matplotlib.pyplot as plt
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


def filter_if_degree_smaller_then(g, th=1):
    """filter_if_degree_smaller_then."""
    subset = [u for u in g.nodes() if len(g.neighbors(u)) >= th]
    return nx.Graph(g.subgraph(subset))


def filter_if_degree_greater_then(g, th=1):
    """filter_if_degree_greater_then."""
    subset = [u for u in g.nodes() if len(g.neighbors(u)) < th]
    return nx.Graph(g.subgraph(subset))


def show_graph(g, vertex_color='typeof', size=15,
               colormap='Paired', vertex_label=None):
    """show_graph."""
    degrees = [len(g.neighbors(u)) for u in g.nodes()]

    print('num nodes=%d' % len(g))
    print('num edges=%d' % len(g.edges()))
    print('num non edges=%d' % len(list(nx.non_edges(g))))
    print('max degree=%d' % max(degrees))
    print('median degree=%d' % np.percentile(degrees, 50))

    draw_graph(g, size=size, colormap=colormap,
               vertex_color=vertex_color, vertex_label=vertex_label,
               vertex_size=200, edge_label=None)

    # display degree distribution
    size = int((max(degrees) - min(degrees)) / 1.5)
    plt.figure(figsize=(size, 3))
    plt.title('Degree distribution')
    _bins = np.arange(min(degrees), max(degrees) + 2) - .5
    n, bins, patches = plt.hist(degrees, _bins,
                                alpha=0.3,
                                facecolor='navy', histtype='bar',
                                rwidth=0.8, edgecolor='k')
    labels = np.array([str(int(i)) for i in n])
    for xi, yi, label in zip(bins, n, labels):
        plt.text(xi + 0.5, yi, label, ha='center', va='bottom')

    plt.xticks(bins + 0.5)
    plt.xlim((min(degrees) - 1, max(degrees) + 1))
    plt.ylim((0, max(n) * 1.1))
    plt.xlabel('Node degree')
    plt.ylabel('Counts')
    plt.grid(linestyle=":")
    plt.show()


@timeit
def display_edge_predictions(g, tr_graphs, tr_targets,
                             te_graphs, te_targets, preds,
                             vertex_color='_label_', size=15):
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
                graph.edge[u][v]['color'] = 'gray'
        if p == 0:
            # false negative
            if t == 1:
                graph.edge[u][v]['color'] = 'crimson'

    draw_graph(graph, size=size, colormap='Paired', vertex_color=vertex_color,
               vertex_size=100, vertex_label=None, edge_label=None,
               edge_color='color', edge_alpha=1,
               ignore_for_layout='nesting', dark_edge_alpha=.9,
               dark_edge_color='color')
    return graph
