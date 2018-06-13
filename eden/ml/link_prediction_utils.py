#!/usr/bin/env python
"""Provides link prediction utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx
import numpy as np
from eden.util import timeit
from eden.display import draw_graph
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger()


def filter_if_degree_smaller_then(g, th=1):
    """filter_if_degree_smaller_then."""
    subset = [u for u in g.nodes() if len(g.neighbors(u)) >= th]
    return nx.Graph(g.subgraph(subset))


def filter_if_degree_greater_then(g, th=1):
    """filter_if_degree_greater_then."""
    subset = [u for u in g.nodes() if len(g.neighbors(u)) < th]
    return nx.Graph(g.subgraph(subset))


def show_graph(g, vertex_color='typeof', size=15, vertex_label=None):
    """show_graph."""
    degrees = [len(g.neighbors(u)) for u in g.nodes()]

    print(('num nodes=%d' % len(g)))
    print(('num edges=%d' % len(g.edges())))
    print(('num non edges=%d' % len(list(nx.non_edges(g)))))
    print(('max degree=%d' % max(degrees)))
    print(('median degree=%d' % np.percentile(degrees, 50)))

    draw_graph(g, size=size,
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
                             vertex_color='typeof', size=15):
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

    draw_graph(graph, size=size, vertex_color=vertex_color,
               vertex_size=100, vertex_label=None, edge_label=None,
               edge_color='color', edge_alpha=1,
               ignore_for_layout='nesting', dark_edge_alpha=.9,
               dark_edge_color='color')
    return graph
