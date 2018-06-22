#!/usr/bin/env python
"""Provides utilities for aligning graphs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import networkx as nx
from eden.graph import vertex_vectorize
from sklearn.neighbors import NearestNeighbors
from itertools import product
from collections import defaultdict
from eden.display import draw_graph

import logging


def stable(rankings, list_a, list_b):
    """Compute stable alignment.

    rankings[(a, n)] = partner that a ranked n^th
    >>> from itertools import product
    >>> A = ['1','2','3','4','5','6']
    >>> B = ['a','b','c','d','e','f']
    >>> rank = dict()
    >>> rank['1'] = (1,4,2,6,5,3)
    >>> rank['2'] = (3,1,2,4,5,6)
    >>> rank['3'] = (1,2,4,3,5,6)
    >>> rank['4'] = (4,1,2,5,3,6)
    >>> rank['5'] = (1,2,3,6,4,5)
    >>> rank['6'] = (2,1,4,3,5,6)
    >>> rank['a'] = (1,2,3,4,5,6)
    >>> rank['b'] = (2,1,4,3,5,6)
    >>> rank['c'] = (5,1,6,3,2,4)
    >>> rank['d'] = (1,3,2,5,4,6)
    >>> rank['e'] = (4,1,3,6,2,5)
    >>> rank['f'] = (2,1,4,3,6,5)
    >>> Arankings = dict(((a, rank[a][b_]), B[b_]) for (a, b_) in product(A, range(0, 6)))
    >>> Brankings = dict(((b, rank[b][a_]), A[a_]) for (b, a_) in product(B, range(0, 6)))
    >>> rankings = Arankings
    >>> rankings.update(Brankings)
    >>> stable(rankings, A, B)
    [('1', 'a'), ('2', 'b'), ('3', 'd'), ('4', 'f'), ('5', 'c'), ('6', 'e')]
    """
    partners = dict((a, (rankings[(a, 1)], 1)) for a in list_a)
    # whether the current pairing (given by `partners`) is stable
    is_stable = False
    while is_stable is False:
        is_stable = True
        for b in list_b:
            is_paired = False  # whether b has a pair which b ranks <= to n
            for n in range(1, len(list_b) + 1):
                a = rankings[(b, n)]
                a_partner, a_n = partners[a]
                if a_partner == b:
                    if is_paired:
                        is_stable = False
                        partners[a] = (rankings[(a, a_n + 1)], a_n + 1)
                    else:
                        is_paired = True
    stable_list = sorted((a, b) for (a, (b, n)) in partners.items())
    return stable_list


def init_vec(orig_graph):
    graph = orig_graph.copy()
    for u in graph.nodes():
        graph.node[u]['vec'] = []
    return graph


def annotate_with_bfs(orig_graph, start, max_depth=20):
    graph = orig_graph.copy()
    for u in graph.nodes():
        graph.node[u]['vec'].append(1)

    visited, queue, dist = set(), [start], dict()
    dist[start] = 0
    while queue:
        vertex = queue.pop(0)
        if dist[vertex] <= max_depth:
            if vertex not in visited:
                visited.add(vertex)
                val = max_depth - dist[vertex]
                graph.node[vertex]['vec'][-1] = val
                next_nodes = [u for u in graph.neighbors(vertex)
                              if u not in visited]
                next_dist = dist[vertex] + 1
                for u in next_nodes:
                    dist[u] = next_dist
                queue.extend(next_nodes)
    return graph


def make_same_size(GA_orig, GB_orig):
    GA = GA_orig.copy()
    GB = GB_orig.copy()

    na = len(GA)
    nb = len(GB)
    if na > nb:
        for i in range(nb, na):
            GB.add_node(i, label='x')
    if nb > na:
        for i in range(na, nb):
            GA.add_node(i, label='x')
    return GA, GB


def trim_pairings(pairings, GA_orig, GB_orig):
    npairings = []
    for a, b in pairings:
        i, j = int(a[1:]) - 1, int(b[1:]) - 1
        if i < len(GA_orig) and j < len(GB_orig):
            npairings.append((i, j))

    return npairings


def match(GA_orig, GB_orig, order=3, max_depth=10, complexity=4):
    if len(GA_orig) > len(GB_orig):
        GA, GB = GB_orig.copy(), GA_orig.copy()
        logging.warning('Warning: reference graph is B not A')
    else:
        GA, GB = GA_orig.copy(), GB_orig.copy()
    # logging.warning('Matching graph A (%d nodes) to graph B (%d nodes)' % (len(GA_orig), len(GB_orig)))

    GA, GB = make_same_size(GA, GB)

    M = vertex_vectorize([GA, GB], complexity=complexity, normalization=True, inner_normalization=True)
    MA, MB = M[0], M[1]

    nnA = NearestNeighbors(n_neighbors=len(GA)).fit(MA)
    d, BprefA = nnA.kneighbors(MB)

    nnB = NearestNeighbors(n_neighbors=len(GB)).fit(MB)
    d, AprefB = nnB.kneighbors(MA)

    # mark bfv in vec attribute
    GA, GB = init_vec(GA), init_vec(GB)
    for k in range(order):
        ds = d[:, 0]
        id_max_A = np.argsort(ds)[k]
        id_max_B = AprefB[id_max_A][0]

        GA = annotate_with_bfs(GA, id_max_A, max_depth=max_depth)
        GB = annotate_with_bfs(GB, id_max_B, max_depth=max_depth)
    # draw_graph_set([GA,GB],n_graphs_per_line=2, size=9, secondary_vertex_label='vec')

    # vectorize 2nd time with real values this time
    M = vertex_vectorize([GA, GB], complexity=complexity, discrete=False, normalization=False, inner_normalization=False)
    MA, MB = M[0], M[1]

    nnA = NearestNeighbors(n_neighbors=len(GA)).fit(MA)
    d, BprefA = nnA.kneighbors(MB)

    nnB = NearestNeighbors(n_neighbors=len(GB)).fit(MB)
    d, AprefB = nnB.kneighbors(MA)

    A = ['A%d' % (i + 1) for i in range(len(GA))]
    B = ['B%d' % (i + 1) for i in range(len(GB))]

    Arankings = dict(((A[i], j + 1), B[AprefB[i, j]]) for i, j in product(range(len(GA)), range(len(GA))))
    Brankings = dict(((B[i], j + 1), A[BprefA[i, j]]) for i, j in product(range(len(GB)), range(len(GB))))

    rankings = Arankings
    rankings.update(Brankings)
    pairings = stable(rankings, A, B)

    # remove dummy node pairings
    npairings = trim_pairings(pairings, GA_orig, GB_orig)
    orderA, orderB = list(zip(*sorted(npairings)))
    return orderB


def _max_common_subgraph(GA, GB, pairings):
    matches = dict([(i, j) for i, j in enumerate(pairings)])
    node_ids = []
    for i, j in GA.edges():
        ii = matches[i]
        jj = matches[j]
        li = GA.node[i]['label']
        lii = GB.node[ii]['label']
        lj = GA.node[j]['label']
        ljj = GB.node[jj]['label']
        if ((ii, jj) in GB.edges() or (jj, ii) in GB.edges()) and li == lii and lj == ljj:
            node_ids.append(ii)
            node_ids.append(jj)
    G = nx.subgraph(GB, node_ids)
    cc = nx.connected_components(G)
    return cc, G


def max_common_subgraph(GA, GB, pairings):
    cc, G = _max_common_subgraph(GA, GB, pairings)
    len_ccs = [(len(ci), ci) for ci in cc]
    if not len_ccs:
        return None
    n_cc, max_cc_ids = max(len_ccs)
    max_cc = nx.subgraph(G, max_cc_ids)
    return max_cc


def max_common_subgraphs(GA, GB, pairings):
    cc, G = _max_common_subgraph(GA, GB, pairings)
    cc_graphs = [nx.subgraph(G, ci) for ci in cc]
    return cc_graphs


def compute_matching_edges_fraction(GA, GB, pairings):
    count = 0
    matches = dict([(i, j) for i, j in enumerate(pairings)])
    for i, j in GA.edges():
        ii = matches[i]
        jj = matches[j]
        if (ii, jj) in GB.edges():
            count += 1
    return float(count) / len(GA.edges())


def compute_matching_neighborhoods_fraction(GA, GB, pairings):
    count = 0
    matches = dict([(i, j) for i, j in enumerate(pairings)])
    matching_edges = defaultdict(list)
    for i, j in GA.edges():
        ii = matches[i]
        jj = matches[j]
        if (ii, jj) in GB.edges():
            matching_edges[i].append(j)
            matching_edges[j].append(i)
    for u in GA.nodes():
        if matching_edges.get(u, False):
            neighbors = nx.neighbors(GA, u)
            matches_neighborhood = True
            for v in neighbors:
                if v not in matching_edges[u]:
                    matches_neighborhood = False
                    break
            if matches_neighborhood:
                count += 1
    return float(count) / len(GA.nodes())


def compute_max_common_subgraph_size(GA, GB, pairings):
    G = max_common_subgraph(GA, GB, pairings)
    if G is None:
        return 0
    return len(G) / len(GB)


def compute_max_common_subgraphs_size(GA, GB, pairings):
    gs = max_common_subgraphs(GA, GB, pairings)
    if not gs:
        return 0
    return sum(len(g) if g else 0 for g in gs) / float(len(GB))


def compute_quality(GA, GB, pairings):
    return compute_max_common_subgraphs_size(GA, GB, pairings)


def random_optimize(GA, GB, n_iter=20):
    best_c = None
    best_order = None
    best_depth = None
    best_quality = -1
    for it in range(n_iter):
        c = random.randint(1, 7)
        order = random.randint(1, 10)
        depth = random.randint(1, 20)
        pairings = match(GA, GB, complexity=c, order=order, max_depth=depth)
        quality = compute_quality(GA, GB, pairings)
        if quality > best_quality:
            best_quality = quality
            best_c = c
            best_order = order
            best_depth = depth
    logging.debug('[random search] quality:%.2f c:%d o:%d d:%d' % (best_quality, best_c, best_order, best_depth))
    return best_quality, best_c, best_order, best_depth


def line_optimize(
        GA,
        GB,
        n_iter,
        best_c,
        best_order,
        best_depth):

    best_quality = -1
    c_range = range(8, 0, -1)
    order_range = range(10, 0, -1)
    depth_range = range(20, 0, -1)

    for it in range(n_iter):
        c = best_c
        order = best_order
        depth = best_depth
        for c in c_range:
            pairings = match(GA, GB, complexity=c, order=order, max_depth=depth)
            quality = compute_quality(GA, GB, pairings)
            if quality >= best_quality:
                best_quality = quality
                best_c = c
                best_order = order
                best_depth = depth
        c = best_c
        order = best_order
        depth = best_depth
        for order in order_range:
            pairings = match(GA, GB, complexity=c, order=order, max_depth=depth)
            quality = compute_quality(GA, GB, pairings)
            if quality >= best_quality:
                best_quality = quality
                best_c = c
                best_order = order
                best_depth = depth
        c = best_c
        order = best_order
        depth = best_depth
        for depth in depth_range:
            pairings = match(GA, GB, complexity=c, order=order, max_depth=depth)
            quality = compute_quality(GA, GB, pairings)
            if quality >= best_quality:
                best_quality = quality
                best_c = c
                best_order = order
                best_depth = depth
    logging.debug('[line search]   quality:%.2f c:%d o:%d d:%d' % (best_quality, best_c, best_order, best_depth))
    return best_quality, best_c, best_order, best_depth


def optimize(GA, GB, n_iter_random=20, n_iter_line=4):
    best_quality, best_c, best_order, best_depth = random_optimize(GA, GB, n_iter=n_iter_random)
    best_quality, best_c, best_order, best_depth = line_optimize(GA, GB, n_iter_line, best_c, best_order, best_depth)
    return best_quality, best_c, best_order, best_depth


def draw_match(GA, GB, pairings, size=10):
    G = nx.disjoint_union(GA, GB)
    for i, jj in enumerate(pairings):
        j = len(GA) + jj
        if G.node[i]['label'] == G.node[j]['label']:
            G.add_edge(i, j, label=i, nesting=True)
    draw_graph(
        G,
        size=size,
        vertex_border=False,
        vertex_size=400,
        edge_label=None,
        edge_alpha=.2,
        dark_edge_color='label',
        dark_edge_dotted=False,
        dark_edge_alpha=1,
        colormap='Set2',
        ignore_for_layout='nesting')


# -----------------------------------------------------------------------------


class GraphAligner(object):
    """Aligns graphs."""

    def __init__(self, complexity=3, order=1, max_depth=5):
        """construct."""
        self.set_params(complexity, order, max_depth)
        self.pairings = None
        self.quality = None
        self.max_common_graph = None

    def set_params(self, complexity=3, order=1, max_depth=5):
        self.complexity = complexity
        self.order = order
        self.max_depth = max_depth

    def _store(self, graph_a, graph_b):
        GA = graph_a.copy()
        GB = graph_b.copy()
        if len(GA) > len(GB):
            GA, GB = GB, GA
        self.ref_graph = GA
        self.second_ref_graph = GB

    def _optimize(self, n_iter_random=30, n_iter_line=3):
        GA, GB = self.ref_graph, self.second_ref_graph
        q, c, o, d = optimize(
            GA,
            GB,
            n_iter_random=n_iter_random,
            n_iter_line=n_iter_line)
        self.quality = q
        self.complexity = c
        self.order = o
        self.max_depth = d

    def match(self, graph_a, graph_b, n_iter_random=30, n_iter_line=3):
        self._store(graph_a, graph_b)
        self._optimize(n_iter_random, n_iter_line)
        self._match(graph_a, graph_b)
        return self

    def _match(self, graph_a, graph_b):
        self._store(graph_a, graph_b)
        GA, GB = self.ref_graph, self.second_ref_graph
        self.pairings = match(
            GA,
            GB,
            complexity=self.complexity,
            order=self.order,
            max_depth=self.max_depth)
        self.quality = compute_quality(GA, GB, self.pairings)
        return self

    def get_matching_quality(self):
        return self.quality

    def draw_match(self):
        GA, GB = self.ref_graph, self.second_ref_graph
        draw_match(GA, GB, self.pairings)
        return self

    def max_common_subgraph(self):
        GA, GB = self.ref_graph, self.second_ref_graph
        self.max_common_graph = max_common_subgraph(GA, GB, self.pairings)
        return self.max_common_graph

    def max_common_subgraphs(self):
        GA, GB = self.ref_graph, self.second_ref_graph
        self.common_graphs = max_common_subgraphs(GA, GB, self.pairings)
        return self.common_graphs
