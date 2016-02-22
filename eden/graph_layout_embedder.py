#!/usr/bin/env python
"""Provides layout in 2D of vector instances."""

import numpy as np
import networkx as nx
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------


class GraphLayoutEmbedder(object):
    """Transform high dimensional vectors to two dimensional vectors.

    Take in input list of selectors, then for each point find the closest
    selected instance and materialize an edge between the two.
    Finally output 2D coordinates of the corresponding graph embedding using
    a force layout algorithm.
    """

    def __init__(self,
                 bias=1,
                 n_nearest_neighbors_density=None,
                 n_nearest_neighbor_links=1,
                 n_clusters=2,
                 layout_prog='sfdp',
                 layout_prog_args='-Goverlap=scale',
                 random_state=1,
                 weight_fact_pred=20,
                 weight_fact_union=60,
                 n_iterations=50,
                 metric='rbf', **kwds):
        """Constructor."""
        self.bias = bias
        self.n_nearest_neighbors_density = n_nearest_neighbors_density
        self.n_nearest_neighbor_links = n_nearest_neighbor_links
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.weight_fact_pred = weight_fact_pred
        self.weight_fact_union = weight_fact_union
        self.layout_prog = layout_prog
        self.layout_prog_args = layout_prog_args
        self.metric = metric
        self.kwds = kwds
        self.random_state = random_state
        self.probs = None
        self.target_pred = None
        self.graph = None
        self.knn_graph = None
        self.skeleton_graph = None

    def _cluster_prediction(self, data_matrix=None):
        from sklearn.cluster import KMeans
        clusterer = KMeans(n_clusters=self.n_clusters)
        target_pred = clusterer.fit_predict(data_matrix)
        return target_pred

    def _compute_probs(self,
                       data_matrix=None,
                       kernel_matrix=None,
                       target=None):
        if data_matrix is None:
            raise Exception('Currently data_matrix cannot be None.')
        if target is None:
            target = self.target_pred
        lr = LogisticRegression()
        skf = StratifiedKFold(target, n_folds=3)
        results = []
        for train_index, test_index in skf:
            if data_matrix is not None:
                data_matrix_train = data_matrix[train_index]
                data_matrix_test = data_matrix[test_index]
            else:
                # kernel_matrix_train = \
                # kernel_matrix[np.ix_(train_index, train_index)]
                # kernel_matrix_test = \
                # kernel_matrix[np.ix_(test_index, test_index)]
                raise Exception('kernel matrix not supported yet.')
            target_train = target[train_index]
            # target_test = target[test_index]
            lr.fit(data_matrix_train, target_train)
            probs = lr.predict_proba(data_matrix_test)
            results += [(i, prob) for i, prob in zip(test_index, probs)]
        probs = np.array([prob for i, prob in sorted(results)])
        return probs

    def fit(self, data_matrix=None, kernel_matrix=None, target=None):
        """fit."""
        if target is None:
            self.target_pred = self._cluster_prediction(
                data_matrix=data_matrix)
        self.probs = self._compute_probs(data_matrix=data_matrix,
                                         kernel_matrix=kernel_matrix,
                                         target=target)
        return self

    def fit_transform(self, data_matrix=None, kernel_matrix=None, target=None):
        """fit_transform."""
        self.fit(data_matrix=data_matrix,
                 kernel_matrix=kernel_matrix,
                 target=target)
        return self.transform(data_matrix=data_matrix,
                              kernel_matrix=kernel_matrix)

    def transform(self, data_matrix=None, kernel_matrix=None, target=None):
        """transform."""
        if data_matrix is not None:
            kernel_matrix = pairwise_kernels(data_matrix,
                                             metric=self.metric,
                                             **self.kwds)
        if target is not None:
            self.skeleton_graph = self._build_skeleton_graph(target=target)
        else:
            self.skeleton_graph = self._build_skeleton_graph(
                target=self.target_pred)
        self.knn_graph = self._build_knn_graph(data_matrix=data_matrix,
                                               kernel_matrix=kernel_matrix)
        self._combine_skeleton_with_knn_graph(self.skeleton_graph,
                                              self.knn_graph,
                                              target=target)
        # graph layout
        embedded_data_matrix = self._graph_layout(self.knn_graph)
        # normalize display using 2D PCA
        scaler = PCA(n_components=2)
        embedded_data_matrix = scaler.fit_transform(embedded_data_matrix)
        return embedded_data_matrix

    def _combine_skeleton_with_knn_graph(self,
                                         pred_graph,
                                         nn_graph,
                                         target=None):
        id_offset = max(nn_graph.nodes()) + 1
        offset_pred_graph = nx.relabel_nodes(pred_graph,
                                             lambda x: x + id_offset)
        fixed = offset_pred_graph.nodes()
        for u, v in offset_pred_graph.edges():
            offset_pred_graph[u][v]['weight'] = \
                offset_pred_graph[u][v]['weight'] * self.weight_fact_pred
        pos_offset_pred_graph = nx.spring_layout(offset_pred_graph,
                                                 iterations=self.n_iterations)

        union_graph = nx.union(nn_graph, offset_pred_graph)
        if target is not None:
            y = target
        else:
            y = self.target_pred
        for u, t in zip(nn_graph.nodes(), y):
            union_graph.add_edge(u, t + id_offset, weight=self.bias)
        for u, v in union_graph.edges():
            union_graph[u][v]['weight'] = \
                union_graph[u][v]['weight'] * self.weight_fact_union
            if np.isinf(union_graph[u][v]['weight']):
                union_graph[u][v]['weight'] = 1
        pos_union_graph = nx.spring_layout(union_graph,
                                           iterations=self.n_iterations,
                                           pos=pos_offset_pred_graph,
                                           fixed=fixed)
        filtered_pos_union_graph = {}
        for id in pos_union_graph:
            if id not in offset_pred_graph.nodes():
                filtered_pos_union_graph[id] = pos_union_graph[id]
            else:
                union_graph.remove_node(id)

        return union_graph, filtered_pos_union_graph

    def _build_knn_graph(self, data_matrix=None, kernel_matrix=None):
        # make a graph with instances as nodes
        graph = self._build_graph(kernel_matrix)
        # add the distance attribute to edges
        if data_matrix is not None:
            graph = self._add_distance_to_edges(graph, data_matrix)
        else:
            graph = self._add_similarity_to_edges(graph, kernel_matrix)
        return graph

    def _build_skeleton_graph(self, target=None):
        # find correlations between classes
        ids = np.argsort(target)
        d = defaultdict(list)
        for t, row in zip(target[ids], self.probs[ids]):
            d[t].append(row)

        graph = nx.Graph()
        for k in d:
            graph.add_node(k)
        for k in d:
            a = np.mean(np.vstack(d[k]), axis=0)
            for i, w in enumerate(a):
                weight = w
                len = 1 / float(w) + 0.001
                graph.add_edge(k, i, weight=weight, len=len)
        return graph

    def _add_distance_to_edges(self, graph, data_matrix):
        for e in graph.edges():
            src_id, dest_id = e[0], e[1]
            dist = np.linalg.norm(data_matrix[src_id] - data_matrix[dest_id])
            graph[src_id][dest_id]['len'] = dist
            graph[src_id][dest_id]['weight'] = 1 / dist
        return graph

    def _add_similarity_to_edges(self, graph, kernel_matrix):
        for e in graph.edges():
            src_id, dest_id = e[0], e[1]
            length = 1 - kernel_matrix[src_id, dest_id] ** 2
            graph[src_id][dest_id]['len'] = length
            graph[src_id][dest_id]['weight'] = 1 / length
        return graph

    def _build_graph(self, kernel_matrix):
        graph = nx.Graph()
        graph.add_nodes_from(range(kernel_matrix.shape[0]))

        # build shift tree
        self.link_ids = self._kernel_shift_links(kernel_matrix)
        for i, link in enumerate(self.link_ids):
            if i != link:
                graph.add_edge(i, link)

        # build knn edges
        if self.n_nearest_neighbor_links > 0:
            # find the closest selected instance and instantiate knn edges
            graph = self._add_knn_links(graph, kernel_matrix)
        return graph

    def _kernel_shift_links(self, kernel_matrix):
        data_size = kernel_matrix.shape[0]
        # compute instance density as average pairwise similarity
        density = np.sum(kernel_matrix, 0) / data_size
        # compute list of nearest neighbors
        kernel_matrix_sorted = np.argsort(-kernel_matrix)
        # make matrix of densities ordered by nearest neighbor
        density_matrix = density[kernel_matrix_sorted]
        # if a denser neighbor cannot be found then assign link to the
        # instance itself
        link_ids = list(range(density_matrix.shape[0]))
        # for all instances determine link link
        for i, row in enumerate(density_matrix):
            i_density = row[0]
            # for all neighbors from the closest to the furthest
            for jj, d in enumerate(row):
                # proceed until n_nearest_neighbors_density have been explored
                if self.n_nearest_neighbors_density is not None and \
                        jj > self.n_nearest_neighbors_density:
                    break
                j = kernel_matrix_sorted[i, jj]
                if jj > 0:
                    j_density = d
                    # if the density of the neighbor is higher than the
                    # density of the instance assign link
                    if j_density > i_density:
                        link_ids[i] = j
                        break
        return link_ids

    def _add_knn_links(self, graph, kernel_matrix):
        data_size = kernel_matrix.shape[0]
        # compute list of nearest neighbors
        kernel_matrix_sorted = np.argsort(-kernel_matrix)

        for i in range(data_size):
            # add edges to the knns
            for jj in range(self.n_nearest_neighbor_links):
                j = kernel_matrix_sorted[i, jj]
                if i != j:
                    graph.add_edge(i, j)
        return graph

    def _graph_layout(self, graph):
        two_dimensional_data_matrix = \
            nx.graphviz_layout(graph,
                               prog=self.layout_prog,
                               args=self.layout_prog_args)
        two_dimensional_data_list = [list(two_dimensional_data_matrix[i])
                                     for i in range(len(graph))]
        embedded_data_matrix = scale(np.array(two_dimensional_data_list))
        return embedded_data_matrix
