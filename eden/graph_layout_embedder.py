#!/usr/bin/env python

import logging

import numpy as np
import networkx as nx
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------


class GraphLayoutEmbedder(object):

    """
    Transform a set of high dimensional vectors to a set of two dimensional vectors.

    Take in input list of selectors, then for each point find the closest selected instance and materialize
    an edge between the two. Finally output 2D coordinates of the corresponding graph embedding using the sfdp
    Graphviz algorithm.

    Parameters
    ----------

    """

    def __init__(self,
                 n_nearest_neighbors=10,
                 n_nearest_neighbor_links=1,
                 layout_prog='sfdp',
                 layout_prog_args='-Goverlap=scale',
                 random_state=1,
                 metric='rbf', **kwds):
        self.n_nearest_neighbors = n_nearest_neighbors
        self.n_nearest_neighbor_links = n_nearest_neighbor_links
        self.layout_prog = layout_prog
        self.layout_prog_args = layout_prog_args
        self.metric = metric
        self.kwds = kwds
        self.random_state = random_state

    def fit(self, data_matrix=None, kernel_matrix=None, target=None):
        return self

    def fit_transform(self, data_matrix=None, kernel_matrix=None, target=None):
        self.fit(data_matrix=data_matrix, kernel_matrix=kernel_matrix, target=target)
        return self.transform(data_matrix=data_matrix, kernel_matrix=kernel_matrix)

    def transform(self, data_matrix=None, kernel_matrix=None):
        if data_matrix is not None:
            kernel_matrix = pairwise_kernels(data_matrix, metric=self.metric, **self.kwds)
        # make a graph with instances as nodes
        graph = self._build_graph(kernel_matrix)
        # add the distance attribute to edges
        if data_matrix is not None:
            self.graph = self._add_distance_to_edges(graph, data_matrix)
        else:
            self.graph = self._add_similarity_to_edges(graph, kernel_matrix)
        # use graph layout
        embedded_data_matrix = self._graph_layout(self.graph)
        # normalize display using 2D PCA
        embedded_data_matrix = PCA(n_components=2).fit_transform(embedded_data_matrix)
        return embedded_data_matrix

    def _add_distance_to_edges(self, graph, data_matrix):
        for e in graph.edges():
            src_id, dest_id = e[0], e[1]
            dist = np.linalg.norm(data_matrix[src_id] - data_matrix[dest_id])
            graph[src_id][dest_id]['len'] = dist
            graph[src_id][dest_id]['weight'] = 1 / graph[src_id][dest_id]['len']
        return graph

    def _add_similarity_to_edges(self, graph, kernel_matrix):
        for e in graph.edges():
            src_id, dest_id = e[0], e[1]
            graph[src_id][dest_id]['len'] = 1 - kernel_matrix[src_id, dest_id] ** 2
            graph[src_id][dest_id]['weight'] = 1 / graph[src_id][dest_id]['len']
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
        # if a denser neighbor cannot be found then assign link to the instance itself
        link_ids = list(range(density_matrix.shape[0]))
        # for all instances determine link link
        for i, row in enumerate(density_matrix):
            i_density = row[0]
            # for all neighbors from the closest to the furthest
            for jj, d in enumerate(row):
                # proceed until n_nearest_neighbors have been explored
                if self.n_nearest_neighbors is not None and jj > self.n_nearest_neighbors:
                    break
                j = kernel_matrix_sorted[i, jj]
                if jj > 0:
                    j_density = d
                    # if the density of the neighbor is higher than the density of the instance assign link
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
        two_dimensional_data_matrix = nx.graphviz_layout(graph,
                                                         prog=self.layout_prog, args=self.layout_prog_args)
        two_dimensional_data_list = [list(two_dimensional_data_matrix[i]) for i in range(len(graph))]
        embedded_data_matrix = scale(np.array(two_dimensional_data_list))
        return embedded_data_matrix
