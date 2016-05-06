#!/usr/bin/env python
"""Provides layout in 2D of vector instances."""

import numpy as np
import networkx as nx
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import random
from sklearn.decomposition import PCA

from eden.util import serialize_dict

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

# -----------------------------------------------------------------------------


class Embedder(object):
    """Transform high dimensional vectors to two dimensional vectors.

    Take in input list of selectors, then for each point find the closest
    selected instance and materialize an edge between the two.
    Finally output 2D coordinates of the corresponding graph embedding using
    a force layout algorithm.
    """

    def __init__(self,
                 quick_shift_threshold=None,
                 nearest_neighbors_threshold=4,
                 repulsion=.1,
                 attraction=20,
                 iterations=100,
                 random_state=1,
                 true_class_bias=1,
                 multi_class_bias=1,
                 multi_class_threshold=0.1,
                 edge_weight_threshold=0.03,
                 fixed_class_nodes=False,
                 cmap='rainbow',
                 metric='rbf', **kwds):
        """Constructor."""
        self.random_state = random_state
        self.quick_shift_threshold = quick_shift_threshold
        self.nearest_neighbors_threshold = nearest_neighbors_threshold
        self.repulsion = repulsion
        self.attraction = attraction
        self.iterations = iterations
        self.true_class_bias = true_class_bias
        self.multi_class_bias = multi_class_bias
        self.multi_class_threshold = multi_class_threshold
        self.edge_weight_threshold = edge_weight_threshold
        self.fixed_class_nodes = fixed_class_nodes
        self.cmap = cmap
        self.metric = metric
        self.kwds = kwds

        self.probs = None
        self.target_pred = None
        self.graph = None
        self.knn_graph = None
        self.skeleton_graph = None

    def __str__(self):
        """String."""
        return "%s:\n%s" % (self.__class__, serialize_dict(self.__dict__))

    def randomize(self, random_state=1):
        """Randomize."""
        self.random_state = random_state
        random.seed(self.random_state)
        self.nearest_neighbors_threshold = \
            int(random.triangular(0,
                                  2 * self.nearest_neighbors_threshold,
                                  self.nearest_neighbors_threshold))
        self.true_class_bias = \
            random.triangular(0,
                              2 * self.true_class_bias,
                              self.true_class_bias)
        self.multi_class_bias = \
            random.triangular(0,
                              2 * self.multi_class_bias,
                              self.multi_class_bias)
        self.multi_class_threshold = \
            random.triangular(0,
                              2 * self.multi_class_threshold,
                              self.multi_class_threshold)
        self.edge_weight_threshold = \
            random.triangular(0,
                              2 * self.edge_weight_threshold,
                              self.edge_weight_threshold)
        self.fixed_class_nodes = random.random() > 0.5
        logger.debug(self.__str__())

    def estimate_probability_distribution(self, data_matrix=None, target=None):
        """Estimate probability distribution of each instance."""
        lr = LogisticRegression()
        skf = StratifiedKFold(target, n_folds=5)
        results = []
        for train_index, test_index in skf:
            data_matrix_train = data_matrix[train_index]
            data_matrix_test = data_matrix[test_index]
            target_train = target[train_index]
            # target_test = target[test_index]
            lr.fit(data_matrix_train, target_train)
            probs = lr.predict_proba(data_matrix_test)
            results += [(i, prob) for i, prob in zip(test_index, probs)]
        probs = np.array([prob for i, prob in sorted(results)])
        return probs

    def fit(self, data_matrix=None, target=None):
        """fit."""
        self.probs = self.estimate_probability_distribution(
            data_matrix=data_matrix, target=target)
        return self

    def fit_transform(self, data_matrix=None, target=None,
                      display=False, display_class_graph=False):
        """fit_transform."""
        self.fit(data_matrix=data_matrix, target=target)
        return self.transform(
            data_matrix=data_matrix,
            target=target,
            display=display,
            display_class_graph=display_class_graph)

    def transform(self, data_matrix=None, target=None,
                  display=False, display_class_graph=False):
        """transform."""
        instance_graph = self.build_instance_graph(
            data_matrix=data_matrix,
            target=target,
            probs=self.probs,
            quick_shift_threshold=self.quick_shift_threshold,
            nearest_neighbors_threshold=self.nearest_neighbors_threshold)
        self.annotate_graph_with_graphviz_layout(instance_graph)
        class_graph = self.build_class_graph(instance_graph)
        self.filter_edges(
            class_graph, edge_weight_threshold=self.edge_weight_threshold)
        self.annotate_graph_with_graphviz_layout(class_graph)
        # self.annotate_graph_with_layout(
        #     class_graph,
        #     repulsion=self.repulsion,
        #     attraction=self.attraction,
        #     iterations=self.iterations,
        #     random_state=self.random_state)
        if self.fixed_class_nodes is False:
            self.unfix_fixed_nodes(class_graph)
        if display_class_graph:
            self.display_graph(
                class_graph,
                display_label=True,
                edge_thickness=40,
                cmap=self.cmap,
                node_size=700)
        union_graph = self.combine_graphs(
            class_graph=class_graph,
            instance_graph=instance_graph,
            true_class_bias=self.true_class_bias,
            multi_class_bias=self.multi_class_bias,
            multi_class_threshold=self.multi_class_threshold)
        self.annotate_graph_with_graphviz_layout(union_graph)
        # self.annotate_graph_with_layout(
        #     union_graph,
        #     random_state=self.random_state,
        #     repulsion=self.repulsion,
        #     attraction=self.attraction,
        #     iterations=self.iterations)
        self.remove_class_nodes(union_graph)
        if display:
            self.display_graph(
                union_graph,
                edge_thickness=30,
                edge_width_threshold=.03,
                cmap=self.cmap,
                node_size=40,
                display_average_class_link=True)
        embedded_data_matrix = [union_graph.node[v]['pos']
                                for v in union_graph.nodes()]
        return np.array(embedded_data_matrix)

    def build_instance_graph(self, data_matrix=None, target=None, probs=None,
                             quick_shift_threshold=None,
                             nearest_neighbors_threshold=3):
        """Build instance graph."""
        size = data_matrix.shape[0]
        # make kernel
        kernel_matrix = pairwise_kernels(data_matrix,
                                         metric=self.metric,
                                         **self.kwds)
        # compute instance density as average pairwise similarity
        density = np.sum(kernel_matrix, 0) / size
        # compute list of nearest neighbors
        kernel_matrix_sorted_ids = np.argsort(-kernel_matrix)
        # make matrix of densities ordered by nearest neighbor
        density_matrix = density[kernel_matrix_sorted_ids]

        # make a graph with instances as nodes
        graph = nx.Graph()
        for v in range(size):
            graph.add_node(v, group=target[v], prob=probs[v])

        # build shift tree
        if quick_shift_threshold != 0:
            self.link_ids = self._kernel_shift_links(
                kernel_matrix=kernel_matrix,
                density_matrix=density_matrix,
                kernel_matrix_sorted_ids=kernel_matrix_sorted_ids,
                quick_shift_threshold=quick_shift_threshold)
            for i, link in enumerate(self.link_ids):
                if i != link:
                    graph.add_edge(i, link, type='shift')

        # build knn edges
        if nearest_neighbors_threshold > 0:
            # find the closest selected instance and instantiate knn edges
            graph = self._add_knn_links(
                graph,
                kernel_matrix=kernel_matrix,
                kernel_matrix_sorted_ids=kernel_matrix_sorted_ids,
                nearest_neighbors_threshold=nearest_neighbors_threshold)

        graph = self._add_distance_to_edges(graph, data_matrix)
        return graph

    def build_class_graph(self, instance_graph):
        """Build graph structure for class instances."""
        # extract groups
        target = np.array([instance_graph.node[v]['group']
                           for v in instance_graph.nodes()])
        probs = np.array([instance_graph.node[v]['prob']
                          for v in instance_graph.nodes()])
        # find correlations between classes
        ids = np.argsort(target)
        cum_prob_distrib = defaultdict(list)
        for group, instance_prob_distribution in zip(target[ids], probs[ids]):
            cum_prob_distrib[group].append(instance_prob_distribution)

        graph = nx.Graph()
        # add one node per class
        for id_group in cum_prob_distrib:
            graph.add_node(id_group,
                           group=id_group,
                           class_node=True,
                           fixed=True)

        # add edges
        for id_group in cum_prob_distrib:
            average_probabilities = \
                np.mean(np.vstack(cum_prob_distrib[id_group]), axis=0)
            for i, average_probability in enumerate(average_probabilities):
                graph.add_edge(id_group, i,
                               weight=average_probability,
                               len=1 / average_probability)
        return graph

    def filter_edges(self, graph, edge_weight_threshold=0.1):
        """Filter edges."""
        edges = graph.edges()
        for u, v in edges:
            if graph.edge[u][v]['weight'] < edge_weight_threshold:
                graph.remove_edge(u, v)

    def multiply_edge_weight(self, graph, edge_weight_factor=10):
        """Filter edges."""
        edges = graph.edges()
        for u, v in edges:
            graph.edge[u][v]['weight'] *= edge_weight_factor
            graph.edge[u][v]['len'] = 1 / graph.edge[u][v]['weight']

    def unfix_fixed_nodes(self, graph):
        """Unfix fixed nodes."""
        for u in graph.nodes():
            if graph.node[u].get('fixed', False) is True:
                graph.node[u]['fixed'] = False

    def combine_graphs(self, true_class_bias=1,
                       multi_class_bias=0, multi_class_threshold=0,
                       class_graph=None, instance_graph=None):
        """Combine graphs."""
        probs = np.array([instance_graph.node[v]['prob']
                          for v in instance_graph.nodes()])

        id_offset = max(instance_graph.nodes()) + 1
        offset_pred_graph = \
            nx.relabel_nodes(class_graph, lambda x: x + id_offset)
        union_graph = nx.union(instance_graph, offset_pred_graph)

        if multi_class_bias != 0:
            for u in instance_graph.nodes():
                for group, prob in enumerate(probs[u]):
                    if prob >= multi_class_threshold:
                        group_id = group + id_offset
                        union_graph.add_edge(
                            u, group_id,
                            weight=prob * multi_class_bias,
                            len=1 / (prob * multi_class_bias))

        if true_class_bias != 0:
            for u in instance_graph.nodes():
                group_id = instance_graph.node[u]['group'] + id_offset
                union_graph.add_edge(u, group_id,
                                     weight=true_class_bias,
                                     len=1 / true_class_bias)

        return union_graph

    def annotate_graph_with_graphviz_layout(self, graph, random_state=1):
        """Annotate graph with layout infoqrmation."""
        for v in graph.nodes():
            if 'pos' in graph.node[v]:
                if graph.node[v].get('fixed', False) is True:
                    x, y = graph.node[v]['pos']
                    graph.node[v]['pos'] = '%.4f, %.4f' % (x, y)
                else:
                    graph.node[v].pop('pos')
        layout_pos = nx.graphviz_layout(
            graph,
            prog='neato',
            args='-Goverlap=False -Gstart=%d' % random_state)
        for v in graph.nodes():
            graph.node[v]['pos'] = layout_pos[v]

    def annotate_graph_with_layout(self, original_graph,
                                   repulsion=.1, attraction=1,
                                   iterations=2000, random_state=1):
        """Annotate graph with layout information."""
        # assign initial random but replicable positions
        random.seed(random_state)
        initialpos = {u: (random.random(), random.random())
                      for u in original_graph.nodes()}
        # find nodes with fixed position
        # and update the initial position for those
        fixed_ids = []
        for u in original_graph.nodes():
            if original_graph.node[u].get('fixed', False) is not False and\
                    original_graph.node[u].get('pos', False) is not False:
                fixed_ids.append(u)
                initialpos[u] = original_graph.node[u]['pos']
        # find nodes with initial position
        # and update the initial position for those
        for u in original_graph.nodes():
            if original_graph.node[u].get('pos', False) is not False:
                initialpos[u] = original_graph.node[u]['pos']

        graph = original_graph.copy()
        for u, v in graph.edges():
            graph[u][v]['weight'] = \
                graph[u][v]['weight'] * attraction

        if fixed_ids:
            layout_pos = nx.spring_layout(graph,
                                          pos=initialpos, k=repulsion,
                                          fixed=fixed_ids,
                                          iterations=iterations)
        else:
            layout_pos = nx.spring_layout(graph,
                                          pos=initialpos, k=repulsion,
                                          iterations=iterations)
        for v in graph.nodes():
            original_graph.node[v]['pos'] = layout_pos[v]

    def remove_class_nodes(self, graph):
        """Clean graph."""
        # remove fixed nodes
        ids = graph.nodes()
        for u in ids:
            if graph.node[u].get('class_node', False) is not False:
                graph.remove_node(u)

    def _get_node_colors(self, graph, cmap=None):
        codes = np.array([graph.node[u]['group'] for u in graph.nodes()])
        cm = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=min(codes),
                                             vmax=max(codes)),
            cmap=cmap)
        cols = np.array(range(min(codes), max(codes) + 1))
        # remove 4th column=alpha
        rgba_color_codes = cm.to_rgba(cols)[:, :3]

        enc = OneHotEncoder(sparse=False)
        codes = enc.fit_transform(codes.reshape(-1, 1))
        instance_cols = np.dot(codes, rgba_color_codes)[:, :3]
        return instance_cols

    def _get_node_layout_positions(self, graph):
        layout_pos = {v: graph.node[v]['pos'] for v in graph.nodes()}
        return layout_pos

    def display_graph(self, graph, display_label=False, display_edge=True,
                      display_average_class_link=False,
                      edge_thickness=20, edge_width_threshold=0.1,
                      cmap='gist_ncar', node_size=700, figure_size=9):
        """Display graph."""
        plt.figure(figsize=(int(1 * figure_size), figure_size))
        layout_pos = self._get_node_layout_positions(graph)
        node_label_dict = {u: u for u in graph.nodes()}
        instance_cols = self._get_node_colors(graph, cmap=cmap)
        if display_label:
            nx.draw_networkx_labels(graph, layout_pos, node_label_dict,
                                    font_size=14, font_weight='black',
                                    font_color='w')
            nx.draw_networkx_labels(graph, layout_pos, node_label_dict,
                                    font_size=14, font_weight='light',
                                    font_color='k')
        nx.draw_networkx_nodes(graph, layout_pos,
                               node_color=instance_cols,
                               cmap=cmap, node_size=node_size,
                               linewidths=1)
        if display_average_class_link is False:
            if display_edge:
                # generic edge
                edges = [(u, v) for u, v in graph.edges()
                         if graph[u][v].get('type', '') == '']
                widths = [graph[u][v]['weight'] * edge_thickness
                          for u, v in edges]
                if edges and widths:
                    nx.draw_networkx_edges(
                        graph, layout_pos, edges=edges, alpha=0.5,
                        width=widths, edge_color='cornflowerblue')
                # knn edges
                knn_edges = [(u, v) for u, v in graph.edges()
                             if graph[u][v].get('type', '') == 'knn']
                knn_widths = [graph[u][v]['weight'] * edge_thickness
                              for u, v in knn_edges]
                if knn_edges and knn_widths:
                    nx.draw_networkx_edges(
                        graph, layout_pos, edges=knn_edges, alpha=0.5,
                        width=knn_widths, edge_color='cornflowerblue')
                # shift edges
                shift_edges = [(u, v) for u, v in graph.edges()
                               if graph[u][v].get('type', '') == 'shift']
                shift_widths = [graph[u][v]['weight'] * edge_thickness
                                for u, v in shift_edges]
                if shift_edges and shift_widths:
                    nx.draw_networkx_edges(
                        graph, layout_pos, edges=shift_edges, alpha=0.9,
                        width=shift_widths, edge_color='cornflowerblue')

        else:
            # rebuild the class graph
            average_graph = self.build_class_graph(graph)
            # add edges between average class
            # find cluster centers in current graph
            group_coords = defaultdict(list)
            node_label_dict = dict()
            ids = graph.nodes()
            for id in ids:
                group_id = graph.node[id]['group']
                group_coords[group_id].append(graph.node[id]['pos'])
            for id in group_coords:
                node_label_dict[id] = average_graph.node[id]['group']
                coordinate_matrix = np.vstack(group_coords[id])
                coords = np.mean(coordinate_matrix, axis=0)
                average_graph.node[id]['pos'] = coords

            # display edges only
            edges = average_graph.edges()
            for u, v in edges:
                if average_graph[u][v]['weight'] < edge_width_threshold:
                    average_graph.remove_edge(u, v)
            edges = average_graph.edges()
            widths = [average_graph[u][v]['weight'] * edge_thickness
                      for u, v in edges]
            layout_pos = self._get_node_layout_positions(average_graph)
            nx.draw_networkx_edges(average_graph, layout_pos, edges=edges,
                                   width=widths, edge_color='cornflowerblue')
            nx.draw_networkx_labels(average_graph, layout_pos, node_label_dict,
                                    font_size=18, font_weight='black',
                                    font_color='w')
            nx.draw_networkx_labels(average_graph, layout_pos, node_label_dict,
                                    font_size=18, font_weight='light',
                                    font_color='k')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.draw()
        plt.show()

    def _add_distance_to_edges(self, graph, data_matrix):
        for src_id, dest_id in graph.edges():
            dist = np.linalg.norm(data_matrix[src_id] - data_matrix[dest_id])
            graph[src_id][dest_id]['weight'] = 1 / dist
            graph[src_id][dest_id]['len'] = dist
        return graph

    def _kernel_shift_links(self, kernel_matrix=None,
                            density_matrix=None,
                            kernel_matrix_sorted_ids=None,
                            quick_shift_threshold=None):
        size = kernel_matrix.shape[0]
        # if a denser neighbor cannot be found then assign link to the
        # instance itself
        link_ids = list(range(size))
        # for all instances determine link link
        for i, row in enumerate(density_matrix):
            i_density = row[0]
            # for all neighbors from the closest to the furthest
            for jj, d in enumerate(row):
                # proceed until quick_shift_threshold have been explored
                if quick_shift_threshold is not None and \
                        jj > quick_shift_threshold:
                    break
                j = kernel_matrix_sorted_ids[i, jj]
                if jj > 0:
                    j_density = d
                    # if the density of the neighbor is higher than the
                    # density of the instance assign link
                    if j_density > i_density:
                        link_ids[i] = j
                        break
        return link_ids

    def _add_knn_links(self, graph,
                       kernel_matrix=None, kernel_matrix_sorted_ids=None,
                       nearest_neighbors_threshold=1):
        size = kernel_matrix.shape[0]
        for i in range(size):
            # add edges to the knns
            for jj in range(nearest_neighbors_threshold):
                j = kernel_matrix_sorted_ids[i, jj]
                if i != j:
                    graph.add_edge(i, j, type='knn')
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
