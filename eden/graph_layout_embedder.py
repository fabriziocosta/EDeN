#!/usr/bin/env python
"""Provides layout in 2D of vector instances."""

from collections import defaultdict
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.cluster import KMeans

import logging
logger = logging.getLogger(__name__)


def serialize_dict(the_dict, full=True, offset='small'):
    """serialize_dict."""
    if the_dict:
        text = []
        for key in sorted(the_dict):
            if offset == 'small':
                line = '%10s: %s' % (key, the_dict[key])
            elif offset == 'large':
                line = '%25s: %s' % (key, the_dict[key])
            elif offset == 'very_large':
                line = '%50s: %s' % (key, the_dict[key])
            else:
                raise Exception('unrecognized option: %s' % offset)
            line = line.replace('\n', ' ')
            if full is False:
                if len(line) > 100:
                    line = line[:100] + '  ...  ' + line[-20:]
            text.append(line)
        return '\n'.join(text)
    else:
        return ""

# -----------------------------------------------------------------------------


class Embedder(object):
    """Transform high dimensional vectors to two dimensional vectors.

    Take in input list of selectors, then for each point find the closest
    selected instance and materialize an edge between the two.
    Finally output 2D coordinates of the corresponding graph embedding using
    a force layout algorithm.
    """

    def __init__(self,
                 nearest_neighbors_threshold=15,
                 quick_shift_threshold=3,
                 class_bias=1,
                 outlier_bias=3,
                 prediction_bias=1,
                 random_state=1,
                 metric='cosine', **kwds):
        """Constructor."""
        self.random_state = random_state
        self.quick_shift_threshold = quick_shift_threshold
        self.nearest_neighbors_threshold = nearest_neighbors_threshold
        self.class_bias = class_bias
        self.outlier_bias = outlier_bias
        self.prediction_bias = prediction_bias
        self.metric = metric
        self.kwds = kwds

        self.probs = None
        self.target_pred = None
        self.graph = None
        self.knn_graph = None

    def __str__(self):
        """String."""
        return "%s:\n%s" % (self.__class__, serialize_dict(self.__dict__))

    def _laplace_smooting(self, probs, target):
        n = len(set(target))
        smoothed_probs = []
        for row_probs in probs:
            smoothed_row_probs = []
            tot = 0
            for p in row_probs:
                val = p + 1 / (float(n))
                tot += val
                smoothed_row_probs.append(val)
            smoothed_row_probs = np.array(smoothed_row_probs) / tot
            smoothed_probs.append(smoothed_row_probs)
        smoothed_probs = np.array(smoothed_probs)
        return smoothed_probs

    def estimate_probability_distribution(self, data_matrix=None, target=None):
        """Estimate probability distribution of each instance."""
        self.target = target
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
        probs = self._laplace_smooting(probs, target)
        return probs

    def fit(self, data_matrix=None, target=None, n_clusters=None):
        """fit."""
        if target is None and n_clusters is not None:
            clusterer = KMeans(n_clusters=n_clusters, init='k-means++',
                               n_init=10, max_iter=300, tol=0.0001,
                               precompute_distances='auto', verbose=0,
                               random_state=1, copy_x=True, n_jobs=1)
            target = clusterer.fit_predict(data_matrix)
        self.probs = self.estimate_probability_distribution(
            data_matrix=data_matrix, target=target)
        return self

    def fit_transform(self, data_matrix=None, target=None, n_clusters=None):
        """fit_transform."""
        self.fit(data_matrix=data_matrix, target=target, n_clusters=n_clusters)
        return self.transform(data_matrix=data_matrix, target=self.target)

    def transform(self, data_matrix=None, target=None):
        """transform."""
        instance_graph = self.build_instance_graph(
            data_matrix=data_matrix,
            target=target,
            probs=self.probs,
            quick_shift_threshold=self.quick_shift_threshold,
            nearest_neighbors_threshold=self.nearest_neighbors_threshold)
        class_graph = self.build_class_graph(instance_graph)
        self.annotate_graph_with_graphviz_layout(instance_graph)
        embedded_data_matrix = [instance_graph.node[v]['pos']
                                for v in instance_graph.nodes()]
        self.class_graph = class_graph
        self.instance_graph = instance_graph

        self.embedded_data_matrix = np.array(embedded_data_matrix)
        self.target = target
        return self.embedded_data_matrix

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
            graph.add_node(v,
                           group=target[v],
                           prob=list(probs[v]),
                           density=density[v],
                           outlier=False)

        # build knn edges
        if nearest_neighbors_threshold > 0:
            # find the closest selected instance and instantiate knn edges
            graph = self._add_knn_links(
                graph,
                kernel_matrix=kernel_matrix,
                kernel_matrix_sorted_ids=kernel_matrix_sorted_ids,
                nneighbors_th=nearest_neighbors_threshold)

        # build shift tree
        first_link_ids = []
        for th in range(1, quick_shift_threshold + 1):
            link_ids = self._kernel_shift_links(
                kernel_matrix=kernel_matrix,
                density_matrix=density_matrix,
                kernel_matrix_sorted_ids=kernel_matrix_sorted_ids,
                quick_shift_threshold=th)
            # use the links from the closest denser neighbor to
            # build a tree and annotate each node with the size
            # of the subtree that it dominates
            if th == 1:
                first_link_ids = link_ids
            for i, link in enumerate(link_ids):
                if i != link:
                    graph.add_edge(i, link, edge_type='shift', rank=th)
        self._annotate_subtree_size(first_link_ids, graph)
        self._annotate_outliers(graph)
        graph = self._compute_len_for_edges(graph, data_matrix, target, probs)
        return graph

    def _annotate_subtree_size(self, link_ids, graph):
        def get_roots(link_ids):
            roots = []
            for i, j in enumerate(link_ids):
                if i == j:
                    roots.append(i)
            return roots

        def convert_links_to_tree(link_ids):
            tree = defaultdict(list)
            for i, j in enumerate(link_ids):
                if i != j:
                    tree[j].append(i)
            return tree

        def post_order_vist(node_id, tree, subtree_sizes):
            count = 1
            for child_id in tree[node_id]:
                count += post_order_vist(child_id, tree, subtree_sizes)
            subtree_sizes[node_id] = count
            return count

        def subtree_sizes(link_ids):
            tree = convert_links_to_tree(link_ids)
            root_ids = get_roots(link_ids)
            subtree_sizes = [0] * len(link_ids)
            total_size = 0
            for root_id in root_ids:
                total_size += post_order_vist(root_id, tree, subtree_sizes)
            return subtree_sizes

        subtree_sizes = subtree_sizes(link_ids)
        for u in graph.nodes():
            if graph.neighbors(u):
                subtree_size = subtree_sizes[u]
                graph.node[u]['subtree_size'] = subtree_size

    def _annotate_outliers(self, graph):
        for u in graph.nodes():
            if graph.neighbors(u):
                knn_size = len([1 for v in graph.neighbors(u)
                                if graph[u][v]['edge_type'] == 'knn'])
                subtree_size = graph.node[u]['subtree_size']
                if knn_size <= 3 and subtree_size == 1:
                    graph.node[u]['outlier'] = True
            else:
                graph.node[u]['outlier'] = True
            # if a node is an outlier then disconnect it
            # if graph.node[u]['outlier'] is True:
            #     neighbors = graph.neighbors(u)
            #     for v in neighbors:
            #         print u, v
            #         graph.remove_edge(u, v)

    def _diameter(self, data_matrix):
        curr_point = data_matrix[0]
        for itera in range(3):
            # find furthest point from curr_point
            id = np.argmax(np.array([np.linalg.norm(point - curr_point)
                                     for point in data_matrix]))
            curr_point = data_matrix[id]
        return max([np.linalg.norm(point - curr_point)
                    for point in data_matrix])

    def _compute_len_for_edges(self, graph, data_matrix, target, probs):
        _max_dist = self._diameter(data_matrix) / 2
        for src_id, dest_id in graph.edges():
            if src_id != dest_id:
                px = data_matrix[src_id]
                pz = data_matrix[dest_id]
                dist = np.linalg.norm(px - pz)
                norm_dist = dist / _max_dist
                src_probs = probs[src_id]
                dest_probs = probs[dest_id]
                p = src_probs.dot(dest_probs)
                prob_of_being_close = 1 - ((1 - p) ** self.prediction_bias)
                desired_dist = norm_dist * (1 - prob_of_being_close)
                if target[src_id] == target[dest_id]:
                    desired_dist /= (1 + self.class_bias)
                    edge_status = 'equal'
                else:
                    edge_status = 'disequal'
                # if one of the endpoints is an outlier
                # then contract the edge according to the outlier_bias
                src_is_outl = graph.node[src_id]['outlier']
                dest_is_outl = graph.node[dest_id]['outlier']
                if src_is_outl or dest_is_outl:
                    desired_dist /= (1 + self.outlier_bias)
                    graph[src_id][dest_id]['outlier'] = True

                graph[src_id][dest_id]['len'] = desired_dist
                graph[src_id][dest_id]['weight'] = 1
                graph[src_id][dest_id]['status'] = edge_status
                graph.node[src_id]['status'] = edge_status
                graph.node[dest_id]['status'] = edge_status
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
        for i, densities in enumerate(density_matrix):
            i_density = densities[0]
            counter = 0
            # for all neighbors from the closest to the furthest
            for jj, j_density in enumerate(densities):
                j = kernel_matrix_sorted_ids[i, jj]
                if jj > 0:
                    # if the density of the neighbor is higher than the
                    # density of the instance count +1
                    if j_density > i_density:
                        counter += 1
                        # proceed until counter reaches quick_shift_threshold
                        if counter >= quick_shift_threshold:
                            link_ids[i] = j
                            break
        return link_ids

    def _add_knn_links(self, graph,
                       kernel_matrix=None, kernel_matrix_sorted_ids=None,
                       nneighbors_th=1):
        size = kernel_matrix.shape[0]
        for i in range(size):
            # add edges to the knns
            for jj in range(1, int(nneighbors_th) + 1):
                j = kernel_matrix_sorted_ids[i, jj]
                if i != j:
                    # check that within the k-nn also i is a knn of j
                    # i.e. use the symmetric nneighbor notion
                    upto = int(nneighbors_th) + 1
                    i_knns = kernel_matrix_sorted_ids[j, :upto]
                    if i in list(i_knns):
                        graph.add_edge(i, j, edge_type='knn', rank=jj)
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
                           class_node=True)

        # add edges
        for id_group in cum_prob_distrib:
            average_probabilities = \
                np.mean(np.vstack(cum_prob_distrib[id_group]), axis=0)
            for i, average_probability in enumerate(average_probabilities):
                if id_group != i:
                    graph.add_edge(id_group, i,
                                   weight=average_probability,
                                   len=1 - average_probability)
        self._filter_knn_edges(graph, knn_threshold=3)
        return graph

    def _filter_knn_edges(self, graph, knn_threshold=3):
        to_be_removed_edges = []
        for u in graph.nodes():
            vs = graph.neighbors(u)
            sorted_edges = sorted(
                [(graph[u][v]['weight'], v) for v in vs], reverse=True)
            for weight, v in sorted_edges[knn_threshold:]:
                to_be_removed_edges.append((u, v))
                to_be_removed_edges.append((v, u))
        to_be_removed_edges = set(to_be_removed_edges)
        edges = graph.edges()
        for u, v in edges:
            if (u, v) in to_be_removed_edges:
                graph.remove_edge(u, v)

    def annotate_graph_with_graphviz_layout(self, graph, random_state=1):
        """Annotate graph with layout information."""
        # remove previous positional annotations
        for v in graph.nodes():
            if 'pos' in graph.node[v]:
                graph.node[v].pop('pos')

        layout_pos = nx.graphviz_layout(
            graph,
            prog='neato',
            args='-Goverlap=False -Gstart=%d' % random_state)

        # annotate nodes with positional information
        for v in graph.nodes():
            # check that no NaN is assigned as a coordinate
            if(layout_pos[v][0] == layout_pos[v][0] and
               layout_pos[v][1] == layout_pos[v][1]):
                pass
            else:
                print v, layout_pos[v], graph.neighbors(v)
                raise Exception('Assigned NaN value to position')
            graph.node[v]['pos'] = layout_pos[v]

    def _get_node_colors(self, codes, cmap=None):
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
        for pos in layout_pos:
            assert(layout_pos[pos][0] == layout_pos[pos][0] and
                   layout_pos[pos][1] == layout_pos[pos][1]),\
                'NaN position detected'
        return layout_pos

    def _convex_hull(self, data_matrix, target, remove_outer_layer=True):
        conv_hulls = []
        for c in set(target):
            points = []
            for z, xy in zip(target, data_matrix):
                if z == c:
                    points.append(xy)
            points = np.array(points)
            hull = ConvexHull(points)
            conv_hulls.append(points[hull.vertices])
        if remove_outer_layer:
            # remove conv hulls
            second_conv_hulls = []
            for c in set(target):
                points = []
                all_points = []
                for z, xy in zip(target, data_matrix):
                    if z == c:
                        all_points.append(xy)
                        if xy not in conv_hulls[c]:
                            points.append(xy)
                points = np.array(points)
                all_points = np.array(all_points)
                # if the points in the convex hull are more than half of the
                # total, then do not remove them from the set
                if len(points) <= 2:
                    hull = ConvexHull(all_points)
                    second_conv_hulls.append(all_points[hull.vertices])
                else:
                    hull = ConvexHull(points)
                    second_conv_hulls.append(points[hull.vertices])
            return second_conv_hulls
        else:
            return conv_hulls

    def display(self, target_dict=None, true_target=None,
                display_hull=True, remove_outer_layer=False,
                display=True, display_only_clean=False,
                display_clean=False, display_class_graph=False,
                file_name='', cmap='rainbow', figure_size=9):
        """Display."""
        if display_only_clean:
            self.display_graph(
                self.instance_graph,
                target_dict=target_dict,
                true_target=true_target,
                display_hull=False,
                figure_size=figure_size,
                cmap=cmap,
                node_size=40,
                display_average_class_link=False,
                display_edge=False,
                file_name=file_name + '_1_clean.pdf')
        else:
            if display_clean:
                self.display_graph(
                    self.instance_graph,
                    target_dict=target_dict,
                    true_target=true_target,
                    display_hull=False,
                    figure_size=figure_size,
                    cmap=cmap,
                    node_size=40,
                    display_average_class_link=False,
                    display_edge=False,
                    file_name=file_name + '_1_clean.pdf')
                self.display_graph(
                    self.instance_graph,
                    target_dict=target_dict,
                    true_target=true_target,
                    display_edge=True,
                    display_hull=False,
                    figure_size=figure_size,
                    cmap=cmap,
                    node_size=40,
                    edge_thickness=.01,
                    display_average_class_link=False,
                    file_name=file_name + '_1.1_denser.pdf')
                if display_hull:
                    self.display_graph(
                        self.instance_graph,
                        target_dict=target_dict,
                        true_target=true_target,
                        display_hull=True,
                        remove_outer_layer=remove_outer_layer,
                        cmap=cmap,
                        figure_size=figure_size,
                        node_size=40,
                        display_average_class_link=False,
                        display_edge=False,
                        file_name=file_name + '_2_clean_hull.pdf')
            self.display_graph(
                self.instance_graph,
                target_dict=target_dict,
                true_target=true_target,
                display_hull=display_hull,
                remove_outer_layer=remove_outer_layer,
                figure_size=figure_size,
                cmap=cmap,
                node_size=40,
                display_average_class_link=True,
                display_label=False,
                file_name=file_name + '_3.pdf')
            if display_class_graph:
                self.display_graph(
                    self.instance_graph,
                    target_dict=target_dict,
                    display_only_class=True,
                    display_label=True,
                    display_hull=False,
                    cmap=cmap,
                    figure_size=figure_size,
                    node_size=700,
                    file_name=file_name + '_4_target.pdf')
        if display:
            plt.show()

    def display_graph(self, graph, target_dict=None, true_target=None,
                      display_label=False, display_edge=True,
                      display_average_class_link=False,
                      display_only_class=False,
                      display_hull=True, remove_outer_layer=False,
                      edge_thickness=40,
                      cmap='gist_ncar', node_size=600, figure_size=15,
                      file_name=''):
        """Display graph."""
        if target_dict is None:
            target_dict = {i: i for i in set(self.target)}
        fig, ax = plt.subplots(figsize=(figure_size, figure_size))
        if display_only_class:
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
                group_id = average_graph.node[id]['group']
                node_label_dict[id] = target_dict[group_id]
                coordinate_matrix = np.vstack(group_coords[id])
                coords = np.mean(coordinate_matrix, axis=0)
                average_graph.node[id]['pos'] = coords

            # display edges only
            self._filter_knn_edges(average_graph, knn_threshold=3)

            edges = average_graph.edges()
            widths = [average_graph[u][v]['weight'] * edge_thickness
                      for u, v in edges]
            layout_pos = self._get_node_layout_positions(average_graph)
            nx.draw_networkx_edges(average_graph, layout_pos, edgelist=edges,
                                   width=widths, edge_color='cornflowerblue')
            codes = np.array([average_graph.node[u]['group']
                              for u in average_graph.nodes()])
            instance_cols = self._get_node_colors(codes, cmap=cmap)
            nx.draw_networkx_nodes(average_graph, layout_pos,
                                   node_color=instance_cols,
                                   cmap=cmap, node_size=node_size,
                                   linewidths=1)
            nx.draw_networkx_labels(average_graph, layout_pos, node_label_dict,
                                    font_size=18, font_weight='black',
                                    font_color='w')
            nx.draw_networkx_labels(average_graph, layout_pos, node_label_dict,
                                    font_size=18, font_weight='light',
                                    font_color='k')
        else:
            if display_hull:
                patches = []
                for points in self._convex_hull(
                        self.embedded_data_matrix, self.target,
                        remove_outer_layer=remove_outer_layer):
                    polygon = Polygon(points)
                    patches.append(polygon)
                p = PatchCollection(patches,
                                    cmap=plt.get_cmap(cmap),
                                    alpha=0.2)
                p.set_array(np.array(range(len(set(self.target)))))
                ax.add_collection(p)
            layout_pos = self._get_node_layout_positions(graph)
            if true_target is not None:
                codes = true_target
            else:
                codes = np.array([graph.node[u]['group']
                                  for u in graph.nodes()])
            instance_cols = self._get_node_colors(codes, cmap=cmap)
            if display_label:
                node_label_dict = {u: target_dict[u] for u in graph.nodes()}
                nx.draw_networkx_labels(graph, layout_pos, node_label_dict,
                                        font_size=14, font_weight='black',
                                        font_color='w')
                nx.draw_networkx_labels(graph, layout_pos, node_label_dict,
                                        font_size=14, font_weight='light',
                                        font_color='k')

            # outliers = [u for u in graph.nodes() if graph.node[u]['outlier']]
            # nx.draw_networkx_nodes(graph, layout_pos,
            #                        nodelist=outliers,
            #                        node_size=node_size,
            #                        linewidths=6)
            if display_hull and display_average_class_link is False:
                linewidths = 0
            else:
                linewidths = 1
            nx.draw_networkx_nodes(graph, layout_pos,
                                   node_color=instance_cols,
                                   cmap=cmap, node_size=node_size,
                                   linewidths=linewidths)

            if display_average_class_link is False:
                if display_edge:
                    # knn edges
                    knn_edges = [(u, v) for u, v in graph.edges()
                                 if graph[u][v].get('edge_type', '') == 'knn']
                    knn_widths = [graph[u][v]['weight'] * edge_thickness
                                  for u, v in knn_edges]
                    if knn_edges and knn_widths:
                        nx.draw_networkx_edges(
                            graph, layout_pos, edgelist=knn_edges, alpha=0.2,
                            edge_color='lightseagreen')
                    # shift edges
                    shift_edges = [
                        (u, v) for u, v in graph.edges()
                        if graph[u][v].get('edge_type', '') == 'shift']
                    shift_widths = [graph[u][v]['weight'] * edge_thickness
                                    for u, v in shift_edges]
                    if shift_edges and shift_widths:
                        nx.draw_networkx_edges(
                            graph, layout_pos, edgelist=shift_edges, alpha=0.2,
                            edge_color='cornflowerblue')
                    # principal shift edges
                    qs_th = 1
                    shift_edges = [
                        (u, v) for u, v in graph.edges()
                        if graph[u][v].get('edge_type', '') == 'shift' and
                        graph[u][v].get('rank', 0) == qs_th]
                    shift_widths = [graph[u][v]['weight'] * edge_thickness
                                    for u, v in shift_edges]
                    if shift_edges and shift_widths:
                        nx.draw_networkx_edges(
                            graph, layout_pos, edgelist=shift_edges, alpha=0.8,
                            width=2.5, edge_color='cornflowerblue')
                    # mark outlier edges taht are contracted
                    out_edges = [
                        (u, v) for u, v in graph.edges()
                        if graph[u][v].get('outlier', False) is True]
                    if out_edges:
                        nx.draw_networkx_edges(
                            graph, layout_pos, edgelist=out_edges, alpha=1,
                            width=.5, edge_color='r')

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
                    group_id = average_graph.node[id]['group']
                    node_label_dict[id] = target_dict[group_id]
                    coordinate_matrix = np.vstack(group_coords[id])
                    coords = np.mean(coordinate_matrix, axis=0)
                    average_graph.node[id]['pos'] = coords

                # display edges only
                self._filter_knn_edges(average_graph, knn_threshold=3)

                edges = average_graph.edges()
                widths = [average_graph[u][v]['weight'] * edge_thickness
                          for u, v in edges]
                layout_pos = self._get_node_layout_positions(average_graph)
                nx.draw_networkx_edges(average_graph, layout_pos,
                                       edgelist=edges,
                                       width=widths,
                                       edge_color='cornflowerblue')
                codes = np.array([average_graph.node[u]['group']
                                  for u in average_graph.nodes()])
                instance_cols = self._get_node_colors(codes, cmap=cmap)
                nx.draw_networkx_nodes(average_graph, layout_pos,
                                       node_color=instance_cols,
                                       cmap=cmap, node_size=700,
                                       alpha=0.5, linewidths=0)
                nx.draw_networkx_labels(average_graph, layout_pos,
                                        node_label_dict,
                                        font_size=18, font_weight='black',
                                        font_color='w')
                nx.draw_networkx_labels(average_graph, layout_pos,
                                        node_label_dict,
                                        font_size=18, font_weight='light',
                                        font_color='k')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.draw()
        if file_name:
            plt.savefig(file_name, bbox_inches='tight',
                        transparent=True, pad_inches=0)
