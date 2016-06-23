#!/usr/bin/env python
"""Provides layout in 2D of vector instances."""

import random
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
                 quick_shift_threshold=None,
                 nearest_neighbors_threshold=5,
                 true_class_bias=.87,
                 true_class_threshold=3,
                 multi_class_bias=0,
                 multi_class_threshold=3,
                 random_state=1,
                 metric='rbf', **kwds):
        """Constructor."""
        self.random_state = random_state
        self.quick_shift_threshold = quick_shift_threshold
        self.nearest_neighbors_threshold = nearest_neighbors_threshold
        self.true_class_bias = true_class_bias
        self.multi_class_bias = multi_class_bias
        self.multi_class_threshold = multi_class_threshold
        self.true_class_threshold = true_class_threshold
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
        self.true_class_threshold = \
            random.triangular(0,
                              2 * self.true_class_threshold,
                              self.true_class_threshold)
        logger.debug(self.__str__())

    def _laplace_smooting(self, probs):
        n = len(probs)
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
        probs = self._laplace_smooting(probs)
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
        union_graph = self.combine_graphs(
            class_graph=class_graph,
            instance_graph=instance_graph,
            true_class_bias=self.true_class_bias,
            multi_class_bias=self.multi_class_bias,
            multi_class_threshold=self.multi_class_threshold)
        self.annotate_graph_with_graphviz_layout(union_graph)
        self.remove_class_nodes(union_graph)
        embedded_data_matrix = [union_graph.node[v]['pos']
                                for v in union_graph.nodes()]
        self.class_graph = class_graph
        self.instance_graph = instance_graph
        self.union_graph = union_graph

        self.embedded_data_matrix = np.array(embedded_data_matrix)
        self.target = target
        return self.embedded_data_matrix

    def display(self, target_dict=None, true_target=None,
                display_hull=True, remove_outer_layer=False,
                display=True, display_only_clean=False,
                display_clean=False, display_class_graph=False,
                file_name='', cmap='rainbow', figure_size=9):
        """Display."""
        if display_only_clean:
            self.display_graph(
                self.union_graph,
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
                    self.union_graph,
                    target_dict=target_dict,
                    true_target=true_target,
                    display_hull=False,
                    figure_size=figure_size,
                    cmap=cmap,
                    node_size=40,
                    display_average_class_link=False,
                    display_edge=False,
                    file_name=file_name + '_1_clean.pdf')
                if display_hull:
                    self.display_graph(
                        self.union_graph,
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
                self.union_graph,
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
                    self.union_graph,
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
            graph.add_node(v, group=target[v], prob=list(probs[v]))

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
                           class_node=True)

        # add edges
        for id_group in cum_prob_distrib:
            average_probabilities = \
                np.mean(np.vstack(cum_prob_distrib[id_group]), axis=0)
            for i, average_probability in enumerate(average_probabilities):
                if id_group != i:
                    graph.add_edge(
                        id_group, i,
                        weight=average_probability,
                        len=self._prob_to_len(average_probability))
        self._filter_knn_edges(
            graph, knn_threshold=self.true_class_threshold)
        return graph

    def _filter_knn_edges(self, graph, knn_threshold=2):
        surviving_edges = []
        for u in graph.nodes():
            vs = graph.neighbors(u)
            sorted_edges = sorted(
                [(graph[u][v]['weight'], v) for v in vs], reverse=True)
            for w, v in sorted_edges[:knn_threshold]:
                surviving_edges.append((u, v))
                surviving_edges.append((v, u))
        surviving_edges = set(surviving_edges)
        edges = graph.edges()
        for u, v in edges:
            if (u, v) not in surviving_edges:
                graph.remove_edge(u, v)

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

        mcb = multi_class_bias
        if mcb != 0:
            # add links from each instance to the class nodes
            # with a length inversely proportional to the prob
            for u in instance_graph.nodes():
                # find k-th largest prob value (k=multi_class_threshold)
                # and instantiate only the k most probable edges
                th = sorted(probs[u], reverse=True)[multi_class_threshold]
                for group, prob in enumerate(probs[u]):
                    if prob >= th:
                        group_id = group + id_offset
                        length = (1 - mcb) * self._prob_to_len(prob)
                        union_graph.add_edge(u, group_id, len=length)

        if true_class_bias != 0:
            # add links from each instance to its assigned class
            for u in instance_graph.nodes():
                group_id = instance_graph.node[u]['group'] + id_offset
                union_graph.add_edge(u, group_id,
                                     len=1 - true_class_bias)

        return union_graph

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

    def remove_class_nodes(self, graph):
        """Clean graph."""
        ids = graph.nodes()
        for u in ids:
            if graph.node[u].get('class_node', False) is not False:
                graph.remove_node(u)

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
        fig, ax = plt.subplots(figsize=(int(1 * figure_size), figure_size))
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
            self._filter_knn_edges(average_graph,
                                   knn_threshold=self.true_class_threshold)

            edges = average_graph.edges()
            widths = [average_graph[u][v]['weight'] * edge_thickness
                      for u, v in edges]
            layout_pos = self._get_node_layout_positions(average_graph)
            nx.draw_networkx_edges(average_graph, layout_pos, edges=edges,
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
                p = PatchCollection(patches, cmap=plt.get_cmap(cmap), alpha=0.2)
                p.set_array(np.array(range(len(set(self.target)))))
                ax.add_collection(p)
            layout_pos = self._get_node_layout_positions(graph)
            if true_target is not None:
                codes = true_target
            else:
                codes = np.array([graph.node[u]['group'] for u in graph.nodes()])
            instance_cols = self._get_node_colors(codes, cmap=cmap)
            if display_label:
                node_label_dict = {u: target_dict[u] for u in graph.nodes()}
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
                    group_id = average_graph.node[id]['group']
                    node_label_dict[id] = target_dict[group_id]
                    coordinate_matrix = np.vstack(group_coords[id])
                    coords = np.mean(coordinate_matrix, axis=0)
                    average_graph.node[id]['pos'] = coords

                # display edges only
                self._filter_knn_edges(average_graph,
                                       knn_threshold=self.true_class_threshold)

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
        if file_name:
            plt.savefig(file_name, bbox_inches='tight', transparent=True, pad_inches=0)

    def _prob_to_len(self, val):
        assert(0 <= val <= 1)
        return 1 - val

    def _weigth_to_len(self, val):
        _minimal_val_ = 0.01
        _maximal_val_ = 100
        val = max(_minimal_val_, float(val))
        val = min(_maximal_val_, float(val))
        length = 1 / val
        # length = math.sqrt(- math.log(val))
        return length

    def _diameter(self, data_matrix):
        curr_point = data_matrix[0]
        for itera in range(3):
            # find furthest point from curr_point
            id = np.argmax(np.array([np.linalg.norm(point - curr_point)
                                     for point in data_matrix]))
            curr_point = data_matrix[id]
        return max([np.linalg.norm(point - curr_point)
                    for point in data_matrix])

    def _add_distance_to_edges(self, graph, data_matrix):
        _max_dist = self._diameter(data_matrix)
        for src_id, dest_id in graph.edges():
            if src_id != dest_id:
                px = data_matrix[src_id]
                pz = data_matrix[dest_id]
                dist = np.linalg.norm(px - pz)
                graph[src_id][dest_id]['len'] = dist / _max_dist
                graph[src_id][dest_id]['weight'] = 1
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
            for jj in range(1, nearest_neighbors_threshold + 1):
                j = kernel_matrix_sorted_ids[i, jj]
                if i != j:
                    graph.add_edge(i, j, type='knn')
        return graph
