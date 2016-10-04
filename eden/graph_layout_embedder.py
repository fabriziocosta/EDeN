#!/usr/bin/env python
"""Provides layout in 2D of vectors."""

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
from sklearn.metrics.pairwise import pairwise_distances

import logging
logger = logging.getLogger(__name__)


class GraphEmbedder(object):
    """Transform high dimensional labeled vectors to two dimensional vectors.

    A discrete label for each instance is expected.
    A graph is built where nodes are instances and there exist two types
    of edges: the 'knn' edges and the 'k_shift' edges.
    A knn edge is an edge to the k-th nearest instance that has the same
    label.
    A k_shift edge is an edge to the k-th nearest instance that is denser
    and has a different label.
    The density is defined as the sum of the pairwise cosine similarity between
    an instance and all the other instances.
    The desired edge length is the euclidean distance between the instances.
    If the endpoints of an edge have the same label then the desired distance
    is divided by 1 + class_bias.
    A k-shift edge is deleted if at least one of the endpoints of is an
    outlier.
    Outlier nodes are defined as those instances that have no mutual
    k=knn_outlier neighbors.

    Finally the embedding is computed as the 2D coordinates of the
    corresponding graph embedding using the force layout algorithm from
    Tomihisa Kamada, and Satoru Kawai. "An algorithm for drawing general
    undirected graphs.", Information processing letters 31, no. 1 (1989): 7-15.
    """

    def __init__(self,
                 k=5,
                 class_bias=2,
                 random_state=1,
                 metric='rbf', **kwds):
        """Constructor."""
        self.name = self.__class__.__name__
        self.__version__ = 'v2.0.1199'

        self.random_state = random_state
        self.k = k
        self.class_bias = class_bias
        self.metric = metric
        self.kwds = kwds

        self.probs = None
        self.target_pred = None
        self.graph = None
        self.knn_graph = None

        logger.debug('%s' % str(self))

    def __str__(self):
        """String."""
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

        return "%s:\n%s" % (self.__class__, serialize_dict(self.__dict__))

    def transform(self, data_matrix=None, target=None):
        """transform."""
        if target is None:
            target = np.array([1] * data_matrix.shape[0])
        instance_graph = self.build_graph(data_matrix=data_matrix,
                                          target=target,
                                          k=self.k)
        self.annotate_graph_with_graphviz_layout(instance_graph)
        embedded_data_matrix = [instance_graph.node[v]['pos']
                                for v in instance_graph.nodes()]
        self.instance_graph = instance_graph

        self.embedded_data_matrix = np.array(embedded_data_matrix)
        self.target = target
        return self.embedded_data_matrix

    def build_graph(self, data_matrix=None, target=None, k=3):
        """Build graph."""
        size = data_matrix.shape[0]
        # make kernel
        kernel_matrix = pairwise_kernels(data_matrix,
                                         metric=self.metric,
                                         **self.kwds)
        # compute instance density as average pairwise similarity
        density = np.sum(kernel_matrix, 0) / size
        # compute list of nearest neighbors
        distance_matrix = pairwise_distances(data_matrix)
        knn_ids = np.argsort(distance_matrix)
        # make matrix of densities ordered by nearest neighbor
        density_matrix = density[knn_ids]

        # make a graph with instances as nodes
        graph = nx.Graph()
        for v in range(size):
            graph.add_node(v, group=target[v], outlier=False)

        # build knn edges
        if k > 0:
            # find the closest selected instance and instantiate knn edges
            graph = self._add_knn_links(
                graph,
                target,
                kernel_matrix=kernel_matrix,
                knn_ids=knn_ids,
                nneighbors_th=k)
            self._annotate_outliers(
                graph,
                nneighbors_th=k,
                kernel_matrix=kernel_matrix,
                knn_ids=knn_ids)

        # build shift tree
        for th in range(1, k + 1):
            link_ids = self._kernel_shift_links(
                kernel_matrix=kernel_matrix,
                density_matrix=density_matrix,
                knn_ids=knn_ids,
                k_quick_shift=th,
                target=target)
            for i, link in enumerate(link_ids):
                if i != link:
                    graph.add_edge(i, link, edge_type='shift', rank=th)
        graph = self._compute_edge_len(graph, data_matrix, target)
        return graph

    def _diameter(self, data_matrix):
        curr_point = data_matrix[0]
        for itera in range(3):
            # find furthest point from curr_point
            id = np.argmax(np.array([np.linalg.norm(point - curr_point)
                                     for point in data_matrix]))
            curr_point = data_matrix[id]
        return max([np.linalg.norm(point - curr_point)
                    for point in data_matrix])

    def _compute_edge_len(self, graph, data_matrix, target):
        _max_dist = self._diameter(data_matrix)
        for src_id, dest_id in graph.edges():
            if src_id != dest_id:
                px = data_matrix[src_id]
                pz = data_matrix[dest_id]
                dist = np.linalg.norm(px - pz)
                desired_dist = dist / _max_dist
                if desired_dist == 0:
                    desired_dist = _max_dist / float(data_matrix.shape[0])
                # if endpoints of an edge have the same
                # class then contract teh desired edge length
                if target[src_id] == target[dest_id]:
                    desired_dist /= (1 + self.class_bias)
                graph[src_id][dest_id]['len'] = desired_dist
                graph[src_id][dest_id]['weight'] = 1

                # if shift edge has one endpoint that is an
                # outlier then remove the edge altogether
                i_outl = graph.node[src_id]['outlier']
                j_outl = graph.node[dest_id]['outlier']
                if i_outl or j_outl:
                    if graph[src_id][dest_id]['edge_type'] == 'shift':
                        graph.remove_edge(src_id, dest_id)

        return graph

    def _kernel_shift_links(self, kernel_matrix=None,
                            density_matrix=None,
                            knn_ids=None,
                            k_quick_shift=None,
                            target=None):
        num_targets = len(set(target))
        # if there are fewer targets than  k_quick_shift then reduce
        effective_threshold = min(k_quick_shift, num_targets)
        size = kernel_matrix.shape[0]
        # if a denser neighbor cannot be found then assign link to the
        # instance itself
        link_ids = list(range(size))
        # for all instances determine link link
        for i, densities in enumerate(density_matrix):
            i_density = densities[0]
            counter = 0
            classes = set()
            classes.add(target[i])
            # for all neighbors from the closest to the furthest
            for jj, j_density in enumerate(densities):
                j = knn_ids[i, jj]
                if jj > 0:
                    # if the density of the neighbor is higher than the
                    # density of the instance and the class is different
                    # from previous classes then count +1
                    if j_density > i_density and target[j] not in classes:
                        counter += 1
                        classes.add(target[j])
                # proceed until counter reaches k_quick_shift
                if counter >= effective_threshold:
                    link_ids[i] = j
                    break
        return link_ids

    def _add_knn_links(self, graph, target, nneighbors_th=1,
                       kernel_matrix=None, knn_ids=None):
        size = kernel_matrix.shape[0]
        for i in range(size):
            # add edges to the k nns with same class
            k = 0
            for jj in range(size):
                j = knn_ids[i, jj]
                if i != j:
                    if target[i] == target[j]:
                        graph.add_edge(i, j, edge_type='knn', rank=k)
                        k += 1
                # after having added at most nneighbors_th links exit
                if k > nneighbors_th:
                    break
        return graph

    def _annotate_outliers(self, graph, nneighbors_th=1,
                           kernel_matrix=None, knn_ids=None):
        size = kernel_matrix.shape[0]
        for i in range(size):
            counter = 0
            # add edges to the knns
            for jj in range(1, int(nneighbors_th) + 1):
                j = knn_ids[i, jj]
                if i != j:
                    # check that within the k-nn also i is a knn of j
                    # i.e. use the symmetric nneighbor notion
                    upto = int(nneighbors_th) + 1
                    i_knns = knn_ids[j, :upto]
                    if i in list(i_knns):
                        counter += 1
                        break
            if counter > 0:
                outlier_status = False
            else:
                outlier_status = True
            graph.node[i]['outlier'] = outlier_status

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

    def display(self,
                target_dict=None,
                true_target=None,
                display=True,
                display_outliers=False,
                file_name='',
                cmap='rainbow',
                figure_size=15):
        """Display."""
        self.display_graph(
            self.instance_graph,
            target_dict=target_dict,
            true_target=true_target,
            display_hull=False,
            figure_size=figure_size,
            cmap=cmap,
            node_size=40,
            display_edge=False,
            display_outliers=display_outliers,
            file_name=file_name + '_1_clean.pdf')
        if display:
            plt.show()

    def display_links(self,
                      target_dict=None,
                      true_target=None,
                      display=True,
                      display_outliers=False,
                      file_name='',
                      cmap='rainbow',
                      figure_size=15):
        """display_links."""
        self.display_graph(
            self.instance_graph,
            target_dict=target_dict,
            true_target=true_target,
            display_edge=False,
            display_edges=True,
            display_hull=False,
            figure_size=figure_size,
            cmap=cmap,
            node_size=40,
            edge_thickness=.01,
            display_outliers=display_outliers,
            file_name=file_name + '_2_links.pdf')
        self.display_graph(
            self.instance_graph,
            target_dict=target_dict,
            true_target=true_target,
            display_edge=True,
            display_edges=False,
            display_hull=False,
            figure_size=figure_size,
            cmap=cmap,
            node_size=40,
            edge_thickness=.01,
            display_outliers=display_outliers,
            file_name=file_name + '_3_link.pdf')
        if display:
            plt.show()

    def display_hull(self,
                     target_dict=None,
                     true_target=None,
                     remove_outer_layer=False,
                     display=True,
                     display_outliers=False,
                     file_name='',
                     cmap='rainbow',
                     figure_size=15):
        """display_hull."""
        self.display_graph(
            self.instance_graph,
            target_dict=target_dict,
            true_target=true_target,
            display_hull=True,
            remove_outer_layer=remove_outer_layer,
            cmap=cmap,
            figure_size=figure_size,
            node_size=40,
            display_outliers=display_outliers,
            display_edge=True,
            file_name=file_name + '_4_hull.pdf')
        if display:
            plt.show()

    def display_graph(self,
                      graph,
                      target_dict=None,
                      true_target=None,
                      display_label=False,
                      display_edge=True,
                      display_edges=False,
                      display_outliers=False,
                      display_hull=True,
                      remove_outer_layer=False,
                      edge_thickness=40,
                      cmap='gist_ncar',
                      node_size=600,
                      figure_size=15,
                      file_name=''):
        """Display graph."""
        if target_dict is None:
            target_dict = {i: i for i in set(self.target)}
        fig, ax = plt.subplots(figsize=(figure_size, figure_size))

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
        if display_outliers:
            outliers = [(u, col)
                        for u, col in zip(graph.nodes(), instance_cols)
                        if graph.node[u]['outlier']]
            nodelist = [u for u, col in outliers]
            node_color = [col for u, col in outliers]
            nx.draw_networkx_nodes(graph, layout_pos,
                                   node_color=node_color,
                                   nodelist=nodelist,
                                   node_size=node_size,
                                   alpha=.2,
                                   linewidths=0)
            nx.draw_networkx_nodes(graph, layout_pos,
                                   node_color=node_color,
                                   nodelist=nodelist,
                                   node_size=node_size / 2,
                                   alpha=.45,
                                   linewidths=0)
            non_outliers = [(u, col)
                            for u, col in zip(graph.nodes(), instance_cols)
                            if not graph.node[u]['outlier']]
            nodelist = [u for u, col in non_outliers]
            node_color = [col for u, col in non_outliers]
            nx.draw_networkx_nodes(graph, layout_pos,
                                   node_color=node_color,
                                   nodelist=nodelist,
                                   node_size=node_size,
                                   markeredgecolor='k',
                                   linewidths=1)
        else:
            nx.draw_networkx_nodes(graph, layout_pos,
                                   node_color=instance_cols,
                                   cmap=cmap, node_size=node_size,
                                   linewidths=1)
        if display_edges:
            # knn edges
            knn_edges = [(u, v) for u, v in graph.edges()
                         if graph[u][v].get('edge_type', '') == 'knn']
            knn_colors = [- graph[u][v]['rank'] for u, v in knn_edges]
            if knn_edges and knn_colors:
                nx.draw_networkx_edges(
                    graph, layout_pos, edgelist=knn_edges,
                    edge_cmap=plt.get_cmap('OrRd'),
                    edge_color=knn_colors, alpha=.4)
            # shift edges
            shift_edges = [
                (u, v) for u, v in graph.edges()
                if graph[u][v].get('edge_type', '') == 'shift']
            shift_colors = [-graph[u][v]['rank'] for u, v in shift_edges]
            if shift_edges and shift_colors:
                if len(set(shift_colors)) > 2:
                    nx.draw_networkx_edges(
                        graph, layout_pos, edgelist=shift_edges,
                        edge_cmap=plt.get_cmap('YlGnBu'),
                        edge_color=shift_colors, alpha=.2)
                else:
                    nx.draw_networkx_edges(
                        graph, layout_pos, edgelist=shift_edges,
                        edge_color='cornflowerblue', alpha=.2)
        if display_edge:
            # principal shift edges
            qs_th = 1
            shift_edges = [
                (u, v) for u, v in graph.edges()
                if graph[u][v].get('edge_type', '') == 'shift' and
                graph[u][v].get('rank', 0) == qs_th]
            if shift_edges:
                nx.draw_networkx_edges(
                    graph, layout_pos, edgelist=shift_edges, alpha=0.2,
                    width=1, edge_color='cornflowerblue')
        if display_hull:
            self._draw_class_id(graph,
                                target_dict=target_dict,
                                cmap=cmap,
                                node_size=600)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.draw()
        if file_name:
            plt.savefig(file_name, bbox_inches='tight',
                        transparent=True, pad_inches=0)

    def _draw_class_id(self, graph,
                       target_dict=None,
                       cmap=None,
                       node_size=800):
        group_coords = defaultdict(list)
        node_label_dict = dict()
        ids = graph.nodes()
        for id in ids:
            group_id = graph.node[id]['group']
            group_coords[group_id].append(graph.node[id]['pos'])
        average_graph = nx.Graph()
        for group_id in group_coords:
            node_label_dict[group_id] = target_dict[group_id]
            coordinate_matrix = np.vstack(group_coords[group_id])
            coords = np.mean(coordinate_matrix, axis=0)
            average_graph.add_node(group_id, pos=coords)
        layout_pos = self._get_node_layout_positions(average_graph)
        # codes = np.array([u for u in average_graph.nodes()])
        # instance_cols = self._get_node_colors(codes, cmap=cmap)
        nx.draw_networkx_nodes(average_graph, layout_pos,
                               node_color='w',
                               cmap=cmap, node_size=node_size,
                               alpha=.7, linewidths=1)
        nx.draw_networkx_labels(average_graph, layout_pos, node_label_dict,
                                font_size=16, font_weight='black',
                                font_color='w')
        nx.draw_networkx_labels(average_graph, layout_pos, node_label_dict,
                                font_size=16, font_weight='light',
                                font_color='k')

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
