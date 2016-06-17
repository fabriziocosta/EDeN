#!/usr/bin/env python
"""Provides drawing utilities."""

import networkx as nx
import pylab as plt
import math
from matplotlib.font_manager import FontProperties
import json
from networkx.readwrite import json_graph
from eden.util import _serialize_list


class SetEncoder(json.JSONEncoder):
    """SetEncoder."""

    def default(self, obj):
        """default."""
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def serialize_graph(graph):
    """Make string."""
    json_data = json_graph.node_link_data(graph)
    serial_data = json.dumps(json_data,
                             separators=(',', ':'),
                             indent=4,
                             cls=SetEncoder)
    return serial_data


def draw_graph(graph,
               vertex_label='label',
               secondary_vertex_label=None,
               edge_label='label',
               secondary_edge_label=None,
               vertex_color=None,
               vertex_alpha=0.6,
               vertex_border=False,
               vertex_size=600,

               edge_color=None,
               edge_alpha=0.5,

               dark_edge_color=None,
               dark_edge_dotted=True,
               dark_edge_alpha=0.3,

               size=10,
               size_x_to_y_ratio=1,
               font_size=9,
               layout='graphviz',
               prog='neato',
               pos=None,

               colormap='YlOrRd',
               vmin=0,
               vmax=1,
               invert_colormap=False,
               verbose=True,
               file_name=None,
               title_key='id',
               ignore_for_layout="edge_attribute",

               logscale=False):
    """Plot graph layout."""
    if size is not None:
        size_x = size
        size_y = int(float(size) / size_x_to_y_ratio)
        plt.figure(figsize=(size_x, size_y))
    plt.grid(False)
    plt.axis('off')

    if vertex_label is not None:
        if secondary_vertex_label:
            vertex_labels = dict()
            for u, d in graph.nodes(data=True):
                label1 = _serialize_list(d.get(vertex_label, 'N/A'))
                label2 = _serialize_list(d.get(secondary_vertex_label, 'N/A'))
                vertex_labels[u] = '%s\n%s' % (label1, label2)
        else:
            vertex_labels = dict()
            for u, d in graph.nodes(data=True):
                label = d.get(vertex_label, 'N/A')
                vertex_labels[u] = _serialize_list(label)

    edges_normal = [(u, v) for (u, v, d) in graph.edges(data=True)
                    if d.get('nesting', False) is False]
    edges_nesting = [(u, v) for (u, v, d) in graph.edges(data=True)
                     if d.get('nesting', False) is True]

    if edge_label is not None:
        if secondary_edge_label:
            edge_labels = dict([((u, v,), '%s\n%s' %
                                 (d.get(edge_label, ''),
                                  d.get(secondary_edge_label, '')))
                                for u, v, d in graph.edges(data=True)])
        else:
            edge_labels = dict([((u, v,), d.get(edge_label, ''))
                                for u, v, d in graph.edges(data=True)])

    if vertex_color is None:
        node_color = 'white'
    elif vertex_color in ['_labels_', '_label_', '__labels__', '__label__']:
        node_color = []
        for u, d in graph.nodes(data=True):
            label = d.get('label', '.')
            node_color.append(hash(_serialize_list(label)))
        color_set = set(node_color)
        color_map = {c: i for i, c in enumerate(color_set)}
        node_color = [color_map[c] for c in node_color]
    else:
        if invert_colormap:
            node_color = [- d.get(vertex_color, 0)
                          for u, d in graph.nodes(data=True)]
        else:
            node_color = [d.get(vertex_color, 0)
                          for u, d in graph.nodes(data=True)]
        if logscale is True:
            log_threshold = 0.01
            node_color = [math.log(c) if c > log_threshold
                          else math.log(log_threshold)
                          for c in node_color]

    if edge_color is None:
        edge_colors = 'black'
    elif edge_color in ['_labels_', '_label_', '__labels__', '__label__']:
        edge_colors = [hash(str(d.get('label', '.')))
                       for u, v, d in graph.edges(data=True)
                       if 'nesting' not in d]
    else:
        if invert_colormap:
            edge_colors = [- d.get(edge_color, 0)
                           for u, v, d in graph.edges(data=True)
                           if 'nesting' not in d]
        else:
            edge_colors = [d.get(edge_color, 0)
                           for u, v, d in graph.edges(data=True)
                           if 'nesting' not in d]
    if dark_edge_color is None:
        dark_edge_colors = 'black'
    else:
        dark_edge_colors = [d.get(dark_edge_color, 0)
                            for u, v, d in graph.edges(data=True)
                            if 'nesting' in d]
    tmp_edge_set = [(a, b, d)
                    for (a, b, d) in graph.edges(data=True)
                    if ignore_for_layout in d]
    graph.remove_edges_from(tmp_edge_set)

    if pos is None:
        if layout == 'graphviz':
            graph_copy = graph.copy()
            # remove all attributes for graphviz layout
            for u, d in graph_copy.nodes(data=True):
                graph_copy.node[u] = {}
            for u, v, d in graph_copy.edges(data=True):
                graph_copy.edge[u][v] = {}
            pos = nx.graphviz_layout(graph_copy, prog=prog, args="-Gstart=rand")
        elif layout == "RNA":
            import RNA
            rna_object = RNA.get_xy_coordinates(graph.graph['structure'])
            pos = {i: (rna_object.get(i).X, rna_object.get(i).Y)
                   for i in range(len(graph.graph['structure']))}
        elif layout == 'circular':
            pos = nx.circular_layout(graph)
        elif layout == 'random':
            pos = nx.random_layout(graph)
        elif layout == 'spring':
            pos = nx.spring_layout(graph)
        elif layout == 'shell':
            pos = nx.shell_layout(graph)
        elif layout == 'spectral':
            pos = nx.spectral_layout(graph)
        else:
            raise Exception('Unknown layout format: %s' % layout)

    if vertex_border is False:
        linewidths = 0.001
    else:
        linewidths = 1

    graph.add_edges_from(tmp_edge_set)

    nx.draw_networkx_nodes(graph, pos,
                           node_color=node_color,
                           alpha=vertex_alpha,
                           node_size=vertex_size,
                           linewidths=linewidths,
                           cmap=plt.get_cmap(colormap))

    nx.draw_networkx_edges(graph, pos,
                           edgelist=edges_normal,
                           width=2,
                           edge_color=edge_colors,
                           cmap=plt.get_cmap(colormap),
                           alpha=edge_alpha)
    if dark_edge_dotted:
        style = 'dotted'
    else:
        style = 'solid'
    nx.draw_networkx_edges(graph, pos,
                           edgelist=edges_nesting,
                           width=1,
                           edge_color=dark_edge_colors,
                           style=style,
                           alpha=dark_edge_alpha)
    if edge_label is not None:
        nx.draw_networkx_edge_labels(graph,
                                     pos,
                                     edge_labels=edge_labels,
                                     font_size=font_size)
    if vertex_label is not None:
        nx.draw_networkx_labels(graph,
                                pos,
                                vertex_labels,
                                font_size=font_size,
                                font_color='black')
    if title_key:
        title = str(graph.graph.get(title_key, ''))
        font = FontProperties()
        font.set_family('monospace')
        plt.title(title, fontproperties=font)
    if size is not None:
        # here we decide if we output the image.
        # note: if size is not set, the canvas has been created outside
        # of this function.
        # we wont write on a canvas that we didn't create ourselves.
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name,
                        bbox_inches='tight',
                        transparent=True,
                        pad_inches=0)
            plt.close()


def draw_adjacency_graph(adjacency_matrix,
                         node_color=None,
                         size=10,
                         layout='graphviz',
                         prog='neato',
                         node_size=80,
                         colormap='autumn'):
    """draw_adjacency_graph."""
    graph = nx.from_scipy_sparse_matrix(adjacency_matrix)

    plt.figure(figsize=(size, size))
    plt.grid(False)
    plt.axis('off')

    if layout == 'graphviz':
        pos = nx.graphviz_layout(graph, prog=prog)
    else:
        pos = nx.spring_layout(graph)

    if len(node_color) == 0:
        node_color = 'gray'
    nx.draw_networkx_nodes(graph, pos,
                           node_color=node_color,
                           alpha=0.6,
                           node_size=node_size,
                           cmap=plt.get_cmap(colormap))
    nx.draw_networkx_edges(graph, pos, alpha=0.5)
    plt.show()


# draw a whole set of graphs::
def draw_graph_set(graphs,
                   n_graphs_per_line=5,
                   size=4,
                   edge_label=None,
                   **args):
    """draw_graph_set."""
    graphs = list(graphs)
    while graphs:
        draw_graph_row(graphs[:n_graphs_per_line],
                       n_graphs_per_line=n_graphs_per_line,
                       edge_label=edge_label,
                       size=size, **args)
        graphs = graphs[n_graphs_per_line:]


# draw a row of graphs
def draw_graph_row(graphs, contract=True, n_graphs_per_line=5, size=4, **args):
    """draw_graph_row."""
    count = len(graphs)
    size_y = size
    size_x = size * n_graphs_per_line
    plt.figure(figsize=(size_x, size_y))
    plt.xlim(xmax=3)

    for i in range(count):
        plt.subplot(1, n_graphs_per_line, i + 1)
        g = graphs[i]
        draw_graph(g, size=None, **args)
    plt.show()


def dendrogram(data,
               vectorizer,
               method="ward",
               color_threshold=1,
               size=10,
               filename=None):
    """dendrogram.

    "median","centroid","weighted","single","ward","complete","average"
    """
    if hasattr(data, '__iter__'):
        iterable = data
    else:
        raise Exception('ERROR: Input must be iterable')
    import itertools
    iterable_1, iterable_2 = itertools.tee(iterable)
    # get labels
    labels = []
    for graph in iterable_2:
        label = graph.graph.get('id', None)
        if label:
            labels.append(label)
    # transform input into sparse vectors
    data_matrix = vectorizer.transform(iterable_1)

    # labels
    if not labels:
        labels = [str(i) for i in range(data_matrix.shape[0])]

    # embed high dimensional sparse vectors in 2D
    from sklearn import metrics
    from scipy.cluster.hierarchy import linkage, dendrogram
    distance_matrix = metrics.pairwise.pairwise_distances(data_matrix)
    linkage_matrix = linkage(distance_matrix, method=method)
    plt.figure(figsize=(size, size))
    dendrogram(linkage_matrix,
               color_threshold=color_threshold,
               labels=labels,
               orientation='right')
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def plot_embedding(data_matrix, y,
                   labels=None,
                   image_file_name=None,
                   title=None,
                   cmap='rainbow',
                   density=False):
    """plot_embedding."""
    import matplotlib.pyplot as plt
    from matplotlib import offsetbox
    from PIL import Image
    from eden.embedding import embed_dat_matrix_two_dimensions

    if title is not None:
        plt.title(title)
    if density:
        embed_dat_matrix_two_dimensions(data_matrix,
                                        y=y,
                                        instance_colormap=cmap)
    else:
        plt.scatter(data_matrix[:, 0], data_matrix[:, 1],
                    c=y,
                    cmap=cmap,
                    alpha=.7,
                    s=30,
                    edgecolors='black')
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    if image_file_name is not None:
        num_instances = data_matrix.shape[0]
        ax = plt.subplot(111)
        for i in range(num_instances):
            img = Image.open(image_file_name + str(i) + '.png')
            imagebox = \
                offsetbox.AnnotationBbox(offsetbox.OffsetImage(img, zoom=1),
                                         data_matrix[i],
                                         pad=0,
                                         frameon=False)
            ax.add_artist(imagebox)
    if labels is not None:
        for id in range(data_matrix.shape[0]):
            label = str(labels[id])
            x = data_matrix[id, 0]
            y = data_matrix[id, 1]
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(0, 0),
                         textcoords='offset points')


def plot_embeddings(data_matrix, y,
                    labels=None,
                    save_image_file_name=None,
                    image_file_name=None,
                    size=20,
                    cmap='rainbow',
                    density=False,
                    knn=16,
                    knn_density=16,
                    k_threshold=0.9,
                    metric='rbf',
                    **args):
    """plot_embeddings."""
    import matplotlib.pyplot as plt
    import time

    plt.figure(figsize=(size, size))

    start = time.time()
    from sklearn.decomposition import TruncatedSVD
    data_matrix_ = TruncatedSVD(n_components=2).fit_transform(data_matrix)
    duration = time.time() - start
    plt.subplot(221)
    plot_embedding(data_matrix_, y,
                   labels=labels,
                   title="SVD (%.1f sec)" % duration,
                   cmap=cmap,
                   density=density,
                   image_file_name=image_file_name)

    start = time.time()
    from sklearn import manifold
    from sklearn.metrics.pairwise import pairwise_distances
    distance_matrix = pairwise_distances(data_matrix)
    data_matrix_ = manifold.MDS(n_components=2,
                                n_init=1,
                                max_iter=100,
                                dissimilarity='precomputed').fit_transform(
        distance_matrix)
    duration = time.time() - start
    plt.subplot(222)
    plot_embedding(data_matrix_, y,
                   labels=labels,
                   title="MDS (%.1f sec)" % duration,
                   cmap=cmap,
                   density=density,
                   image_file_name=image_file_name)

    start = time.time()
    from sklearn import manifold
    data_matrix_ = manifold.TSNE(n_components=2,
                                 init='random',
                                 random_state=0).fit_transform(data_matrix)
    duration = time.time() - start
    plt.subplot(223)
    plot_embedding(data_matrix_, y,
                   labels=labels,
                   title="t-SNE (%.1f sec)" % duration,
                   cmap=cmap,
                   density=density,
                   image_file_name=image_file_name)

    start = time.time()
    from eden.embedding import quick_shift_tree_embedding
    tree_embedding_knn = knn
    data_matrix_ = quick_shift_tree_embedding(data_matrix,
                                              knn=tree_embedding_knn,
                                              knn_density=knn_density,
                                              k_threshold=k_threshold,
                                              **args)
    duration = time.time() - start
    plt.subplot(224)
    plot_embedding(data_matrix_,
                   y,
                   labels=labels,
                   title="KQST knn=%d (%.1f sec)" %
                         (knn, duration), cmap=cmap, density=density,
                   image_file_name=image_file_name)

    if save_image_file_name:
        plt.savefig(save_image_file_name)
    else:
        plt.show()
