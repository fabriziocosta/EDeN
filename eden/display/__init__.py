#!/usr/bin/env python
"""Provides drawing utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pylab as plt
import math
import networkx as nx
import json
from networkx.readwrite import json_graph
from matplotlib.font_manager import FontProperties

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from eden.util import _serialize_list
from eden.display.graph_layout import KKEmbedder
from sklearn.preprocessing import minmax_scale
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


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


def map_labels_to_colors(graphs):
    """Map all node labels into a real in [0,1]."""
    label_set = set()
    for g in graphs:
        for u in g.nodes():
            label_set.add(g.nodes[u]['label'])
    dim = len(label_set)
    label_colors = dict()
    for i, label in enumerate(sorted(label_set)):
        label_colors[label] = float(i) / dim
    return label_colors


def draw_graph(graph,
               vertex_label='label',
               vertex_color=None,
               vertex_color_dict=None,
               vertex_fixed_color=None,
               vertex_alpha=0.6,
               vertex_border=1,
               vertex_size=600,
               vertex_images=None,
               vertex_image_scale=0.1,
               compact=False,
               colormap='YlOrRd',
               vmin=0,
               vmax=1,
               invert_colormap=False,
               secondary_vertex_label=None,
               secondary_vertex_color=None,
               secondary_vertex_fixed_color=None,
               secondary_vertex_alpha=0.6,
               secondary_vertex_border=1,
               secondary_vertex_size=600,
               secondary_vertex_colormap='YlOrRd',
               secondary_vertex_vmin=0,
               secondary_vertex_vmax=1,

               edge_label='label',
               secondary_edge_label=None,
               edge_colormap='YlOrRd',
               edge_vmin=0,
               edge_vmax=1,
               edge_color=None,
               edge_fixed_color=None,
               edge_width=None,
               edge_alpha=0.5,

               dark_edge_colormap='YlOrRd',
               dark_edge_vmin=0,
               dark_edge_vmax=1,
               dark_edge_color=None,
               dark_edge_fixed_color=None,
               dark_edge_dotted=True,
               dark_edge_alpha=0.3,

               size=10,
               size_x_to_y_ratio=1,
               font_size=9,
               layout='graphviz',
               prog='neato',
               pos=None,

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
        axes = plt.gca()
    plt.grid(False)
    plt.axis('off')
    plt.axis('equal')

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
            if vertex_color_dict is not None:
                node_color.append(vertex_color_dict.get(label, 0))
            else:
                node_color.append(hash(_serialize_list(label)))
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
    if edge_width is None:
        widths = 1
    elif isinstance(edge_width, int):
        widths = edge_width
    else:
        widths = [d.get(edge_width, 1)
                  for u, v, d in graph.edges(data=True)
                  if 'nesting' not in d]
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
    tmp_edge_set = [(u, v)
                    for u, v in graph.edges()
                    if graph.edges[u, v].get(ignore_for_layout, False)]
    if tmp_edge_set:
        graph.remove_edges_from(tmp_edge_set)

    if pos is None:
        if layout == 'graphviz':
            graph_copy = graph.copy()
            pos = nx.nx_pydot.graphviz_layout(graph_copy,
                                              prog=prog)
        elif layout == "RNA":
            import RNA  # this is part of the vienna RNA package
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
        elif layout == 'KK':
            pos = KKEmbedder().transform(graph)
        else:
            raise Exception('Unknown layout format: %s' % layout)
        _pos = minmax_scale(np.array([pos[i] for i in pos]))
        pos = {i: (p[0], p[1]) for i, p in zip(pos, _pos)}

    if vertex_border is False:
        linewidths = 0.001
    else:
        linewidths = vertex_border

    if tmp_edge_set:
        graph.add_edges_from(tmp_edge_set)

    if secondary_vertex_border is False:
        secondary_linewidths = 0.001
    else:
        secondary_linewidths = secondary_vertex_border
    if secondary_vertex_fixed_color is not None:
        secondary_node_color = secondary_vertex_fixed_color
    if secondary_vertex_color is not None:
        secondary_node_color = [d.get(secondary_vertex_color, 0)
                                for u, d in graph.nodes(data=True)]
    if secondary_vertex_fixed_color is not None or \
            secondary_vertex_color is not None:
        secondary_nodes = nx.draw_networkx_nodes(
            graph, pos,
            node_color=secondary_node_color,
            alpha=secondary_vertex_alpha,
            node_size=secondary_vertex_size,
            linewidths=secondary_linewidths,
            cmap=plt.get_cmap(
                secondary_vertex_colormap),
            vmin=secondary_vertex_vmin, vmax=secondary_vertex_vmax)
        secondary_nodes.set_edgecolor('k')
    if vertex_fixed_color is not None:
        node_color = vertex_fixed_color
    if compact:
        nodes = nx.draw_networkx_nodes(graph, pos,
                                       node_color='w',
                                       alpha=1,
                                       node_size=vertex_size,
                                       linewidths=linewidths)
        nodes.set_edgecolor('k')
        nx.draw_networkx_nodes(graph, pos,
                               node_color=node_color,
                               alpha=vertex_alpha,
                               node_size=vertex_size,
                               linewidths=None,
                               cmap=plt.get_cmap(colormap),
                               vmin=vmin, vmax=vmax)

    else:
        nodes = nx.draw_networkx_nodes(graph, pos,
                                       node_color=node_color,
                                       alpha=vertex_alpha,
                                       node_size=vertex_size,
                                       linewidths=linewidths,
                                       cmap=plt.get_cmap(colormap),
                                       vmin=vmin, vmax=vmax)
        nodes.set_edgecolor('k')

    if edge_fixed_color is not None:
        edge_colors = edge_fixed_color
    nx.draw_networkx_edges(graph, pos,
                           edgelist=edges_normal,
                           width=widths,
                           edge_color=edge_colors,
                           edge_cmap=plt.get_cmap(edge_colormap),
                           edge_vmin=edge_vmin, edge_vmax=edge_vmax,
                           alpha=edge_alpha)
    if dark_edge_dotted:
        style = 'dotted'
    else:
        style = 'solid'
    if dark_edge_fixed_color is not None:
        dark_edge_colors = dark_edge_fixed_color
    nx.draw_networkx_edges(graph, pos,
                           edgelist=edges_nesting,
                           width=1,
                           edge_cmap=plt.get_cmap(dark_edge_colormap),
                           edge_vmin=dark_edge_vmin, edge_vmax=dark_edge_vmax,
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
                                font_weight='normal',
                                font_color='black')
    if vertex_images is not None:
        for im, xy_pos in zip(vertex_images, pos):
            x, y = pos[xy_pos]
            oi = OffsetImage(im, zoom=vertex_image_scale, alpha=.5)
            box = AnnotationBbox(oi, (x, y), frameon=False)
            axes.add_artist(box)
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
                   pos=None,
                   **args):
    """draw_graph_set."""
    graphs = list(graphs)
    if pos:
        for graph, pos_dict in zip(graphs, pos):
            graph.graph['pos_dict'] = pos_dict

    counter = 0
    while graphs:
        counter += 1
        draw_graph_row(graphs[:n_graphs_per_line],
                       index=counter,
                       n_graphs_per_line=n_graphs_per_line,
                       edge_label=edge_label,
                       size=size, **args)
        graphs = graphs[n_graphs_per_line:]

# draw a row of graphs


def draw_graph_row(graphs,
                   index=0,
                   contract=True,
                   n_graphs_per_line=5,
                   size=4,
                   xlim=None,
                   ylim=None,
                   **args):
    """draw_graph_row."""
    dim = len(graphs)
    size_y = size
    size_x = size * n_graphs_per_line * args.get('size_x_to_y_ratio', 1)
    plt.figure(figsize=(size_x, size_y))

    if xlim is not None:
        plt.xlim(xlim)
        plt.ylim(ylim)
    else:
        plt.xlim(xmax=3)

    for i in range(dim):
        plt.subplot(1, n_graphs_per_line, i + 1)
        graph = graphs[i]
        draw_graph(graph,
                   size=None,
                   pos=graph.graph.get('pos_dict', None),
                   **args)
    if args.get('file_name', None) is None:
        plt.show()
    else:
        row_file_name = '%d_' % (index) + args['file_name']
        plt.savefig(row_file_name,
                    bbox_inches='tight',
                    transparent=True,
                    pad_inches=0)
        plt.close()


def dendrogram(data,
               vectorizer,
               method="ward",
               color_threshold=1,
               size=10,
               filename=None):
    """dendrogram.

    "median","centroid","weighted","single","ward","complete","average"
    """
    data = list(data)
    # get labels
    labels = []
    for graph in data:
        label = graph.graph.get('id', None)
        if label:
            labels.append(label)
    # transform input into sparse vectors
    data_matrix = vectorizer.transform(data)

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
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(img, zoom=1),
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
    if data_matrix.shape[1] > 2:
        from sklearn.decomposition import TruncatedSVD
        data_matrix_ = TruncatedSVD(n_components=2).fit_transform(data_matrix)
    else:
        data_matrix_ = data_matrix
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


def heatmap(values, xlabel, ylabel, xticklabels, yticklabels, cmap=None,
            vmin=None, vmax=None, ax=None, fmt="%0.2f"):
    """heatmap."""
    if ax is None:
        ax = plt.gca()
    # plot the mean cross-validation scores
    img = ax.pcolor(values, cmap=cmap, vmin=vmin, vmax=vmax)
    img.update_scalarmappable()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(xticklabels)) + .5)
    ax.set_yticks(np.arange(len(yticklabels)) + .5)
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_aspect(1)

    for p, color, value in zip(img.get_paths(),
                               img.get_facecolors(),
                               img.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.mean(color[:3]) > 0.5:
            c = 'k'
        else:
            c = 'w'
        ax.text(x, y, fmt % value, color=c, ha="center", va="center")
    return img


def plot_confusion_matrix(y_true, y_pred, size=None, normalize=False):
    """plot_confusion_matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fmt = "%d"
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = "%.2f"
    xticklabels = list(sorted(set(y_pred)))
    yticklabels = list(sorted(set(y_true)))
    if size is not None:
        plt.figure(figsize=(size, size))
    heatmap(cm, xlabel='Predicted label', ylabel='True label',
            xticklabels=xticklabels, yticklabels=yticklabels,
            cmap=plt.cm.Blues, fmt=fmt)
    if normalize:
        plt.title("Confusion matrix (norm.)")
    else:
        plt.title("Confusion matrix")
    plt.gca().invert_yaxis()


def plot_confusion_matrices(y_true, y_pred, size=12):
    """plot_confusion_matrices."""
    plt.figure(figsize=(size, size))
    plt.subplot(121)
    plot_confusion_matrix(y_true, y_pred, normalize=False)
    plt.subplot(122)
    plot_confusion_matrix(y_true, y_pred, normalize=True)
    plt.tight_layout(w_pad=5)
    plt.show()


def plot_precision_recall_curve(y_true, y_score, size=None):
    """plot_precision_recall_curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if size is not None:
        plt.figure(figsize=(size, size))
        plt.axis('equal')
    plt.plot(recall, precision, lw=2, color='navy')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.grid()
    plt.title('Precision-Recall AUC={0:0.2f}'.format(average_precision_score(
        y_true, y_score)))


def plot_roc_curve(y_true, y_score, size=None):
    """plot_roc_curve."""
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_true, y_score)
    if size is not None:
        plt.figure(figsize=(size, size))
        plt.axis('equal')
    plt.plot(false_positive_rate, true_positive_rate, lw=2, color='navy')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.ylim([-0.05, 1.05])
    plt.xlim([-0.05, 1.05])
    plt.grid()
    plt.title('Receiver operating characteristic AUC={0:0.2f}'.format(
        roc_auc_score(y_true, y_score)))


def plot_aucs(y_true, y_score, size=12):
    """plot_confusion_matrices."""
    plt.figure(figsize=(size, size / 2.0))
    plt.subplot(121, aspect='equal')
    plot_roc_curve(y_true, y_score)
    plt.subplot(122, aspect='equal')
    plot_precision_recall_curve(y_true, y_score)
    plt.tight_layout(w_pad=5)
    plt.show()
