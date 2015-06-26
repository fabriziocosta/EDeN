import networkx as nx
import pylab as plt
import json
from networkx.readwrite import json_graph
import numpy as np


def draw_graph(graph,
               vertex_label='label',
               secondary_vertex_label=None,
               edge_label='label',
               secondary_edge_label=None,
               vertex_color=None,
               vertex_alpha=0.6,
               edge_color=None,
               edge_alpha=0.5,
               size=10,
               size_x_to_y_ratio=1,
               node_size=600,
               font_size=9,
               layout='graphviz',
               prog='neato',
               node_border=False,
               colormap='YlOrRd',
               vmin=0,
               vmax=1,
               invert_colormap=False,
               verbose=True,
               file_name=None,
               title_key='info'):

    if size is not None:
        size_x = size
        size_y = int(float(size) / size_x_to_y_ratio)
        plt.figure(figsize=(size_x, size_y))
    plt.grid(False)
    plt.axis('off')

    if vertex_label is not None:
        if secondary_vertex_label:
            vertex_labels = dict([(u, '%s\n%s' % (d.get(vertex_label, 'N/A'), d.get(secondary_vertex_label, 'N/A')))
                                  for u, d in graph.nodes(data=True)])
        else:
            vertex_labels = dict([(u, d.get(vertex_label, 'N/A')) for u, d in graph.nodes(data=True)])

    edges_normal = [(u, v) for (u, v, d) in graph.edges(data=True) if d.get('nesting', False) is False]
    edges_nesting = [(u, v) for (u, v, d) in graph.edges(data=True) if d.get('nesting', False) is True]

    if edge_label is not None:
        if secondary_edge_label:
            edge_labels = dict([((u, v, ), '%s\n%s' % (d.get(edge_label, 'N/A'), d.get(secondary_edge_label, 'N/A')))
                                for u, v, d in graph.edges(data=True)])
        else:
            edge_labels = dict([((u, v, ), d.get(edge_label, 'N/A')) for u, v, d in graph.edges(data=True)])

    if vertex_color is None:
        node_color = 'white'
    elif vertex_color == '_labels_':
        node_color = [hash(d.get('label', '.')) for u, d in graph.nodes(data=True)]
    else:
        if invert_colormap:
            node_color = [- d.get(vertex_color, 0) for u, d in graph.nodes(data=True)]
        else:
            node_color = [d.get(vertex_color, 0) for u, d in graph.nodes(data=True)]

    if edge_color is None:
        edge_color = 'black'
    elif edge_color == '_labels_':
        edge_color = [hash(d.get('label', '.')) for u, v, d in graph.edges(data=True)]
    else:
        if invert_colormap:
            edge_color = [- d.get(edge_color, 0) for u, v, d in graph.edges(data=True)]
        else:
            edge_color = [d.get(edge_color, 0) for u, v, d in graph.edges(data=True)]

    if layout == 'graphviz':
        pos = nx.graphviz_layout(graph, prog=prog)
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

    if node_border is False:
        linewidths = 0.001
    else:
        linewidths = 1

    nx.draw_networkx_nodes(graph, pos,
                           node_color=node_color,
                           alpha=vertex_alpha,
                           node_size=node_size,
                           linewidths=linewidths,
                           cmap=plt.get_cmap(colormap))
    if vertex_label is not None:
        nx.draw_networkx_labels(graph, pos, vertex_labels, font_size=font_size, font_color='black')
    nx.draw_networkx_edges(graph, pos,
                           edgelist=edges_normal,
                           width=2,
                           edge_color=edge_color,
                           cmap=plt.get_cmap(colormap),
                           alpha=edge_alpha)
    nx.draw_networkx_edges(graph, pos,
                           edgelist=edges_nesting,
                           width=1,
                           edge_color='k',
                           style='dashed',
                           alpha=edge_alpha)
    if edge_label is not None:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=font_size)
    if title_key:
        title = str(graph.graph.get(title_key, ''))
        plt.title(title)
    if size is not None:
        # here we decide if we output the image.
        # note: if size is not set, the canvas has been created outside of this function.
        # we wont write on a canvas that we didn't create ourselves.
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name, bbox_inches='tight', transparent=True, pad_inches=0)
            plt.close()


def draw_adjacency_graph(adjacency_matrix,
                         node_color=None,
                         size=10,
                         layout='graphviz',
                         prog='neato',
                         node_size=80,
                         colormap='autumn'):

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


class SetEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def serialize_graph(graph):
    json_data = json_graph.node_link_data(graph)
    serial_data = json.dumps(json_data, separators=(',', ':'), indent = 4, cls = SetEncoder)
    return serial_data


def embed_two_dimensions(data, vectorizer, size=10, n_components=5, colormap='YlOrRd'):
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

    # transform iterable into sparse vectors
    data_matrix = vectorizer.transform(iterable_1)
    # embed high dimensional sparse vectors in 2D
    from sklearn import metrics
    distance_matrix = metrics.pairwise.pairwise_distances(data_matrix)

    from sklearn.manifold import MDS
    feature_map = MDS(n_components=n_components, dissimilarity='precomputed')
    explicit_data_matrix = feature_map.fit_transform(distance_matrix)

    from sklearn.decomposition import TruncatedSVD
    pca = TruncatedSVD(n_components=2)
    low_dimension_data_matrix = pca.fit_transform(explicit_data_matrix)

    plt.figure(figsize=(size, size))
    embed_dat_matrix_two_dimensions(low_dimension_data_matrix, labels=labels, density_colormap=colormap)
    plt.show()


def embed_dat_matrix_two_dimensions(low_dimension_data_matrix, y=None, labels=None, density_colormap='Blues', instance_colormap='YlOrRd'):
    from sklearn.preprocessing import scale
    low_dimension_data_matrix = scale(low_dimension_data_matrix)
    # make mesh
    x_min, x_max = low_dimension_data_matrix[:, 0].min(), low_dimension_data_matrix[:, 0].max()
    y_min, y_max = low_dimension_data_matrix[:, 1].min(), low_dimension_data_matrix[:, 1].max()
    step_num = 50
    h = min((x_max - x_min) / step_num, (y_max - y_min) / step_num)  # step size in the mesh
    b = h * 10  # border size
    x_min, x_max = low_dimension_data_matrix[:, 0].min() - b, low_dimension_data_matrix[:, 0].max() + b
    y_min, y_max = low_dimension_data_matrix[:, 1].min() - b, low_dimension_data_matrix[:, 1].max() + b
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # induce a one class model to estimate densities
    from sklearn.svm import OneClassSVM
    gamma = max(x_max - x_min, y_max - y_min)
    clf = OneClassSVM(gamma=gamma, nu=0.1)
    clf.fit(low_dimension_data_matrix)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max] . [y_min, y_max].
    if hasattr(clf, "decision_function"):
        score_matrix = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        score_matrix = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    # Put the result into a color plot
    levels = np.linspace(min(score_matrix), max(score_matrix), 40)
    score_matrix = score_matrix.reshape(xx.shape)

    if y is None:
        y = 'white'

    plt.contourf(xx, yy, score_matrix, cmap=plt.get_cmap(density_colormap), alpha=0.9, levels=levels)
    plt.scatter(low_dimension_data_matrix[:, 0], low_dimension_data_matrix[:, 1],
                alpha=.5,
                s=70,
                edgecolors='gray',
                c=y,
                cmap=plt.get_cmap(instance_colormap))
    # labels
    if labels is not None:
        for id in range(low_dimension_data_matrix.shape[0]):
            label = labels[id]
            x = low_dimension_data_matrix[id, 0]
            y = low_dimension_data_matrix[id, 1]
            plt.annotate(label, xy=(x, y), xytext = (0, 0), textcoords = 'offset points')


def dendrogram(data, vectorizer, method="ward", color_threshold=1, size=10, filename=None):
    '"median","centroid","weighted","single","ward","complete","average"'
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
    dendrogram(linkage_matrix, color_threshold=color_threshold, labels=labels, orientation='right')
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def quick_shift_tree_embedding(data_matrix, knn=10, knn_density=None, k_threshold=0.9, metric='linear', **args):
    if knn_density is None:
        knn_density = knn
    n_instances = data_matrix.shape[0]
    # extract pairwise similarity matrix with desired kernel
    from sklearn import metrics
    kernel_matrix = metrics.pairwise.pairwise_kernels(data_matrix, metric=metric, **args)
    # compute instance density as average pairwise similarity
    import numpy as np
    density = np.sum(kernel_matrix, 0) / n_instances
    # compute list of nearest neighbors
    kernel_matrix_sorted = np.argsort(-kernel_matrix)
    # make matrix of densities ordered by nearest neighbor
    density_matrix = density[kernel_matrix_sorted]
    parent_dict = {}
    # for all instances determine parent link
    for i, row in enumerate(density_matrix):
        i_density = row[0]
        # if a densed neighbor cannot be found then assign parent to the instance itself
        parent_dict[i] = i
        # for all neighbors from the closest to the furthest
        for jj, d in enumerate(row):
            # proceed until k neighbors have been explored
            if jj > knn_density:
                break
            j = kernel_matrix_sorted[i, jj]
            if jj > 0:
                j_density = d
                # if the density of the neighbor is higher than the density of the instance assign parent
                if j_density > i_density:
                    parent_dict[i] = j
                    break
    # make a graph with instances as nodes
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(range(n_instances))
    # add edge between instance and parent
    for i in range(n_instances):
        j = parent_dict[i]
        graph.add_edge(i, j, weight=1)
    # determine threshold as k-th quantile on pairwise similarity on the knn similarity
    knn_similarities = kernel_matrix[kernel_matrix_sorted[:, knn]]
    # vectorized_pairwise_similarity = np.ravel(kernel_matrix)
    k_quantile = np.percentile(knn_similarities, k_threshold * 100)
    # add edge between instance and k-th nearest neighbor if similarity > threshold
    for i in range(n_instances):
        # id of k-th nearest neighbor
        jd = kernel_matrix_sorted[i, knn]
        # similarity of k-th nearest neighbor
        kd = kernel_matrix[i, jd]
        if kd > k_quantile:
            graph.add_edge(i, jd, weight=1)
    # use graph layout algorithm to determine coordinates
    two_dimensional_data_matrix = nx.graphviz_layout(graph, prog='sfdp', args='-Goverlap=scale')
    two_dimensional_data_list = []
    for i in range(kernel_matrix.shape[0]):
        two_dimensional_data_list.append(list(two_dimensional_data_matrix[i]))
    embedding_data_matrix = np.array(two_dimensional_data_list)
    from sklearn.preprocessing import scale
    return scale(embedding_data_matrix)


def plot_embedding(data_matrix, y, labels=None, image_file_name=None, title=None, cmap='gnuplot', density=False):
    import matplotlib.pyplot as plt
    from matplotlib import offsetbox
    from PIL import Image

    if title is not None:
        plt.title(title)
    if density:
        embed_dat_matrix_two_dimensions(data_matrix, y=y, instance_colormap=cmap)
    else:
        plt.scatter(data_matrix[:, 0], data_matrix[:, 1], c=y, cmap=cmap, alpha=.7, s=30, edgecolors='gray')
        plt.xticks([])
        plt.yticks([])
    if image_file_name is not None:
        num_instances = data_matrix.shape[0]
        ax = plt.subplot(111)
        for i in range(num_instances):
            img = Image.open(image_file_name + str(i) + '.png')
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img, zoom=1), data_matrix[i], pad=0, frameon=False)
            ax.add_artist(imagebox)
    if labels is not None:
        for id in range(data_matrix.shape[0]):
            label = str(labels[id])
            x = data_matrix[id, 0]
            y = data_matrix[id, 1]
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords = 'offset points')


def plot_embeddings(data_matrix, y, labels=None, save_image_file_name=None, image_file_name=None, size=25, cmap='gnuplot', density=False, knn=16, knn_density=16, k_threshold=0.9, metric='rbf', **args):
    import matplotlib.pyplot as plt
    import time

    plt.figure(figsize=(size, size))

    start = time.time()
    from sklearn import decomposition
    data_matrix_ = decomposition.TruncatedSVD(n_components=2).fit_transform(data_matrix)
    duration = time.time() - start
    plt.subplot(321)
    plot_embedding(data_matrix_, y, labels=labels, title="SVD (%.1f sec)" % duration, cmap=cmap, density=density, image_file_name=image_file_name)

    start = time.time()
    from sklearn import manifold
    from sklearn.metrics.pairwise import pairwise_distances
    distance_matrix = pairwise_distances(data_matrix)
    data_matrix_ = manifold.MDS(n_components=2, n_init=1, max_iter=100, dissimilarity='precomputed').fit_transform(distance_matrix)
    duration = time.time() - start
    plt.subplot(322)
    plot_embedding(data_matrix_, y, labels=labels, title="MDS (%.1f sec)" % duration, cmap=cmap, density=density, image_file_name=image_file_name)

    start = time.time()
    from sklearn import manifold
    data_matrix_ = manifold.TSNE(n_components=2, init='random', random_state=0).fit_transform(data_matrix)
    duration = time.time() - start
    plt.subplot(323)
    plot_embedding(data_matrix_, y, labels=labels, title="t-SNE (%.1f sec)" % duration, cmap=cmap, density=density, image_file_name=image_file_name)

    start = time.time()
    from eden.util.display import quick_shift_tree_embedding
    tree_embedding_knn = knn / 4
    data_matrix_ = quick_shift_tree_embedding(data_matrix, knn=tree_embedding_knn, knn_density=knn_density / 4, k_threshold=k_threshold, metric=metric, **args)
    duration = time.time() - start
    plt.subplot(324)
    plot_embedding(data_matrix_, y, labels=labels, title="KQST knn=%d (%.1f sec)" %
                   (tree_embedding_knn, duration), cmap=cmap, density=density, image_file_name=image_file_name)

    start = time.time()
    from eden.util.display import quick_shift_tree_embedding
    tree_embedding_knn = knn
    data_matrix_ = quick_shift_tree_embedding(data_matrix, knn=tree_embedding_knn, knn_density=knn_density, k_threshold=k_threshold, metric=metric, **args)
    duration = time.time() - start
    plt.subplot(325)
    plot_embedding(data_matrix_, y, labels=labels, title="KQST knn=%d (%.1f sec)" %
                   (knn, duration), cmap=cmap, density=density, image_file_name=image_file_name)

    start = time.time()
    from eden.util.display import quick_shift_tree_embedding
    tree_embedding_knn = knn * 2
    data_matrix_ = quick_shift_tree_embedding(data_matrix, knn=tree_embedding_knn, knn_density=knn_density * 2, k_threshold=k_threshold, metric=metric, **args)
    duration = time.time() - start
    plt.subplot(326)
    plot_embedding(data_matrix_, y, labels=labels, title="KQST knn=%d (%.1f sec)" %
                   (tree_embedding_knn, duration), cmap=cmap, density=density, image_file_name=image_file_name)
    if save_image_file_name:
        plt.savefig(save_image_file_name)
    else:
        plt.show()


# draw a whole set of graphs::
def draw_graph_set(graphs, n_graphs_per_line=5, size=4, edge_label=None, **args):
    graphs = list(graphs)
    while graphs:
        draw_graph_row(graphs[:n_graphs_per_line], n_graphs_per_line=n_graphs_per_line, edge_label=edge_label, size=size, **args)
        graphs = graphs[n_graphs_per_line:]


# draw a row of graphs
def draw_graph_row(graphs, contract=True, n_graphs_per_line=5, size=4, headlinehook=lambda x: "", **args):
    count = len(graphs)
    size_y = size
    size_x = size * n_graphs_per_line
    plt.figure(figsize=(size_x, size_y))
    plt.xlim(xmax=3)

    for i in range(count):
        plt.subplot(1, n_graphs_per_line, i + 1)
        graphs[i].graph['info'] = headlinehook(graphs[i])
        g = graphs[i]
        draw_graph(g, size=None, **args)
    plt.show()
