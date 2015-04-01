import networkx as nx
import pylab as plt
import json
from networkx.readwrite import json_graph
from collections import defaultdict
import numpy as np


def draw_graph(graph,
               vertex_label='label',
               secondary_vertex_label=None,
               edge_label='label',
               secondary_edge_label=None,
               vertex_color=None,
               vertex_alpha=0.6,
               size=10,
               size_x_to_y_ratio=1,
               node_size=600,
               font_size=9,
               layout='graphviz',
               prog='neato',
               node_border=False,
               colormap='YlOrRd',
               invert_colormap=False,
               verbose=True,
               file_name=None):

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

    edges_normal = [(u, v) for (u, v, d) in graph.edges(data=True) if d.get('nesting', False) == False]
    edges_nesting = [(u, v) for (u, v, d) in graph.edges(data=True) if d.get('nesting', False) == True]

    if edge_label is not None:
        if secondary_edge_label:
            edge_labels = dict([((u, v, ), '%s\n%s' % (d.get(edge_label, 'N/A'), d.get(secondary_edge_label, 'N/A')))
                                for u, v, d in graph.edges(data=True)])
        else:
            edge_labels = dict([((u, v, ), d.get(edge_label, 'N/A')) for u, v, d in graph.edges(data=True)])

    if vertex_color is None:
        node_color = 'white'
    else:
        if invert_colormap:
            node_color = [- d.get(vertex_color, 0) for u, d in graph.nodes(data=True)]
        else:
            node_color = [d.get(vertex_color, 0) for u, d in graph.nodes(data=True)]

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

    if node_border == False:
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
                           edge_color='k',
                           alpha=0.5)
    nx.draw_networkx_edges(graph, pos,
                           edgelist=edges_nesting,
                           width=1,
                           edge_color='k',
                           style='dashed',
                           alpha=0.5)
    if edge_label is not None:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=font_size)
    if verbose:
        title = str(graph.graph.get('id', '')) + "\n" + str(graph.graph.get('info', ''))
        plt.title(title)
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name, bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close()


def draw_adjacency_graph(A,
                         node_color=None,
                         size=10,
                         layout='graphviz',
                         prog='neato',
                         node_size=80,
                         colormap='autumn'):

    graph = nx.from_scipy_sparse_matrix(A)

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


def embed2D(data, vectorizer, size=10, n_components=5, n_jobs=1, colormap='YlOrRd'):
    import numpy as np
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
    X = vectorizer.transform(iterable_1, n_jobs=n_jobs)
    # embed high dimensional sparse vectors in 2D
    from sklearn import metrics
    D = metrics.pairwise.pairwise_distances(X)

    from sklearn.manifold import MDS
    feature_map = MDS(n_components=n_components, dissimilarity='precomputed')
    X_explicit = feature_map.fit_transform(D)

    from sklearn.decomposition import TruncatedSVD
    pca = TruncatedSVD(n_components=2)
    X_reduced = pca.fit_transform(X_explicit)

    plt.figure(figsize=(size, size))
    embed_dat_matrix_2D(X_reduced, labels=labels, density_colormap=colormap)
    plt.show()


def embed_dat_matrix_2D(X_reduced, y=None, labels=None, density_colormap='Blues', instance_colormap='YlOrRd'):
    from sklearn.preprocessing import scale
    X_reduced = scale(X_reduced)
    # make mesh
    x_min, x_max = X_reduced[:, 0].min(), X_reduced[:, 0].max()
    y_min, y_max = X_reduced[:, 1].min(), X_reduced[:, 1].max()
    step_num = 50
    h = min((x_max - x_min) / step_num, (y_max - y_min) / step_num)  # step size in the mesh
    b = h * 10  # border size
    x_min, x_max = X_reduced[:, 0].min() - b, X_reduced[:, 0].max() + b
    y_min, y_max = X_reduced[:, 1].min() - b, X_reduced[:, 1].max() + b
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # induce a one class model to estimate densities
    from sklearn.svm import OneClassSVM
    gamma = max(x_max - x_min, y_max - y_min)
    clf = OneClassSVM(gamma=gamma, nu=0.1)
    clf.fit(X_reduced)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max] . [y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    # Put the result into a color plot
    levels = np.linspace(min(Z), max(Z), 40)
    Z = Z.reshape(xx.shape)

    if y is None:
        y = 'white'

    plt.contourf(xx, yy, Z, cmap=plt.get_cmap(density_colormap), alpha=0.9, levels=levels)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
                alpha=.5,
                s=70,
                edgecolors='gray',
                c=y,
                cmap=plt.get_cmap(instance_colormap))
    # labels
    if labels is not None:
        for id in range(X_reduced.shape[0]):
            label = labels[id]
            x = X_reduced[id, 0]
            y = X_reduced[id, 1]
            plt.annotate(label, xy=(x, y), xytext = (0, 0), textcoords = 'offset points')


def dendrogram(data, vectorizer, method="ward", color_threshold=1, size=10, n_jobs=1, filename=None):
    '"median","centroid","weighted","single","ward","complete","average"'
    import numpy as np
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
    X = vectorizer.transform(iterable_1, n_jobs=n_jobs)

    # labels
    if not labels:
        labels = [str(i) for i in range(X.shape[0])]

    # embed high dimensional sparse vectors in 2D
    from sklearn import metrics
    from scipy.cluster.hierarchy import linkage, dendrogram
    D = metrics.pairwise.pairwise_distances(X)
    Z = linkage(D, method=method)
    plt.figure(figsize=(size, size))
    dendrogram(Z, color_threshold=color_threshold, labels=labels, orientation='right')
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def KernelQuickShiftTreeEmbedding(X, knn=10, knn_density=None, k_threshold=0.9, metric='linear', **args):
    if knn_density is None:
        knn_density = knn
    n_instances = X.shape[0]
    # extract pairwise similarity matrix with desired kernel
    from sklearn import metrics
    K = metrics.pairwise.pairwise_kernels(X, metric=metric, **args)
    # compute instance density as average pairwise similarity
    import numpy as np
    density = np.sum(K, 0) / n_instances
    # compute list of nearest neighbors
    Ka = np.argsort(-K)
    # make matrix of densities ordered by nearest neighbor
    Kad = density[Ka]
    parent_dict = {}
    # for all instances determine parent link
    for i, row in enumerate(Kad):
        i_density = row[0]
        # if a densed neighbor cannot be found then assign parent to the instance itself
        parent_dict[i] = i
        # for all neighbors from the closest to the furthest
        for jj, d in enumerate(row):
            # proceed until k neighbors have been explored
            if jj > knn_density:
                break
            j = Ka[i, jj]
            if jj > 0:
                j_density = d
                # if the density of the neighbor is higher than the density of the instance assign parent
                if j_density > i_density:
                    parent_dict[i] = j
                    break
    # make a graph with instances as nodes
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(range(n_instances))
    # add edge between instance and parent
    for i in range(n_instances):
        j = parent_dict[i]
        G.add_edge(i, j, weight=1)
    # determine threshold as k-th quantile on pairwise similarity on the knn similarity
    knn_similarities = K[Ka[:, knn]]
    # vectorized_pairwise_similarity = np.ravel(K)
    k_quantile = np.percentile(knn_similarities, k_threshold * 100)
    # add edge between instance and k-th nearest neighbor if similarity > threshold
    for i in range(n_instances):
        # id of k-th nearest neighbor
        jd = Ka[i, knn]
        # similarity of k-th nearest neighbor
        kd = K[i, jd]
        if kd > k_quantile:
            G.add_edge(i, jd, weight=1)
    # use graph layout algorithm to determine coordinates
    X_ = nx.graphviz_layout(G, prog='sfdp', args='-Goverlap=scale')
    X_2D = []
    for i in range(K.shape[0]):
        X_2D.append(list(X_[i]))
    X_emb = np.array(X_2D)
    from sklearn.preprocessing import scale
    return scale(X_emb)


def plot_embedding(X, y, labels=None, image_file_name=None, title=None, cmap='gnuplot', density=False):
    import matplotlib.pyplot as plt
    from matplotlib import offsetbox
    from PIL import Image
    import numpy as np

    if title is not None:
        plt.title(title)
    if density:
        embed_dat_matrix_2D(X, y=y, instance_colormap=cmap)
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, alpha=.7, s=30, edgecolors='gray')
        plt.xticks([])
        plt.yticks([])
    if image_file_name is not None:
        num_instances = X.shape[0]
        large_images = np.array([[1., 1]])
        ax = plt.subplot(111)
        for i in range(num_instances):
            img = Image.open(image_file_name + str(i) + '.png')
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img, zoom=1), X[i], pad=0, frameon=False)
            ax.add_artist(imagebox)
    if labels is not None:
        for id in range(X.shape[0]):
            label = str(labels[id])
            x = X[id, 0]
            y = X[id, 1]
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords = 'offset points')


def plot_embeddings(X, y, labels=None, image_file_name=None, size=25, cmap='gnuplot', density=False, knn=16, knn_density=16, k_threshold=0.9, metric='rbf', **args):
    import matplotlib.pyplot as plt
    import time

    plt.figure(figsize=(size, size))

    start = time.time()
    from sklearn import decomposition
    X_ = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
    duration = time.time() - start
    plt.subplot(321)
    plot_embedding(X_, y, labels=labels, title="SVD (%.1f sec)" % duration, cmap=cmap, density=density, image_file_name=image_file_name)

    start = time.time()
    from sklearn import manifold
    X_ = manifold.MDS(n_components=2, n_init=1, max_iter=100).fit_transform(X)
    duration = time.time() - start
    plt.subplot(322)
    plot_embedding(X_, y, labels=labels, title="MDS (%.1f sec)" % duration, cmap=cmap, density=density, image_file_name=image_file_name)

    start = time.time()
    from sklearn import manifold
    X_ = manifold.TSNE(n_components=2, init='random', random_state=0).fit_transform(X)
    duration = time.time() - start
    plt.subplot(323)
    plot_embedding(X_, y, labels=labels, title="t-SNE (%.1f sec)" % duration, cmap=cmap, density=density, image_file_name=image_file_name)

    start = time.time()
    from eden.util.display import KernelQuickShiftTreeEmbedding
    X_ = KernelQuickShiftTreeEmbedding(X, knn=knn / 4, knn_density=knn_density / 4, k_threshold=k_threshold, metric=metric, **args)
    duration = time.time() - start
    plt.subplot(324)
    plot_embedding(X_, y, labels=labels, title="KQST knn=%d (%.1f sec)" %
                   (knn / 4, duration), cmap=cmap, density=density, image_file_name=image_file_name)

    start = time.time()
    from eden.util.display import KernelQuickShiftTreeEmbedding
    X_ = KernelQuickShiftTreeEmbedding(X, knn=knn, knn_density=knn_density, k_threshold=k_threshold, metric=metric, **args)
    duration = time.time() - start
    plt.subplot(325)
    plot_embedding(X_, y, labels=labels, title="KQST knn=%d (%.1f sec)" %
                   (knn, duration), cmap=cmap, density=density, image_file_name=image_file_name)

    start = time.time()
    from eden.util.display import KernelQuickShiftTreeEmbedding
    X_ = KernelQuickShiftTreeEmbedding(X, knn=knn * 2, knn_density=knn_density * 2, k_threshold=k_threshold, metric=metric, **args)
    duration = time.time() - start
    plt.subplot(326)
    plot_embedding(X_, y, labels=labels, title="KQST knn=%d (%.1f sec)" %
                   (knn * 2, duration), cmap=cmap, density=density, image_file_name=image_file_name)

    plt.show()
