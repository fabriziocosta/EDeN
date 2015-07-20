import pylab as plt
import numpy as np
from eden.util.display import plot_embeddings


def matrix_factorization(data_matrix, n=10):
    import pymf
    mf = pymf.SIVM(data_matrix.T, num_bases=n)
    mf.factorize()
    return mf.W.T, mf.H.T


def low_dimensional_embedding(data_matrix, low_dim=None):
    from sklearn import random_projection
    n_rows, n_cols = data_matrix.shape
    # perform data dimension reduction inly if #features > #data points
    if n_cols <= n_rows:
        return_data_matrix = data_matrix
    else:
        if n_rows < 5000:
            n_components = n_rows
        else:
            n_components = 'auto'
        transformer = random_projection.SparseRandomProjection(n_components=n_components, dense_output=True)
        data_matrix_new = transformer.fit_transform(data_matrix)
        basis_data_matrix, coordinates_data_matrix = matrix_factorization(data_matrix_new, n=low_dim)
        return_data_matrix = coordinates_data_matrix
    return return_data_matrix


def embedding_quality(data_matrix, y, opts, low_dim=None):
    if low_dim is not None:
        data_matrix = low_dimensional_embedding(data_matrix, low_dim=low_dim)
    # compute embedding quality
    data_matrix_emb = quick_shift_tree_embedding(data_matrix, **opts)

    from sklearn.cluster import KMeans
    km = KMeans(init='k-means++', n_clusters=len(set(y)), n_init=50)
    yp = km.fit_predict(data_matrix_emb)

    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(y, yp)


def display_embedding(data_matrix, y, opts, low_dim=None):
    if low_dim is not None:
        data_matrix = low_dimensional_embedding(data_matrix, low_dim=low_dim)
    plot_embeddings(data_matrix, y, size=25, **opts)


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


def embed_dat_matrix_two_dimensions(low_dimension_data_matrix,
                                    y=None,
                                    labels=None,
                                    density_colormap='Blues',
                                    instance_colormap='YlOrRd'):
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


def quick_shift_tree_embedding(data_matrix,
                               knn=10,
                               knn_density=None,
                               k_threshold=0.9,
                               metric='linear',
                               **args):
    def parents(density_matrix=None, knn_density=None, kernel_matrix_sorted=None):
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
        return parent_dict

    def knns(kernel_matrix=None, kernel_matrix_sorted=None, knn=None, k_threshold=None):
        knn_dict = {}
        for i in range(n_instances):
            # id of k-th nearest neighbor
            jd = kernel_matrix_sorted[i, knn]
            knn_dict[i] = jd
        return knn_dict

    def knns_old(kernel_matrix=None, kernel_matrix_sorted=None, knn=None, k_threshold=None):
        knn_dict = {}
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
                knn_dict[i] = jd
        return knn_dict

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

    # compute edges
    parent_dict = parents(density_matrix, knn_density, kernel_matrix_sorted)
    knn_dict = knns(kernel_matrix, kernel_matrix_sorted, knn, k_threshold)

    # make a graph with instances as nodes
    import networkx as nx
    graph = nx.Graph()
    graph.add_nodes_from(range(n_instances))
    # add edge between instance and parent
    for i in range(n_instances):
        j = parent_dict[i]
        graph.add_edge(i, j, weight=1)
        if i in knn_dict:
            jd = knn_dict[i]
            graph.add_edge(i, jd, weight=1)

    # use graph layout algorithm to determine coordinates
    two_dimensional_data_matrix = nx.graphviz_layout(graph, prog='sfdp', args='-Goverlap=scale')
    two_dimensional_data_list = []
    for i in range(kernel_matrix.shape[0]):
        two_dimensional_data_list.append(list(two_dimensional_data_matrix[i]))
    embedding_data_matrix = np.array(two_dimensional_data_list)
    from sklearn.preprocessing import scale
    return scale(embedding_data_matrix)
