from collections import defaultdict
import random
import logging
import pylab as plt
import numpy as np

import networkx as nx
from sklearn import random_projection
from sklearn.preprocessing import scale
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_kernels
import pymf

from eden.util.display import plot_embeddings

logger = logging.getLogger(__name__)


def embedding_quality_score(data_matrix_high_dim, data_matrix_low_dim, n_neighbors=8, return_scores=False):
    """Find k nearest neighbors in high and low dimensional case and return average
    neighborhood intersection size."""

    neigh_high = NearestNeighbors(n_neighbors=n_neighbors)
    neigh_high.fit(data_matrix_high_dim)
    ns_high = neigh_high.kneighbors(data_matrix_high_dim, return_distance=False)

    neigh_low = NearestNeighbors(n_neighbors=n_neighbors)
    neigh_low.fit(data_matrix_low_dim)
    ns_low = neigh_low.kneighbors(data_matrix_low_dim, return_distance=False)
    # for each row get intersection and return average intersection size over n_neighbors
    scores = []
    for row_high, row_low in zip(ns_high, ns_low):
        intersection = np.intersect1d(row_high, row_low, assume_unique=False)
        scores.append(len(intersection) / float(n_neighbors))
    average_score = np.mean(scores)
    if return_scores:
        return average_score, scores
    else:
        return average_score


def averaged_embedding_quality_score(data_matrix_high_dim,
                                     data_matrix_low_dim,
                                     n_neighbor_list=[2, 4, 8],
                                     return_scores=False):
    average_score_list = []
    scores_list = []
    for n_neighbors in n_neighbor_list:
        average_score, scores = embedding_quality_score(data_matrix_high_dim,
                                                        data_matrix_low_dim,
                                                        n_neighbors=n_neighbors,
                                                        return_scores=True)
        average_score_list.append(average_score)
        scores_list.append(scores)
    # compute average embedding_quality_score
    average_average_score = np.mean(average_score_list)
    # compute average_scores for each instance
    n_istances = len(scores_list[0])
    n_scores = len(scores_list)
    average_scores = []
    for i in range(n_istances):
        scores = [scores_list[j][i] for j in range(n_scores)]
        average_score = np.average(scores)
        average_scores.append(average_score)

    if return_scores:
        return average_average_score, average_scores
    else:
        return average_average_score


class SelectorEmbedder2D(object):

    """
    Transform a set of high dimensional vectors to a set of two dimensional vectors.

    Take in input list of selectors, then for each point find the closest selected instance and materialize
    an edge between the two. Finally output 2D coordinates of the corresponding graph embedding using the sfdp
    Graphviz algorithm.

    """

    def __init__(self, selectors, n_nearest_neighbors=None, metric='cosine', **kwds):
        self.selectors = selectors
        self.n_nearest_neighbors = n_nearest_neighbors
        self.metric = metric
        self.kwds = kwds
        self.selected_instances_list = []
        self.selected_instances_ids_list = []

    def fit(self, data_matrix):
        # find selected instances
        targets = list(range(data_matrix.shape[0]))
        for selector in self.selectors:
            selected_instances, selected_instances_ids = selector.fit_transform(data_matrix, targets=targets)
            self.selected_instances_list.append(selected_instances)
            self.selected_instances_ids_list.append(selected_instances_ids)

    def fit_transform(self, data_matrix, return_links=False):
        self.fit(data_matrix)
        return self.transform(data_matrix, return_links=return_links)

    def transform(self, data_matrix, return_links=False):
        # make a graph with instances as nodes
        graph = self._init_graph(data_matrix)
        # find the closest selected instance and instantiate an edge
        for selected_instances, selected_instances_ids in \
                zip(self.selected_instances_list, self.selected_instances_ids_list):
            graph = self._update_graph(graph, data_matrix, selected_instances, selected_instances_ids)
        if return_links:
            if self.n_nearest_neighbors is not None:
                return self._graph_embedding(graph), self.parent_ids
            else:
                return self._graph_embedding(graph), None
        else:
            return self._graph_embedding(graph)

    def _init_graph(self, data_matrix):
        graph = nx.Graph()
        graph.add_nodes_from(range(data_matrix.shape[0]))
        if self.n_nearest_neighbors is not None:
            self.parent_ids = self.parents(data_matrix)
            for i, parent in enumerate(self.parent_ids):
                graph.add_edge(i, parent, weight=1)
        return graph

    def _update_graph(self, graph, data_matrix, selected_instances, selected_instances_ids):
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(selected_instances)
        knns = nn.kneighbors(data_matrix, return_distance=False)
        for i, knn in enumerate(knns):
            id = knn[0]
            original_id = selected_instances_ids[id]
            # add edge
            graph.add_edge(i, original_id, weight=1)
        return graph

    def _graph_embedding(self, graph):
        two_dimensional_data_matrix = nx.graphviz_layout(graph, prog='sfdp', args='-Goverlap=scale')
        two_dimensional_data_list = [list(two_dimensional_data_matrix[i]) for i in range(len(graph))]
        embedded_data_matrix = scale(np.array(two_dimensional_data_list))
        return embedded_data_matrix

    def parents(self, data_matrix):
        data_size = data_matrix.shape[0]
        kernel_matrix = pairwise_kernels(data_matrix, metric=self.metric, **self.kwds)
        # compute instance density as average pairwise similarity
        density = np.sum(kernel_matrix, 0) / data_size
        # compute list of nearest neighbors
        kernel_matrix_sorted = np.argsort(-kernel_matrix)
        # make matrix of densities ordered by nearest neighbor
        density_matrix = density[kernel_matrix_sorted]
        # if a denser neighbor cannot be found then assign parent to the instance itself
        parent_ids = list(range(density_matrix.shape[0]))
        # for all instances determine parent link
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
                    # if the density of the neighbor is higher than the density of the instance assign parent
                    if j_density > i_density:
                        parent_ids[i] = j
                        break
        return parent_ids


# ------------------------------------------------------------------------------------------------------


def plot(data_matrix, y=None, alpha=None, links=None, size=10, reliability=True):
    cmap = 'rainbow'
    if alpha is None:
        plt.figure(figsize=(size, size))
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.scatter(data_matrix[:, 0], data_matrix[:, 1], c=y, cmap=cmap, alpha=.7, s=60, edgecolors='black')
    else:
        if reliability:
            plt.figure(figsize=(2 * size, size))
            plt.subplot(121)
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            if links is not None:
                # make xlist and ylist with Nones
                parent_data_matrix = data_matrix[links]
                x_list = []
                for x_start, x_end in zip(data_matrix[:, 0], parent_data_matrix[:, 0]):
                    x_list.append(x_start)
                    x_list.append(x_end)
                    x_list.append(None)
                y_list = []
                for y_start, y_end in zip(data_matrix[:, 1], parent_data_matrix[:, 1]):
                    y_list.append(y_start)
                    y_list.append(y_end)
                    y_list.append(None)
                plt.plot(x_list, y_list, '-', color='cornflowerblue', alpha=0.3)
            plt.scatter(data_matrix[:, 0], data_matrix[:, 1],
                        c=alpha, cmap='Greys', s=95, edgecolors='none')
            plt.scatter(data_matrix[:, 0], data_matrix[:, 1],
                        alpha=0.55, c=y, cmap=cmap, s=30, edgecolors='none')

            plt.subplot(122)
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.scatter(data_matrix[:, 0], data_matrix[:, 1], c=alpha, cmap='Greys', s=95, edgecolors='none')
        else:
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.scatter(data_matrix[:, 0], data_matrix[:, 1], c=alpha, cmap='Greys', s=140, edgecolors='none')
            plt.scatter(data_matrix[:, 0], data_matrix[:, 1], c=y, cmap=cmap, s=30, edgecolors='none')
    plt.show()


def feature_construction(data_matrix_original, selectors):
    from eden.selector import CompositeSelector
    selector = CompositeSelector(selectors)
    from eden.selector import Projector
    projector = Projector(selector, metric='cosine')
    data_matrix = projector.fit_transform(data_matrix_original)
    return data_matrix


def make_selectors(opts):
    # compose selector list
    selectors = []
    if 'QuickShiftSelector' in opts and opts['QuickShiftSelector'] is not None:
        from eden.selector import QuickShiftSelector
        selectors.append(QuickShiftSelector(n_instances=opts['QuickShiftSelector']))

    if 'MaxVolSelector' in opts and opts['MaxVolSelector'] is not None:
        from eden.selector import MaxVolSelector
        selectors.append(MaxVolSelector(n_instances=opts['MaxVolSelector']))

    if 'DensitySelector' in opts and opts['DensitySelector'] is not None:
        from eden.selector import DensitySelector
        selectors.append(DensitySelector(n_instances=opts['DensitySelector']))

    if 'SparseSelector' in opts and opts['SparseSelector'] is not None:
        from eden.selector import SparseSelector
        selectors.append(SparseSelector(n_instances=opts['SparseSelector']))

    if 'EqualizingSelector' in opts and opts['EqualizingSelector'] is not None:
        from sklearn.cluster import MiniBatchKMeans
        from eden.selector import EqualizingSelector
        n_clusters = opts['EqualizingSelector'] / 10
        selectors.append(EqualizingSelector(n_instances=opts['EqualizingSelector'],
                                            clustering_algo=MiniBatchKMeans(n_clusters)))

    return selectors


def embed_(data_matrix, y,
           embed_opts=None, basis_opts=None,
           sparse=False, change_of_basis=False, display=True):
    # case for sparse data matrix: use random projection to transform to dense
    if sparse:
        from sklearn.random_projection import SparseRandomProjection
        data_matrix = SparseRandomProjection().fit_transform(data_matrix).toarray()

    # case for cahnge of basis
    if change_of_basis:
        selectors = make_selectors(basis_opts)
        data_matrix = feature_construction(data_matrix, selectors)

    # embedding in 2D
    from eden.embedding import SelectorEmbedder2D
    selectors = make_selectors(embed_opts)
    emb = SelectorEmbedder2D(selectors, n_nearest_neighbors=embed_opts['n_nearest_neighbors'])
    data_matrix_low, parent_ids = emb.fit_transform(data_matrix, return_links=True)

    # embedding quality score
    score, scores = averaged_embedding_quality_score(data_matrix, data_matrix_low, return_scores=True)

    # output management: display or just output embedding quality score
    if display:
        logger.info('Embedding quality [nearest neighbor fraction]: %.2f' % score)
        plot(data_matrix_low, y, alpha=scores, links=parent_ids)
        return data_matrix_low
    else:
        return score


def embed_wo_bias(data_matrix, y, sparse=False, embed_opts=None, basis_opts=None):
    data_matrix_low = embed_(data_matrix, y,
                             embed_opts=embed_opts, basis_opts=basis_opts,
                             change_of_basis=False, sparse=sparse)
    data_matrix_low_basis = embed_(data_matrix, y,
                                   embed_opts=embed_opts, basis_opts=basis_opts,
                                   change_of_basis=True, sparse=sparse)
    return data_matrix_low, data_matrix_low_basis


def sample(parameters):
    import random
    parameters_sample = dict()
    for parameter in parameters:
        values = parameters[parameter]
        value = random.choice(values)
        parameters_sample[parameter] = value
    return parameters_sample


def is_all_none(opts):
    n_nones = sum(1 for key in opts if opts[key] is None)
    if n_nones == len(opts):
        return True
    else:
        return False


def make_opts_list(n_instances, n_iter):
    min_n_instances = int(n_instances * .1)
    max_n_instances = int(n_instances * .9)
    max_vol_n_instances = int(n_instances * .33)
    from numpy.random import randint
    opts_list = {'QuickShiftSelector': [None] * (n_iter / 2) +
                 list(randint(min_n_instances, max_n_instances, size=n_iter / 2)),
                 'MaxVolSelector': [None] * (n_iter / 2) +
                 list(randint(min_n_instances, max_vol_n_instances, size=n_iter / 2)),
                 'DensitySelector': [None] * (n_iter / 2) +
                 list(randint(min_n_instances, max_n_instances, size=n_iter / 2)),
                 'SparseSelector': [None] * (n_iter / 2) +
                 list(randint(min_n_instances, max_n_instances, size=n_iter / 2)),
                 'EqualizingSelector': [None] * (n_iter / 2) +
                 list(randint(min_n_instances, max_n_instances, size=n_iter / 2))
                 }
    return opts_list


def make_opts(opts_list):
    # select one element at random from each list
    opts = sample(opts_list)
    while is_all_none(opts):
        opts = sample(opts_list)
    return opts


def embed(data_matrix, y, sparse=False, n_iter=20, n_repetitions=2,
          return_low_dim_data_matrix=False, verbose=True):
    opts_list = make_opts_list(data_matrix.shape[0], n_iter)

    # iterate n_iter times to find best parameter configuration
    max_score = 0
    for i in range(n_iter):

        # sample from the options
        embed_opts = make_opts(opts_list)
        if random.random() > 0.75:
            embed_opts.update({'n_nearest_neighbors': None})
        else:
            embed_opts.update({'n_nearest_neighbors': random.randint(3, data_matrix.shape[0])})
        basis_opts = make_opts(opts_list)

        # find options with max quality score
        scores = []
        for it in range(n_repetitions):
            score = embed_(data_matrix, y,
                           embed_opts=embed_opts,
                           basis_opts=basis_opts,
                           change_of_basis=True,
                           sparse=sparse,
                           display=False)
            scores.append(score)
        score = np.mean(scores) - np.std(scores)
        if score > max_score:
            best_embed_opts = embed_opts
            best_basis_opts = basis_opts
            max_score = score
            mark = '*'
        else:
            mark = ''
        if verbose:
            logger.info('%.2d/%d  score: %.3f +- %.3f  %s' %
                        (i + 1, n_iter, np.mean(scores), np.std(scores), mark))

    # plot the embedding
    data_matrix_low, data_matrix_low_basis = embed_wo_bias(data_matrix, y,
                                                           sparse=sparse,
                                                           embed_opts=best_embed_opts,
                                                           basis_opts=best_basis_opts)
    if return_low_dim_data_matrix:
        return data_matrix_low, data_matrix_low_basis

# ------------------------------------------------------------------------------------------------------


class Embedder(object):

    """Transform a set of sparse high dimensional vectors to a set of low dimensional dense vectors.

    Under the hood sparse random projection and simplex volume maximization factorization is used.
    """

    def __init__(self, complexity=10, n_kmeans=None, random_state=1):
        self.complexity = complexity
        self.n_kmeans = n_kmeans
        self.transformer = None
        self.matrix_factorizer = None
        self.kmeans = None
        self.random_state = random_state

    def fit(self, data_matrix):
        n_rows, n_cols = data_matrix.shape
        if n_rows <= n_cols:
            n_components = n_rows
        elif n_cols < 5000:
            n_components = n_cols
        else:
            n_components = 'auto'
        self.transformer = random_projection.SparseRandomProjection(n_components=n_components,
                                                                    dense_output=True,
                                                                    random_state=self.random_state)
        data_matrix_new = self.transformer.fit_transform(data_matrix)
        self.matrix_factorizer = pymf.SIVM(data_matrix_new.T, num_bases=self.complexity)
        self.matrix_factorizer.factorize()
        if self.n_kmeans:
            self.kmeans = MiniBatchKMeans(n_clusters=self.n_kmeans)
            self.kmeans.fit(self.matrix_factorizer.H.T)

    def fit_transform(self, data_matrix):
        self.fit(data_matrix)
        if self.n_kmeans:
            return self.kmeans.transform(self.matrix_factorizer.H.T)
        else:
            return self.matrix_factorizer.H.T

    def transform(self, data_matrix):
        basis_data_matrix = self.matrix_factorizer.W
        data_matrix_new = self.transformer.transform(data_matrix)
        self.matrix_factorizer = pymf.SIVM(data_matrix_new.T, num_bases=self.complexity)
        self.matrix_factorizer.W = basis_data_matrix
        self.matrix_factorizer.factorize(compute_w=False)
        if self.n_kmeans:
            return self.kmeans.transform(self.matrix_factorizer.H.T)
        else:
            return self.matrix_factorizer.H.T

# -------------------------------------------------------------------------------------------------


class Embedder2D(object):

    """
    Transform a set of sparse high dimensional vectors to a set of two dimensional vectors.
    """

    def __init__(self,
                 knn=10,
                 knn_density=None,
                 k_threshold=0.7,
                 gamma=None,
                 low_dim=None,
                 post_process_pca=False,
                 random_state=1):
        self.knn = knn
        self.knn_density = knn_density
        self.k_threshold = k_threshold
        self.gamma = gamma
        self.low_dim = low_dim
        self.post_process_pca = post_process_pca
        self.random_state = random_state

    def fit(self, data_matrix, targets, n_iter=10):
        params = {'knn': random.randint(3, 20, size=n_iter),
                  'knn_density': random.randint(3, 20, size=n_iter),
                  'k_threshold': random.uniform(0.2, 0.99, size=n_iter),
                  'gamma': [None] * n_iter + [10 ** x for x in range(-4, -1)],
                  'low_dim': [None] * n_iter + list(random.randint(10, 50, size=n_iter))}
        results = []
        max_score = 0
        for i in range(n_iter):
            opts = self._sample(params)
            score = embedding_quality(data_matrix, targets, opts)
            results.append((score, opts))
            if max_score < score:
                max_score = score
                mark = '*'
                logger.info('%3d/%3d %s %+.4f %s' % (i + 1, n_iter, mark, score, opts))
            else:
                mark = ' '
                logger.debug('%3d/%3d %s %+.4f %s' % (i + 1, n_iter, mark, score, opts))
        best_opts = max(results)[1]

        self._rank_paramters(results)

        self.knn = best_opts['knn']
        self.knn_density = best_opts['knn_density']
        self.k_threshold = best_opts['k_threshold']
        self.gamma = best_opts['gamma']
        self.low_dim = best_opts['low_dim']

    def _rank_paramters(self, score_paramters):
        logger.info('Parameters rank (1-5):')
        sorted_score_parameters = sorted(score_paramters, reverse=True)
        rank_paramters = defaultdict(lambda: defaultdict(list))
        for i, (score, parameters) in enumerate(sorted_score_parameters):
            for key in parameters:
                rank_paramters[key][parameters[key]].append(i)
        for key_i in rank_paramters:
            results = []
            for key_j in rank_paramters[key_i]:
                results.append((np.mean(rank_paramters[key_i][key_j]), key_j))
            results = sorted(results)
            result_string = '%s:' % key_i
            for rank, value in results[:5]:
                result_string = '%s %s ' % (result_string, value)
            logger.debug(result_string)

    def get_parameters(self):
        parameters = {'knn': self.knn,
                      'knn_density': self.knn_density,
                      'k_threshold': self.k_threshold,
                      'gamma': self.gamma,
                      'low_dim': self.low_dim}
        return parameters

    def transform(self, data_matrix):
        return quick_shift_tree_embedding(data_matrix,
                                          knn=self.knn,
                                          knn_density=self.knn_density,
                                          k_threshold=self.k_threshold,
                                          gamma=self.gamma,
                                          post_process_pca=self.post_process_pca,
                                          low_dim=self.low_dim)

    def fit_transform(self, data_matrix, targets, n_iter=10):
        self.fit(data_matrix, targets, n_iter)
        return self.transform(data_matrix)

    def _sample(self, parameters):
        parameters_sample = dict()
        for parameter in parameters:
            values = parameters[parameter]
            value = random.choice(values)
            parameters_sample[parameter] = value
        return parameters_sample

# -------------------------------------------------------------------------------------------------


def matrix_factorization(data_matrix, n=10):
    mf = pymf.SIVM(data_matrix.T, num_bases=n)
    mf.factorize()
    return mf.W.T, mf.H.T


def reduce_dimensionality(data_matrix, n=10):
    W, H = matrix_factorization(data_matrix, n=n)
    return H


def low_dimensional_embedding(data_matrix, low_dim=None):
    n_rows, n_cols = data_matrix.shape
    # perform data dimension reduction only if #features > #data points
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


def display_embedding(data_matrix, y, opts):
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
                               gamma=None,
                               post_process_pca=False,
                               low_dim=None):
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

    if low_dim is not None:
        data_matrix = low_dimensional_embedding(data_matrix, low_dim=low_dim)

    if knn_density is None:
        knn_density = knn
    n_instances = data_matrix.shape[0]
    # extract pairwise similarity matrix with desired kernel
    from sklearn import metrics
    if gamma is None:
        kernel_matrix = metrics.pairwise.pairwise_kernels(data_matrix, metric='linear')
    else:
        kernel_matrix = metrics.pairwise.pairwise_kernels(data_matrix, metric='rbf', gamma=gamma)
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
    if post_process_pca is True:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        low_dimension_data_matrix = pca.fit_transform(embedding_data_matrix)

        from sklearn.preprocessing import scale
        return scale(low_dimension_data_matrix)
    else:
        from sklearn.preprocessing import scale
        return scale(embedding_data_matrix)
