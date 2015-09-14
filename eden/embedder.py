import logging
from copy import deepcopy

import numpy as np

import networkx as nx
from sklearn.preprocessing import scale
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_kernels

from eden.selector import DensitySelector
from eden.iterated_semisupervised_feature_selection import IteratedSemiSupervisedFeatureSelection


logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------

class Projector(object):

    """Constructs features as the instance similarity to a set of instances as
    defined by the selector.

    Parameters
    ----------
    selector : Selector
        TODO.

    metric : string, or callable
        The metric to use when calculating kernel between instances in a
        feature array. If metric is a string, it must be one of the metrics
        in pairwise.PAIRWISE_KERNEL_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a kernel matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.

    `**kwds` : optional keyword parameters
        Any further parameters are passed directly to the kernel function.
    """

    def __init__(self, selector=DensitySelector(), metric='cosine', **kwds):
        self.selector = selector
        self.metric = metric
        self.kwds = kwds

    def fit(self, data_matrix, target=None):
        """Fit the estimator on the samples.

        Parameters
        ----------
        data_matrix : array-like, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        self
        """
        self.selected_instances = self.selector.fit_transform(data_matrix)
        return self

    def fit_transform(self, data_matrix, target=None):
        """Fit the estimator on the samples and transforms features as the instance
        similarity to a set of instances as defined by the selector.

        Parameters
        ----------
        data_matrix : array, shape = (n_samples, n_features)
          Samples.

        target : TODO

        Returns
        -------
        data_matrix : array, shape = (n_samples, n_features_new)
            Transformed array.
        """
        self.fit(data_matrix, target)
        return self.transform(data_matrix, target)

    def transform(self, data_matrix, target=None):
        """Transforms features as the instance similarity to a set of instances as
        defined by the selector.

        Parameters
        ----------
        data_matrix : array, shape = (n_samples, n_features)
          Samples.

        target : TODO

        Returns
        -------
        data_matrix : array, shape = (n_samples, n_features_new)
            Transformed array.
        """
        if self.selected_instances is None:
            raise Exception('transform must be used after fit')
        data_matrix_out = pairwise_kernels(data_matrix,
                                           Y=self.selected_instances,
                                           metric=self.metric,
                                           **self.kwds)
        return data_matrix_out


# -----------------------------------------------------------------------------

class Embedder2DSelector(object):

    """
    Transform a set of high dimensional vectors to a set of two dimensional vectors.

    Take in input list of selectors, then for each point find the closest selected instance and materialize
    an edge between the two. Finally output 2D coordinates of the corresponding graph embedding using the sfdp
    Graphviz algorithm.

    """

    def __init__(self, selectors=[DensitySelector()], n_nearest_neighbors=10, metric='cosine', **kwds):
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
        return self

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
            return self._graph_layout(graph), self.link_ids
        else:
            return self._graph_layout(graph)

    def links(self, data_matrix):
        data_size = data_matrix.shape[0]
        kernel_matrix = pairwise_kernels(data_matrix, metric=self.metric, **self.kwds)
        # compute instance density as average pairwise similarity
        density = np.sum(kernel_matrix, 0) / data_size
        # compute list of nearest neighbors
        kernel_matrix_sorted = np.argsort(-kernel_matrix)
        # make matrix of densities ordered by nearest neighbor
        density_matrix = density[kernel_matrix_sorted]
        # if a denser neighbor cannot be found then assign link to the instance itself
        link_ids = list(range(density_matrix.shape[0]))
        # for all instances determine link link
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
                    # if the density of the neighbor is higher than the density of the instance assign link
                    if j_density > i_density:
                        link_ids[i] = j
                        break
        return link_ids

    def _init_graph(self, data_matrix):
        graph = nx.Graph()
        graph.add_nodes_from(range(data_matrix.shape[0]))
        self.link_ids = self.links(data_matrix)
        for i, link in enumerate(self.link_ids):
            graph.add_edge(i, link)
        return graph

    def _update_graph(self, graph, data_matrix, selected_instances, selected_instances_ids):
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(selected_instances)
        knns = nn.kneighbors(data_matrix, return_distance=0)
        for i, knn in enumerate(knns):
            id = knn[0]
            original_id = selected_instances_ids[id]
            # add edge
            graph.add_edge(i, original_id)
        return graph

    def _graph_layout(self, graph):
        two_dimensional_data_matrix = nx.graphviz_layout(graph, prog='sfdp', args='-Goverlap=scale')
        two_dimensional_data_list = [list(two_dimensional_data_matrix[i]) for i in range(len(graph))]
        embedded_data_matrix = scale(np.array(two_dimensional_data_list))
        return embedded_data_matrix

# -----------------------------------------------------------------------------


class Embedder2DSelectorFeatureConstruction(object):

    """
    Transform a set of high dimensional vectors to a set of two dimensional vectors.

    The data matrix is transformed using the feature_constructor prior to the application of
    the embedder in 2D.

    """

    def __init__(self, embedder=Embedder2DSelector(), feature_constructor=Projector()):
        self.embedder = embedder
        self.feature_constructor = feature_constructor

    def fit(self, data_matrix):
        return self

    def fit_transform(self, data_matrix):
        self.fit(data_matrix)
        return self.transform(data_matrix)

    def transform(self, data_matrix):
        pass

    def randomize(self, data_matrix, amount=1):
        return self

    def optimize(self, data_matrix, target=None, score_func=None, n_iter=20):
        score, obj_dict = max(self._optimize(self, data_matrix, target, score_func, n_iter))
        self.__dict__.update(obj_dict)
        self.score = score

    def _optimize(self, data_matrix, target=None, score_func=None, n_iter=None):
        for i in range(n_iter):
            self.randomize(data_matrix)
            data_matrix_out = self.fit_transform(data_matrix, target)
            score = score_func(data_matrix, data_matrix_out)
            yield (score, deepcopy(self.__dict__))


# -----------------------------------------------------------------------------

class Embedder2DSelectorFeatureConstructionIteratedSemiSupervisedFeatureSelection(object):

    """
    Transform a set of high dimensional vectors to a set of two dimensional vectors.

    The data matrix is reduced using the feature selection prior to the application of
    the embedder in 2D.

    """

    def __init__(self, embedder=Embedder2DSelectorFeatureConstruction(),
                 feature_selection=IteratedSemiSupervisedFeatureSelection()):
        self.embedder = embedder
        self.feature_selection = feature_selection

    def fit(self, data_matrix):
        return self

    def fit_transform(self, data_matrix):
        self.fit(data_matrix)
        return self.transform(data_matrix)

    def transform(self, data_matrix):
        pass

    def randomize(self, data_matrix, amount=1):
        return self

    def optimize(self, data_matrix, target=None, score_func=None, n_iter=20):
        score, obj_dict = max(self._optimize(self, data_matrix, target, score_func, n_iter))
        self.__dict__.update(obj_dict)
        self.score = score

    def _optimize(self, data_matrix, target=None, score_func=None, n_iter=None):
        for i in range(n_iter):
            self.randomize(data_matrix)
            data_matrix_out = self.fit_transform(data_matrix, target)
            score = score_func(data_matrix, data_matrix_out)
            yield (score, deepcopy(self.__dict__))

# ------------------------------------------------------------------------------------------------------


def embedding_refinement(data_matrix_highdim,
                         data_matrix_lowdim,
                         n_neighbors=8,
                         emb_quality_th=1,
                         n_iter=20):
    # extract neighbors list for high dimensional case
    neigh_high = NearestNeighbors(n_neighbors=n_neighbors)
    neigh_high.fit(data_matrix_highdim)
    neighbors_list_highdim = neigh_high.kneighbors(data_matrix_highdim, return_distance=0)
    n_instances = data_matrix_lowdim.shape[0]
    logger.debug('refinements max num iters: %d  k in neqs: %d num insts: %d' %
                 (n_iter, n_neighbors, n_instances))
    for it in range(n_iter):
        average_embedding_quality_score, scores = knn_quality_score(data_matrix_lowdim,
                                                                    neighbors_list_highdim,
                                                                    n_neighbors)
        # select low quality embedded instances
        ids = [i for i, s in enumerate(scores)
               if relative_quality(i, scores, neighbors_list_highdim) <= emb_quality_th]
        # find average position of true knns and move point there
        new_data_matrix_lowdim = compute_average(ids, data_matrix_lowdim, neighbors_list_highdim)
        new_average_embedding_quality_score, new_scores = knn_quality_score(new_data_matrix_lowdim,
                                                                            neighbors_list_highdim,
                                                                            n_neighbors)
        if new_average_embedding_quality_score > average_embedding_quality_score:
            data_matrix_lowdim = new_data_matrix_lowdim
            n_refinements = len(ids)
            frac_refinements = float(n_refinements) / n_instances
            logger.debug('r %.2d neqs: %.3f \t %.2f (%d insts)' %
                         (it + 1, new_average_embedding_quality_score,
                          frac_refinements, n_refinements))
        else:
            break
    return data_matrix_lowdim


def relative_quality(id, scores, neighbors_list_highdim):
    """compute the ratio between the embedding quality score and the average\
    embedding quality score of the neighbors."""
    neighbors_highdim = neighbors_list_highdim[id]
    score = scores[id]
    avg_score = np.mean([scores[i] for i in neighbors_highdim])
    relative_quality_score = score / avg_score
    return relative_quality_score


def compute_average(ids, data_matrix_lowdim, neighbors_list_highdim):
    new_data_matrix_lowdim = data_matrix_lowdim.copy()
    for id in ids:
        # find average position of true knns and move point there
        neighbors_highdim = neighbors_list_highdim[id]
        neighbors = data_matrix_lowdim[neighbors_highdim]
        new_point = np.mean(neighbors, axis=0)
        # updated_point = (new_point - new_data_matrix_lowdim[id]) / 1 + new_data_matrix_lowdim[id]
        new_data_matrix_lowdim[id] = new_point
    return new_data_matrix_lowdim


def knn_quality_score(data_matrix, neighbors_list_highdim, n_neighbors):
    neigh_low = NearestNeighbors(n_neighbors=n_neighbors)
    neigh_low.fit(data_matrix)
    neighbors_list_lowdim = neigh_low.kneighbors(data_matrix, return_distance=0)
    average_embedding_quality_score,\
        scores = knn_quality_score_(neighbors_list_highdim,
                                    neighbors_list_lowdim,
                                    n_neighbors)
    return average_embedding_quality_score, scores


def knn_quality_score_(neighbors_list_highdim, neighbors_list_lowdim, n_neighbors):
    # for each row get intersection and return average intersection size over n_neighbors
    scores = []
    for row_high, row_low in zip(neighbors_list_highdim, neighbors_list_lowdim):
        intersection = np.intersect1d(row_high, row_low, assume_unique=False)
        scores.append(len(intersection) / float(n_neighbors))
    average_embedding_quality_score = np.mean(scores) - np.std(scores)
    return average_embedding_quality_score, scores


def embedding_quality_score(data_matrix_highdim,
                            data_matrix_lowdim,
                            n_neighbors=8,
                            return_scores=False):
    """Find k nearest neighbors in high and low dimensional case and return average
    neighborhood intersection size."""

    neigh_high = NearestNeighbors(n_neighbors=n_neighbors)
    neigh_high.fit(data_matrix_highdim)
    ns_high = neigh_high.kneighbors(data_matrix_highdim, return_distance=0)

    neigh_low = NearestNeighbors(n_neighbors=n_neighbors)
    neigh_low.fit(data_matrix_lowdim)
    ns_low = neigh_low.kneighbors(data_matrix_lowdim, return_distance=0)

    average_embedding_quality_score, scores = knn_quality_score_(ns_high, ns_low, n_neighbors)
    if return_scores:
        return average_embedding_quality_score, scores
    else:
        return average_embedding_quality_score


def averaged_embedding_quality_score(data_matrix_highdim,
                                     data_matrix_lowdim,
                                     n_neighbor_list=[10, 30],
                                     return_scores=False):
    average_score_list = []
    scores_list = []
    for n_neighbors in n_neighbor_list:
        average_score,\
            scores = embedding_quality_score(data_matrix_highdim,
                                             data_matrix_lowdim,
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
