#!/usr/bin/env python

import logging
import random
from copy import deepcopy

import numpy as np
from scipy.sparse.linalg import eigs
import networkx as nx
from sklearn.preprocessing import scale
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sknn.mlp import Regressor, Layer

from eden.selector import CompositeSelector
from eden.selector import AllSelector
from eden.selector import SparseSelector
from eden.selector import MaxVolSelector
from eden.selector import QuickShiftSelector
from eden.selector import DensitySelector
from eden.selector import OnionSelector
from eden.selector import DecisionSurfaceSelector
from eden.iterated_semisupervised_feature_selection import IteratedSemiSupervisedFeatureSelection
from eden.auto_cluster import AutoCluster
from eden.util import serialize_dict

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------

class Projector(object):

    """Constructs features as the instance similarity to a set of instances as
    defined by the selector.

    Parameters
    ----------
    selector : Selector
        TODO.

    scale : bool (default True)
        If true then the data matrix returned is standardized to have 0 mean and unit variance

    scaling_factor : float (default 0.8)
        Multiplicative factor applied after normalization. This can be useful when data needs to be
        post-processed by neural networks and one wishes to push data in a linear region.

    random_state : int (deafault 1)
        The seed used for the pseudo-random generator.

    metric : string, or callable
        The metric to use when calculating kernel between instances in a
        feature array. If metric is a string, it must be one of the metrics
        in pairwise.PAIRWISE_KERNEL_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a kernel matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them.

    **kwds : optional keyword parameters
        Any further parameters are passed directly to the kernel function.
    """

    def __init__(self, selector=AllSelector(),
                 scale=True,
                 scaling_factor=0.8,
                 random_state=1,
                 metric='rbf', **kwds):
        self.selector = selector
        self.scale = scale
        self.scaling_factor = scaling_factor
        self.scaler = StandardScaler()
        self.metric = metric
        self.kwds = kwds
        self.random_state = random_state

    def __repr__(self):
        serial = []
        serial.append('Projector:')
        serial.append('metric: %s' % self.metric)
        if self.kwds is None or len(self.kwds) == 0:
            pass
        else:
            serial.append('params:')
            serial.append(serialize_dict(self.kwds))
        serial.append(str(self.selector))
        return '\n'.join(serial)

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
        self.selected_instances = self.selector.fit_transform(data_matrix, target=target)
        if self.scale:
            self.scale = False
            self.scaler.fit(self.transform(data_matrix))
            self.scale = True
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
        return self.transform(data_matrix)

    def transform(self, data_matrix):
        """Transforms features as the instance similarity to a set of instances as
        defined by the selector.

        Parameters
        ----------
        data_matrix : array, shape = (n_samples, n_features)
          Samples.

        Returns
        -------
        data_matrix : array, shape = (n_samples, n_features_new)
            Transformed array.
        """
        if self.selected_instances is None:
            raise Exception('Error: attempt to use transform on non fit model')
        if self.selected_instances.shape[0] == 0:
            raise Exception('Error: attempt to use transform using 0 selectors')
        data_matrix_out = pairwise_kernels(data_matrix,
                                           Y=self.selected_instances,
                                           metric=self.metric,
                                           **self.kwds)
        if self.scale:
            data_matrix_out = self.scaler.transform(data_matrix_out) * self.scaling_factor
        return data_matrix_out

    def randomize(self, data_matrix, amount=.5):
        random.seed(self.random_state)
        inclusion_threshold = random.uniform(amount, 1)
        selectors = []
        if random.random() > inclusion_threshold:
            selectors.append(QuickShiftSelector(random_state=random.randint(1, 1e9)))
        if random.random() > inclusion_threshold:
            selectors.append(DecisionSurfaceSelector(random_state=random.randint(1, 1e9)))
        if random.random() > inclusion_threshold:
            selectors.append(SparseSelector(random_state=random.randint(1, 1e9)))
        if random.random() > inclusion_threshold:
            selectors.append(MaxVolSelector(random_state=random.randint(1, 1e9)))
        if random.random() > inclusion_threshold:
            selectors.append(DensitySelector(random_state=random.randint(1, 1e9)))
        if random.random() > inclusion_threshold:
            selectors.append(OnionSelector(random_state=random.randint(1, 1e9)))
        if not selectors:
            selectors.append(QuickShiftSelector(random_state=random.randint(1, 1e9)))
        self.selector = CompositeSelector(selectors=selectors)
        self.selector.randomize(data_matrix, amount=amount)
        self.metric = 'rbf'
        self.kwds = {'gamma': random.choice([10 ** x for x in range(-3, 3)])}
        self.random_state = self.random_state ^ random.randint(1, 1e9)

# -----------------------------------------------------------------------------


class Embedder2D(object):

    """
    Transform a set of high dimensional vectors to a set of two dimensional vectors.

    Take in input list of selectors, then for each point find the closest selected instance and materialize
    an edge between the two. Finally output 2D coordinates of the corresponding graph embedding using the sfdp
    Graphviz algorithm.

    """

    def __init__(self,
                 compiled=False,
                 learning_rate=0.002,
                 n_layers=1,
                 n_features_hidden_factor=10,
                 selectors=[QuickShiftSelector()],
                 n_nearest_neighbors=10,
                 n_links=1,
                 layout='force',
                 layout_prog='sfdp',
                 layout_prog_args='-Goverlap=scale',
                 n_eigenvectors=10,
                 random_state=1,
                 metric='rbf', **kwds):
        self.compiled = compiled
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.n_features_hidden_factor = n_features_hidden_factor
        self.selectors = selectors
        self.n_nearest_neighbors = n_nearest_neighbors
        self.n_links = n_links
        self.layout = layout
        self.layout_prog = layout_prog
        self.layout_prog_args = layout_prog_args
        self.n_eigenvectors = n_eigenvectors
        self.metric = metric
        self.kwds = kwds
        self.random_state = random_state
        self.selected_instances_list = []
        self.selected_instances_ids_list = []

    def __repr__(self):
        serial = []
        serial.append('Embedder2D:')
        if self.compiled is True:
            serial.append('compiled: yes')
            serial.append('learning_rate: %.6f' % self.learning_rate)
            serial.append('n_features_hidden_factor: %d' % self.n_features_hidden_factor)
        else:
            serial.append('compiled: no')
        serial.append('layout: %s' % (self.layout))
        serial.append('layout_prog: %s' % (self.layout_prog))
        if self.layout_prog_args:
            serial.append('layout_prog_args: %s' % (self.layout_prog_args))
        serial.append('n_links: %s' % (self.n_links))
        if self.n_nearest_neighbors is None:
            serial.append('n_nearest_neighbors: None')
        else:
            serial.append('n_nearest_neighbors: %d' % self.n_nearest_neighbors)
        serial.append('metric: %s' % self.metric)
        if self.kwds is None or len(self.kwds) == 0:
            pass
        else:
            serial.append('params:')
            serial.append(serialize_dict(self.kwds))
        serial.append('selectors [%d]:' % len(self.selectors))
        for i, selector in enumerate(self.selectors):
            if len(self.selectors) > 1:
                serial.append('%d/%d  ' % (i + 1, len(self.selectors)))
            serial.append(str(selector))
        return '\n'.join(serial)

    def fit(self, data_matrix, target=None):
        if self.compiled is True:
            return self.fit_compiled(data_matrix, target=target)
        else:
            return self._fit(data_matrix, target=target)

    def transform(self, data_matrix):
        if self.compiled is True:
            return self.transform_compiled(data_matrix)
        else:
            return self._transform(data_matrix)

    def fit_transform(self, data_matrix, target=None):
        if self.compiled is True:
            return self.fit_transform_compiled(data_matrix, target=target)
        else:
            return self._fit_transform(data_matrix, target=target)

    def fit_compiled(self, data_matrix_in, target=None):
        data_matrix_out = self._fit_transform(data_matrix_in, target=target)
        n_features_in = data_matrix_in.shape[1]
        n_features_out = data_matrix_out.shape[1]
        n_features_hidden = int(n_features_in * self.n_features_hidden_factor)
        layers = []
        for i in range(self.n_layers):
            layers.append(Layer("Rectifier", units=n_features_hidden, name='hidden%d' % i))
        layers.append(Layer("Linear", units=n_features_out))
        self.net = Regressor(layers=layers,
                             learning_rate=self.learning_rate,
                             valid_size=0.1)
        self.net.fit(data_matrix_in, data_matrix_out)
        return self.net

    def transform_compiled(self, data_matrix):
        return self.net.predict(data_matrix)

    def fit_transform_compiled(self, data_matrix, target=None):
        self.fit_compiled(data_matrix, target=target)
        return self.transform_compiled(data_matrix)

    def _fit(self, data_matrix, target=None):
        # find selected instances
        self.selected_instances_list = []
        self.selected_instances_ids_list = []
        for i, selector in enumerate(self.selectors):
            selected_instances = selector.fit_transform(data_matrix, target=target)
            selected_instances_ids = selector.selected_instances_ids
            self.selected_instances_list.append(selected_instances)
            self.selected_instances_ids_list.append(selected_instances_ids)
        return self

    def _fit_transform(self, data_matrix, target=None):
        return self._fit(data_matrix, target=target)._transform(data_matrix)

    def _transform(self, data_matrix):
        # make a graph with instances as nodes
        graph = self._init_graph(data_matrix)
        if self.n_links > 0:
            # find the closest selected instance and instantiate knn edges
            for selected_instances, selected_instances_ids in \
                    zip(self.selected_instances_list, self.selected_instances_ids_list):
                if len(selected_instances) > 2:
                    graph = self._selection_knn_links(graph,
                                                      data_matrix,
                                                      selected_instances,
                                                      selected_instances_ids)
        self.graph = graph
        # use graph layout
        embedded_data_matrix = self._graph_layout(graph)
        # normalize display using 2D PCA
        embedded_data_matrix = PCA(n_components=2).fit_transform(embedded_data_matrix)
        return embedded_data_matrix

    def _kernel_shift_links(self, data_matrix):
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
        self.link_ids = self._kernel_shift_links(data_matrix)
        for i, link in enumerate(self.link_ids):
            graph.add_edge(i, link)
        return graph

    def _selection_knn_links(self, graph, data_matrix, selected_instances, selected_instances_ids):
        n_neighbors = min(self.n_links, len(selected_instances))
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(selected_instances)
        knns = nn.kneighbors(data_matrix, return_distance=0)
        for i, knn in enumerate(knns):
            # add edges to the knns
            for id in knn:
                original_id = selected_instances_ids[id]
                graph.add_edge(i, original_id)
        return graph

    def _graph_layout(self, graph):
        if self.layout == 'force':
            return self._layout_force(graph)
        elif self.layout == 'laplacian':
            return self._layout_laplacian(graph)
        else:
            raise Exception('Unknown layout type: %s' % self.layout)

    def _layout_force(self, graph):
        two_dimensional_data_matrix = nx.graphviz_layout(graph,
                                                         prog=self.layout_prog, args=self.layout_prog_args)
        two_dimensional_data_list = [list(two_dimensional_data_matrix[i]) for i in range(len(graph))]
        embedded_data_matrix = scale(np.array(two_dimensional_data_list))
        return embedded_data_matrix

    def _layout_laplacian(self, graph):
        nlm = nx.normalized_laplacian_matrix(graph)
        eigvals, eigvects = eigs(nlm, k=self.n_eigenvectors, which='SR')
        eigvals, eigvects = np.real(eigvals), np.real(eigvects)
        return scale(eigvects)

    def randomize(self, data_matrix, amount=.5):
        random.seed(self.random_state)
        inclusion_threshold = random.uniform(amount, 1)
        selectors = []
        if random.random() > inclusion_threshold:
            selectors.append(SparseSelector(random_state=random.randint(1, 1e9)))
        if random.random() > inclusion_threshold:
            selectors.append(MaxVolSelector(random_state=random.randint(1, 1e9)))
        if random.random() > inclusion_threshold:
            selectors.append(QuickShiftSelector(random_state=random.randint(1, 1e9)))
        if random.random() > inclusion_threshold:
            selectors.append(DensitySelector(random_state=random.randint(1, 1e9)))
        if random.random() > inclusion_threshold:
            selectors.append(OnionSelector(random_state=random.randint(1, 1e9)))
        if not selectors:
            selectors.append(DensitySelector(random_state=random.randint(1, 1e9)))
            selectors.append(SparseSelector(random_state=random.randint(1, 1e9)))
        selector = CompositeSelector(selectors=selectors)
        selector.randomize(data_matrix, amount=amount)
        self.selectors = deepcopy(selector.selectors)
        self.metric = 'rbf'
        self.kwds = {'gamma': random.choice([10 ** x for x in range(-3, 3)])}
        if random.random() > inclusion_threshold:
            self.n_nearest_neighbors = random.randint(3, 20)
        else:
            self.n_nearest_neighbors = None
        self.n_links = random.randint(1, 5)
        self.random_state = self.random_state ^ random.randint(1, 1e9)

# -----------------------------------------------------------------------------


class AutoEmbedder(object):

    """
    Transform a set of high dimensional vectors to a set of two dimensional vectors.

    The data matrix is transformed using the feature_constructor prior to the application of
    the embedder in 2D.

    """

    def __init__(self,
                 compiled=False,
                 learning_rate=0.002,
                 n_features_hidden_factor=10,
                 max_n_clusters=8,
                 n_iter=20,
                 inclusion_threshold=0.7,
                 random_state=1):
        self.max_n_clusters = max_n_clusters
        self.n_iter = n_iter
        self.inclusion_threshold = inclusion_threshold
        self.random_state = random_state
        self.feature_selector = IteratedSemiSupervisedFeatureSelection()
        self.feature_constructor = Projector()
        self.embedder = Embedder2D(compiled=compiled,
                                   learning_rate=learning_rate,
                                   n_features_hidden_factor=n_features_hidden_factor)
        self.evaluator = LocalEmbeddingEvaluator()
        self.refiner = LocalEmbeddingRefiner()
        self.auto_cluster = AutoCluster()

    def __repr__(self):
        serial = []
        serial.append('Embedder:')
        serial.append('inclusion_threshold: %.3f' % (self.inclusion_threshold))
        serial.append('n_iter: %d' % (self.n_iter))
        serial.append('max_n_clusters: %d' % (self.max_n_clusters))
        serial.append('-' * 80)
        serial.append(str(self.feature_constructor))
        serial.append('-' * 80)
        serial.append(str(self.embedder))
        serial.append('=' * 80)
        return '\n'.join(serial)

    def fit(self, data_matrix, target=None):
        raise NotImplementedError("Should have implemented this")
        return self

    def fit_transform(self, data_matrix, target=None):
        logger.info('Input data matrix: %d rows  %d cols' %
                    (data_matrix.shape[0], data_matrix.shape[1]))
        # if sparse.issparse(data_matrix):
        #     logger.info('Convert matrix format from sparse to dense using random projections')
        #     data_matrix = SparseRandomProjection().fit_transform(data_matrix).toarray()
        #     logger.info('Data matrix: %d rows  %d cols' %
        #                 (data_matrix.shape[0], data_matrix.shape[1]))
        if target is not None:
            data_matrix_feature_select = self.feature_selector.fit_transform(data_matrix, target)
            logger.info('Feature selection')
            logger.info('Data matrix: %d rows  %d cols' %
                        (data_matrix_feature_select.shape[0], data_matrix_feature_select.shape[1]))
        else:
            data_matrix_feature_select = data_matrix

        self.data_matrix = self.optimize(data_matrix_feature_select, target=target, n_iter=self.n_iter)
        logger.debug('%s' % str(self.__repr__()))
        return self.data_matrix

    def transform(self, data_matrix, target=None):
        data_matrix_feature_constr = self.feature_constructor.fit_transform(data_matrix, target=target)
        data_matrix_lowdim = self.embedder.fit_transform(data_matrix_feature_constr)
        data_matrix_out = self.refiner.embedding_refinement(data_matrix,
                                                            data_matrix_lowdim,
                                                            n_neighbors=8,
                                                            emb_quality_th=1,
                                                            n_iter=20)
        self.score, self.scores = self.evaluator.averaged_embedding_quality_score(data_matrix,
                                                                                  data_matrix_out,
                                                                                  n_neighbor_list=[10, 30],
                                                                                  return_scores=True)
        return data_matrix_out

    def predict(self, data_matrix, target=None):
        self.fit_transform(data_matrix, target=target)
        self.auto_cluster.optimize(self.data_matrix, max_n_clusters=self.max_n_clusters)
        self.predictions = self.auto_cluster.predictions
        logger.debug('embedding score: %.4f' % (self.score))
        return self.predictions

    def randomize(self, data_matrix, amount=1):
        self.feature_constructor.randomize(data_matrix, amount=amount)
        self.embedder.randomize(data_matrix, amount=amount)

    def optimize(self, data_matrix, target=None, n_iter=20):
        score, iter_id, data_matrix_out, obj_dict = max(self._optimize(data_matrix,
                                                                       target=target, n_iter=n_iter))
        self.__dict__.update(obj_dict)
        return data_matrix_out

    def _optimize(self, data_matrix, target=None, n_iter=None):
        for iter_id in range(1, n_iter + 1):
            try:
                self.randomize(data_matrix, amount=self.inclusion_threshold)
                data_matrix_out = self.transform(data_matrix, target=target)
                score = self.score
                yield (score, iter_id, data_matrix_out, deepcopy(self.__dict__))
            except Exception as e:
                logger.debug('Failed iteration. Reason: %s' % e)
                logger.debug('Exception', exc_info=True)
                logger.debug('Current object status:')
                logger.debug(self.__repr__())
                logger.debug('*' * 80)

# -----------------------------------------------------------------------------


class LocalEmbeddingEvaluator(object):

    def knn_quality_score(self, data_matrix, neighbors_list_highdim, n_neighbors):
        neigh_low = NearestNeighbors(n_neighbors=n_neighbors)
        neigh_low.fit(data_matrix)
        neighbors_list_lowdim = neigh_low.kneighbors(data_matrix, return_distance=0)
        average_embedding_quality_score,\
            scores = self.knn_quality_score_(neighbors_list_highdim,
                                             neighbors_list_lowdim,
                                             n_neighbors)
        return average_embedding_quality_score, scores

    def knn_quality_score_(self, neighbors_list_highdim, neighbors_list_lowdim, n_neighbors):
        # for each row get intersection and return average intersection size over n_neighbors
        scores = []
        for row_high, row_low in zip(neighbors_list_highdim, neighbors_list_lowdim):
            intersection = np.intersect1d(row_high, row_low, assume_unique=False)
            scores.append(len(intersection) / float(n_neighbors))
        average_embedding_quality_score = np.mean(scores) - np.std(scores)
        return average_embedding_quality_score, scores

    def embedding_quality_score(self,
                                data_matrix_highdim,
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

        average_embedding_quality_score, scores = self.knn_quality_score_(ns_high, ns_low, n_neighbors)
        if return_scores:
            return average_embedding_quality_score, scores
        else:
            return average_embedding_quality_score

    def averaged_embedding_quality_score(self,
                                         data_matrix_highdim,
                                         data_matrix_lowdim,
                                         n_neighbor_list=[10, 30],
                                         return_scores=False):
        average_score_list = []
        scores_list = []
        for n_neighbors in n_neighbor_list:
            average_score,\
                scores = self.embedding_quality_score(data_matrix_highdim,
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


# -----------------------------------------------------------------------------

class LocalEmbeddingRefiner(object):

    """Adjusts a 2d embedding and provides the score function for 2d embedding
    according to high dimensional representation.
    """

    def __init__(self):
        self.evaluator = LocalEmbeddingEvaluator()

    def embedding_refinement(self,
                             data_matrix_highdim,
                             data_matrix_lowdim,
                             n_neighbors=8,
                             emb_quality_th=1,
                             n_iter=20):
        # extract neighbors list for high dimensional case
        neigh_high = NearestNeighbors(n_neighbors=n_neighbors)
        neigh_high.fit(data_matrix_highdim)
        neighbors_list_highdim = neigh_high.kneighbors(data_matrix_highdim, return_distance=0)
        # n_instances = data_matrix_lowdim.shape[0]
        # logger.debug('refinements max num iters: %d  k in neqs: %d num insts: %d' %
        #             (n_iter, n_neighbors, n_instances))
        for it in range(n_iter):
            average_embedding_quality_score, scores = self.evaluator.knn_quality_score(data_matrix_lowdim,
                                                                                       neighbors_list_highdim,
                                                                                       n_neighbors)
            # select low quality embedded instances
            ids = [i for i, s in enumerate(scores)
                   if self.relative_quality(i, scores, neighbors_list_highdim) <= emb_quality_th]
            # find average position of true knns and move point there
            new_data_matrix_lowdim = self.compute_average(ids, data_matrix_lowdim, neighbors_list_highdim)
            new_average_embedding_quality_score, new_scores = self.evaluator.knn_quality_score(
                new_data_matrix_lowdim,
                neighbors_list_highdim,
                n_neighbors)
            if new_average_embedding_quality_score > average_embedding_quality_score:
                data_matrix_lowdim = new_data_matrix_lowdim
                # n_refinements = len(ids)
                # frac_refinements = float(n_refinements) / n_instances
                # logger.debug('r %.2d neqs: %.3f \t %.2f (%d insts)' %
                #             (it + 1, new_average_embedding_quality_score,
                #              frac_refinements, n_refinements))
            else:
                break
        return data_matrix_lowdim

    def relative_quality(self, id, scores, neighbors_list_highdim):
        """compute the ratio between the embedding quality score and the average\
        embedding quality score of the neighbors."""
        neighbors_highdim = neighbors_list_highdim[id]
        score = scores[id]
        avg_score = np.mean([scores[i] for i in neighbors_highdim])
        relative_quality_score = score / avg_score
        return relative_quality_score

    def compute_average(self, ids, data_matrix_lowdim, neighbors_list_highdim):
        new_data_matrix_lowdim = data_matrix_lowdim.copy()
        for id in ids:
            # find average position of true knns and move point there
            neighbors_highdim = neighbors_list_highdim[id]
            neighbors = data_matrix_lowdim[neighbors_highdim]
            new_point = np.mean(neighbors, axis=0)
            # updated_point = (new_point - new_data_matrix_lowdim[id]) / 1 + new_data_matrix_lowdim[id]
            new_data_matrix_lowdim[id] = new_point
        return new_data_matrix_lowdim


# -----------------------------------------------------------------------------
