#!/usr/bin/env python

from collections import defaultdict
import random
import logging
import math
import pylab as plt
import numpy as np
import scipy.sparse as sp

import networkx as nx
from sklearn import random_projection
from sklearn.preprocessing import scale
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
import pymf

from eden.util.display import plot_embeddings

logger = logging.getLogger(__name__)


class Embedder2DSelector(object):

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
            return self._graph_embedding(graph), self.link_ids
        else:
            return self._graph_embedding(graph)

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

    def _graph_embedding(self, graph):
        # two_dimensional_data_matrix = nx.graphviz_layout(graph, prog='sfdp', args='-Goverlap=scale')
        two_dimensional_data_matrix = nx.graphviz_layout(graph, prog='neato')
        two_dimensional_data_list = [list(two_dimensional_data_matrix[i]) for i in range(len(graph))]
        embedded_data_matrix = scale(np.array(two_dimensional_data_list))
        return embedded_data_matrix

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
    average_embedding_quality_score, scores = knn_quality_score_(neighbors_list_highdim,
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
        average_score, scores = embedding_quality_score(data_matrix_highdim,
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
                link_data_matrix = data_matrix[links]
                x_list = []
                for x_start, x_end in zip(data_matrix[:, 0], link_data_matrix[:, 0]):
                    x_list.append(x_start)
                    x_list.append(x_end)
                    x_list.append(None)
                y_list = []
                for y_start, y_end in zip(data_matrix[:, 1], link_data_matrix[:, 1]):
                    y_list.append(y_start)
                    y_list.append(y_end)
                    y_list.append(None)
                plt.plot(x_list, y_list, '-', color='cornflowerblue', alpha=0.3)
            plt.scatter(data_matrix[:, 0], data_matrix[:, 1],
                        c=alpha, cmap='Greys', s=95, edgecolors='none')
            plt.scatter(data_matrix[:, 0], data_matrix[:, 1],
                        alpha=0.55, c=y, cmap=cmap, s=30, edgecolors='none')
            plt.title('Embedding')

            plt.subplot(122)
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.scatter(data_matrix[:, 0], data_matrix[:, 1],
                        c=alpha, cmap='Greys', s=95, edgecolors='none', alpha=0.8)
            plt.title('Embedding score')
        else:
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.scatter(data_matrix[:, 0], data_matrix[:, 1], c=alpha, cmap='Greys', s=140, edgecolors='none')
            plt.scatter(data_matrix[:, 0], data_matrix[:, 1], c=y, cmap=cmap, s=30, edgecolors='none')
    plt.show()


def optimal_n_clusters(data_matrix, n_clusters_upper_bound=20, clustering_algo=None, algorithm_name=None):
    logger.debug('\t algorithm: %s \t max num clusters: %d' % (algorithm_name, n_clusters_upper_bound))
    max_score = 0
    for n_clusters in range(2, n_clusters_upper_bound):
        clustering_algo.set_params(n_clusters=n_clusters)
        preds = clustering_algo.fit_predict(data_matrix)
        score = silhouette_score(data_matrix, preds)
        if max_score < score:
            max_score = score
            max_preds = preds
            max_n_clusters = n_clusters
            logger.debug('\t num clusters: %.2d ss: %.4f' % (n_clusters, score))
    return max_score, max_preds, max_n_clusters


def recluster(data_matrix, max_n_clusters=20):
    logger.debug('ss=silhouette score')
    clustering_algos = []
    clustering_algos.append(('agglomerative (Ward)', AgglomerativeClustering(linkage='ward')))
    clustering_algos.append(('agglomerative (complete)', AgglomerativeClustering(linkage='complete')))
    clustering_algos.append(('agglomerative (average)', AgglomerativeClustering(linkage='average')))
    clustering_algos.append(('kmeans', MiniBatchKMeans()))
    max_score = 0
    for algorithm_name, clustering_algo in clustering_algos:
        score,\
            preds,\
            n_clusters = optimal_n_clusters(data_matrix,
                                            n_clusters_upper_bound=max_n_clusters,
                                            clustering_algo=clustering_algo,
                                            algorithm_name=algorithm_name)
        if score > max_score:
            max_score = score
            max_preds = preds
            max_n_clusters = n_clusters
            max_algo = algorithm_name
    logger.debug('selected: num clus: %d ss: %.4f algorithm: %s' % (max_n_clusters, max_score, max_algo))
    return max_preds, max_algo


def plot_reclustering(data_matrix, y=None, preds=None, size=10):
    cmap = 'rainbow'
    plt.figure(figsize=(2 * size, size))
    plt.subplot(121)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.scatter(data_matrix[:, 0], data_matrix[:, 1], c=y, cmap=cmap, alpha=.7, s=60, edgecolors='black')
    plt.title('True targets [%d]' % len(set(y)))
    plt.subplot(122)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.scatter(data_matrix[:, 0], data_matrix[:, 1], c=preds, cmap=cmap, alpha=.7, s=60, edgecolors='black')
    plt.title('Predicted targets [%d]' % len(set(preds)))
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


def make_opts_list(n_instances, n_iter):
    min_n_instances = int(n_instances * .1)
    max_n_instances = int(n_instances * .9)
    min_sqrt_n_instances = min(min_n_instances / 2, int(math.sqrt(min_n_instances / 2)))
    max_sqrt_n_instances = min(max_n_instances / 2, int(math.sqrt(max_n_instances / 2)))
    from numpy.random import randint
    opts_list = {'QuickShiftSelector': [None] * n_iter +
                 list(randint(min_n_instances, max_n_instances, size=n_iter)),
                 'MaxVolSelector': [None] * n_iter +
                 list(randint(min_sqrt_n_instances, max_sqrt_n_instances, size=n_iter)),
                 'DensitySelector': [None] * n_iter +
                 list(randint(min_n_instances, max_n_instances, size=n_iter)),
                 'SparseSelector': [None] * n_iter +
                 list(randint(min_n_instances, max_n_instances, size=n_iter)),
                 'EqualizingSelector': [None] * n_iter +
                 list(randint(min_n_instances, max_n_instances, size=n_iter))
                 }
    return opts_list


def make_selector_opts(opts_list):
    def sample(parameters):
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
    # select one element at random from each list
    opts = sample(opts_list)
    while is_all_none(opts):
        opts = sample(opts_list)
    return opts


def make_basis_opts(opts_list, n_instances):
    return make_selector_opts(opts_list)


def make_embed_opts(opts_list, n_instances):
    embed_opts = make_selector_opts(opts_list)
    embed_opts.update({'refine_n_nearest_neighbors': random.randint(3, 6)})
    embed_opts.update({'refine_emb_quality_th': random.uniform(0.75, 1)})
    sqrt_n_instances = int(math.sqrt(n_instances))
    if random.random() > 0.5:
        embed_opts.update({'n_nearest_neighbors': None})
    else:
        embed_opts.update({'n_nearest_neighbors': random.randint(5, sqrt_n_instances)})
    return embed_opts


def make_general_opts():
    general_opts = {}
    if random.random() > 0.5:
        general_opts['change_of_basis'] = True
    else:
        general_opts['change_of_basis'] = False
    return general_opts


def embed_(data_matrix, embed_opts=None, basis_opts=None, change_of_basis=False):
    # case for change of basis
    if change_of_basis:
        selectors = make_selectors(basis_opts)
        data_matrix = feature_construction(data_matrix, selectors)

    # embedding in 2D
    selectors = make_selectors(embed_opts)
    emb = Embedder2DSelector(selectors, n_nearest_neighbors=embed_opts['n_nearest_neighbors'])
    data_matrix_lowdim, link_ids = emb.fit_transform(data_matrix, return_links=True)

    # refine embedding
    data_matrix_lowdim = embedding_refinement(data_matrix,
                                              data_matrix_lowdim,
                                              n_neighbors=embed_opts['refine_n_nearest_neighbors'],
                                              emb_quality_th=embed_opts['refine_emb_quality_th'])

    # embedding quality score
    score, scores = averaged_embedding_quality_score(data_matrix, data_matrix_lowdim, return_scores=True)
    return data_matrix_lowdim, link_ids, score, scores


def optimize_embedding(data_matrix, known_targets=None,
                       min_feature_ratio=.1, n_iter=30, n_repetitions=1):
    # case for sparse data matrix: use random projection to transform to dense
    if sp.issparse(data_matrix):
        logger.info('Convert sparse to dense')
        logger.info('Data matrix: %d rows  %d cols' % (data_matrix.shape[0], data_matrix.shape[1]))
        from sklearn.random_projection import SparseRandomProjection
        data_matrix = SparseRandomProjection().fit_transform(data_matrix).toarray()
        logger.info('Data matrix: %d rows  %d cols' % (data_matrix.shape[0], data_matrix.shape[1]))

    if known_targets is not None:
        logger.info('Feature selection')
        logger.info('Data matrix: %d rows  %d cols' % (data_matrix.shape[0], data_matrix.shape[1]))
        new_data_matrix = iterated_semi_supervised_feature_selection(data_matrix, known_targets,
                                                                     min_feature_ratio=min_feature_ratio)
        if new_data_matrix.shape[1] > 2:
            data_matrix = new_data_matrix
        logger.info('Data matrix: %d rows  %d cols' % (data_matrix.shape[0], data_matrix.shape[1]))

    n_instances = data_matrix.shape[0]
    opts_list = make_opts_list(n_instances, n_iter)
    # iterate n_iter times to find best parameter configuration
    best_score = 0
    logger.debug('neqs = neighborhood embedding quality score')
    for i in range(n_iter):
        random.seed(i)
        # sample from the options
        embed_opts = make_embed_opts(opts_list, n_instances)
        basis_opts = make_basis_opts(opts_list, n_instances)
        general_opts = make_general_opts()
        try:
            # find options with max quality score
            score_list = []
            for it in range(n_repetitions):
                data_matrix_lowdim,\
                    link_ids,\
                    score,\
                    scores = embed_(data_matrix,
                                    embed_opts=embed_opts,
                                    basis_opts=basis_opts,
                                    change_of_basis=general_opts['change_of_basis'])
                score_list.append(score)
            mean_reduced_score = np.mean(score_list) - np.std(score_list)
            if best_score == 0 or mean_reduced_score > best_score:
                # best_embed_opts = embed_opts
                # best_basis_opts = basis_opts
                # best_change_of_basis = change_of_basis
                best_data_matrix_lowdim = data_matrix_lowdim
                best_link_ids = link_ids
                best_scores = scores
                best_score = mean_reduced_score
                mark = '*'
            else:
                mark = ''
            logger.debug('..%.2d/%d   neqs: %.3f (%.3f +- %.3f)  %s' %
                         (i + 1, n_iter, mean_reduced_score, np.mean(scores), np.std(scores), mark))
        except Exception as e:
            logger.debug('Failed iteration: %s' % e)
    return best_data_matrix_lowdim, best_link_ids, best_score, best_scores


def embed(data_matrix, true_targets=None, known_targets=None,
          max_n_clusters=8, score_threshold=0.8, max_iter=4,
          min_feature_ratio=.25, n_iter=30, n_repetitions=1):
    dataset_size = data_matrix.shape[0]
    if known_targets is not None:
        assert(dataset_size == len(known_targets))

    max_best_score = 0
    for it in range(max_iter):
        best_data_matrix_lowdim, \
            best_link_ids, \
            best_score, best_scores = optimize_embedding(data_matrix,
                                                         known_targets=known_targets,
                                                         min_feature_ratio=min_feature_ratio,
                                                         n_iter=n_iter,
                                                         n_repetitions=n_repetitions)
        if best_score > max_best_score:
            max_best_score = best_score
            max_best_data_matrix_lowdim = best_data_matrix_lowdim
            max_best_link_ids = best_link_ids
            max_best_scores = best_scores

            # perform clustering in two dimensional space
            preds, algo = recluster(best_data_matrix_lowdim, max_n_clusters=max_n_clusters)
            max_preds = preds
            max_algo = algo
            # take most convincing instances and their cluster id as next iteration known_targets
            scores_preds = [(score, pred, i) for i, (score, pred) in enumerate(zip(best_scores, preds))]
            scores_preds = sorted(scores_preds, reverse=True)
            known_targets = [-1] * dataset_size
            c = 0
            for score, pred, i in scores_preds:
                if score > score_threshold:
                    known_targets[i] = pred
                    c += 1
            known_targets = np.array(known_targets)
            n_clusters = len(set(known_targets))
            logger.debug('\t[%d/%d] Selected %.3f (%d/%d) convincing instances for %d targets' %
                         (it + 1, max_iter, float(c) / dataset_size, c, dataset_size, n_clusters - 1))
            # check that we have -1 plus at least two other classes
            if n_clusters < 3:
                logger.debug('..requirements are not met, bailing out.')
                break
        else:
            logger.debug('..did not improve, bailing out.')
            break

    # plot the embedding
    logger.info('Embedding quality [nearest neighbor fraction]: %.3f' % max_best_score)
    plot(max_best_data_matrix_lowdim, true_targets, alpha=max_best_scores, links=max_best_link_ids)

    rand_score = adjusted_rand_score(true_targets, max_preds)
    n_clusters_pred = len(set(max_preds))
    n_clusters = len(set(true_targets))
    logger.info('# true classes: %d    # predicted classes: %d   (algorithm: %s clustering)' %
                (n_clusters, n_clusters_pred, max_algo))
    logger.info('Adjusted Rand Score: %.2f ' % rand_score)
    plot_reclustering(max_best_data_matrix_lowdim, true_targets, max_preds)


# ------------------------------------------------------------------------------------------------------


def semisupervised_target(targets, unknown_fraction=None, known_fraction=None, random_state=1):
    if unknown_fraction is not None and known_fraction is not None:
        if unknown_fraction != 1 - known_fraction:
            raise Exception('unknown_fraction and known_fraction are inconsistent. bailing out')
    from sklearn.preprocessing import LabelEncoder
    targets = LabelEncoder().fit_transform(targets)

    if known_fraction is not None:
        unknown_fraction = 1 - known_fraction

    if unknown_fraction == 0:
        return targets
    elif unknown_fraction == 1:
        return None
    else:
        label_ids = [1] * int(len(targets) * unknown_fraction) + \
            [0] * int(len(targets) * (1 - unknown_fraction))
        random.seed(random_state)
        random.shuffle(label_ids)
        random_unlabeled_points = np.where(label_ids)
        labels = np.copy(targets)
        labels[random_unlabeled_points] = -1
        return labels


def semi_supervised_learning(data_matrix, target):
    if -1 in list(target):
        # if -1 is present in target do label spreading
        from sklearn.semi_supervised import LabelSpreading
        label_prop_model = LabelSpreading(kernel='knn', n_neighbors=5)
        label_prop_model.fit(data_matrix, target)
        pred_target = label_prop_model.predict(data_matrix)
        extended_target = []
        for pred_label, label in zip(pred_target, target):
            if label != -1 and pred_label != label:
                extended_target.append(label)
            else:
                extended_target.append(pred_label)
    else:
        extended_target = target
    return np.array(extended_target)


def feature_selection(data_matrix, target):
    from sklearn.feature_selection import RFECV
    from sklearn.linear_model import SGDClassifier
    estimator = SGDClassifier(average=True, shuffle=True, penalty='elasticnet')
    # perform feature rescaling with elastic penalty
    data_matrix = estimator.fit_transform(data_matrix, target)
    # perform recursive feature elimination
    selector = RFECV(estimator, step=0.1, cv=10)
    data_matrix = selector.fit_transform(data_matrix, target)
    return data_matrix


def iterated_semi_supervised_feature_selection(data_matrix, target, n_iter=30, min_feature_ratio=0.1):
    # iterate semisupervised label spreading and feature selection:
    # the idea is that due to feature selection the metric of the space and the label spreading changes
    n_features_orig = data_matrix.shape[1]
    for i in range(n_iter):
        n_features = data_matrix.shape[1]
        target = semi_supervised_learning(data_matrix, target)
        data_matrix = feature_selection(data_matrix, target)
        n_selected_features = data_matrix.shape[1]
        logger.debug('\t From %d features to %d features' % (n_features, n_selected_features))
        if n_selected_features == n_features or \
                n_selected_features < n_features_orig * min_feature_ratio or \
                n_selected_features < 3:
            break
    return data_matrix


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
    plot_embeddings(data_matrix, y, **opts)


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
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')


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
    # two_dimensional_data_matrix = nx.graphviz_layout(graph, prog='sfdp', args='-Goverlap=scale')
    two_dimensional_data_matrix = nx.graphviz_layout(graph, prog='neato')
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
