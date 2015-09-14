from collections import defaultdict
import random
import logging
import math
from copy import deepcopy
import numpy as np

import pymf
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import pairwise_distances

logger = logging.getLogger(__name__)

# TODO: make a onion selector, where one specifies the n_instances and the n_peels
# one removes progressively the outermost n_instances for n_peel times and then returns the n_instances
# that are outermost


# -----------------------------------------------------------------------------


class CompositeSelector(object):

    """
    Takes a list of selectors and returns the disjoint union of all selections.
    """

    def __init__(self, selectors):
        self.selectors = selectors

    def fit(self, data_matrix, target=None):
        for selector in self.selectors:
            selector.fit(data_matrix, target)
        return self

    def fit_transform(self, data_matrix, target=None):
        self.fit(data_matrix, target)
        return self.transform(data_matrix, target)

    def transform(self, data_matrix, target=None):
        if target:
            data_matrix_out_list = []
            target_out_list = []
            for selector in self.selectors:
                data_matrix_out, target_out = selector.transform(data_matrix, target)
                data_matrix_out_list.append(data_matrix_out)
                target_out_list.append(target_out)
            return np.vstack(data_matrix_out_list), np.hstack(target_out_list)
        else:
            data_matrix_out_list = []
            for selector in self.selectors:
                data_matrix_out = selector.transform(data_matrix)
                data_matrix_out_list.append(data_matrix_out)
            return np.vstack(data_matrix_out_list)

    def randomize(self, data_matrix, amount=1):
        for selector in self.selectors:
            selector.randomize(data_matrix)
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


class AbstractSelector(object):

    """Interface declaration for the Selector classes."""

    def _default_n_instances(self, data_size):
        return 2 * int(math.sqrt(data_size))

    def fit(self, data_matrix, target=None):
        raise NotImplementedError("Should have implemented this")

    def fit_transform(self, data_matrix, target=None):
        raise NotImplementedError("Should have implemented this")

    def transform(self, data_matrix, target=None):
        raise NotImplementedError("Should have implemented this")

    def randomize(self, data_matrix, amount=1):
        raise NotImplementedError("Should have implemented this")

# -----------------------------------------------------------------------------


class IdentitySelector(AbstractSelector):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed returning the same instances.
    """

    def __init__(self, n_instances='auto', random_state=None):
        # Note: n_instances is just a placeholder
        self.n_instances = n_instances
        self.random_state = None

    def fit(self, data_matrix, target=None):
        return self

    def fit_transform(self, data_matrix, target=None):
        return data_matrix

    def transform(self, data_matrix, target=None):
        if target is None:
            return data_matrix
        else:
            return data_matrix, target

    def randomize(self, data_matrix, amount=1):
        return self

# -----------------------------------------------------------------------------


class NullSelector(AbstractSelector):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection returns no instances.
    """

    def __init__(self, n_instances='auto', random_state=None):
        # Note: n_instances is just a placeholder
        self.n_instances = n_instances
        self.random_state = None

    def fit(self, data_matrix, target=None):
        return self

    def fit_transform(self, data_matrix, target=None):
        return None

    def transform(self, data_matrix, target=None):
        if target is None:
            return None
        else:
            return None, None

    def randomize(self, data_matrix, amount=1):
        return self


# -----------------------------------------------------------------------------


class RandomSelector(AbstractSelector):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed uniformly at random.
    """

    def __init__(self, n_instances='auto', random_state=1):
        self.n_instances = n_instances
        self.random_state = random_state
        random.seed(random_state)

    def fit(self, data_matrix, target=None):
        if self.n_instances == 'auto':
            self.n_instances = self._default_n_instances(data_matrix.shape[0])
        return self

    def fit_transform(self, data_matrix, target=None):
        self.fit(data_matrix, target)
        return self.transform(data_matrix, target)

    def transform(self, data_matrix, target=None):
        selected_instances_ids = self.select(data_matrix, target)
        if target is None:
            return data_matrix[selected_instances_ids]
        else:
            return data_matrix[selected_instances_ids], list(np.array(target)[selected_instances_ids])

    def select(self, data_matrix, target=None):
        n_instances = data_matrix.shape[0]
        selected_instances_ids = list(range(n_instances))
        random.shuffle(selected_instances_ids)
        selected_instances_ids = sorted(selected_instances_ids[:self.n_instances])
        return selected_instances_ids

    def randomize(self, data_matrix, amount=1):
        min_n_instances = int(data_matrix.shape[0] * 0.1)
        max_n_instances = int(data_matrix.shape[0] * 0.5)
        self.n_instances = random.randint(min_n_instances, max_n_instances)
        return self

# -----------------------------------------------------------------------------


class SparseSelector(AbstractSelector):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed choosing instances that maximizes instances pairwise difference.
    """

    def __init__(self, n_instances='auto', metric='euclidean', random_state=1):
        self.n_instances = n_instances
        self.metric = metric
        self.random_state = random_state
        random.seed(random_state)

    def fit(self, data_matrix, target=None):
        if self.n_instances == 'auto':
            self.n_instances = self._default_n_instances(data_matrix.shape[0])
        return self

    def fit_transform(self, data_matrix, target=None):
        self.fit(data_matrix, target)
        return self.transform(data_matrix, target)

    def transform(self, data_matrix, target=None):
        selected_instances_ids = self.select(data_matrix, target)
        if target is None:
            return data_matrix[selected_instances_ids]
        else:
            return data_matrix[selected_instances_ids], list(np.array(target)[selected_instances_ids])

    def select(self, data_matrix, target=None):
        # extract difference matrix
        diff_matrix = pairwise_distances(data_matrix, metric=self.metric)
        size = data_matrix.shape[0]
        m = np.max(diff_matrix) + 1
        # iterate size - k times, i.e. until only k instances are left
        for t in range(size - self.n_instances):
            # find pairs with smallest difference
            (min_i, min_j) = np.unravel_index(np.argmin(diff_matrix), diff_matrix.shape)
            # choose one instance at random
            if random.random() > 0.5:
                id = min_i
            else:
                id = min_j
            # remove instance with highest score by setting all its pairwise differences to max value
            diff_matrix[id, :] = m
            diff_matrix[:, id] = m
        # extract surviving elements, i.e. element that have 0 on the diagonal
        selected_instances_ids = np.array([i for i, x in enumerate(np.diag(diff_matrix)) if x == 0])
        return selected_instances_ids

    def randomize(self, data_matrix, amount=1):
        min_n_instances = int(data_matrix.shape[0] * 0.1)
        max_n_instances = int(data_matrix.shape[0] * 0.5)
        self.n_instances = random.randint(min_n_instances, max_n_instances)
        # TODO: randomize metric
        return self

# -----------------------------------------------------------------------------


class MaxVolSelector(AbstractSelector):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed choosing instances that maximizes the volume of the convex hull.
    """

    def __init__(self, n_instances='auto', random_state=1):
        self.n_instances = n_instances
        self.random_state = random_state
        random.seed(random_state)

    def fit(self, data_matrix, target=None):
        if self.n_instances == 'auto':
            self.n_instances = self._default_n_instances(data_matrix.shape[0])

    def fit_transform(self, data_matrix, target=None):
        self.fit(data_matrix, target)
        return self.transform(data_matrix, target)

    def transform(self, data_matrix, target=None):
        selected_instances_ids = self.select(data_matrix, target)
        if target is None:
            return data_matrix[selected_instances_ids]
        else:
            return data_matrix[selected_instances_ids], list(np.array(target)[selected_instances_ids])

    def select(self, data_matrix, target=None):
        mf = pymf.SIVM(data_matrix.T, num_bases=self.n_instances)
        mf.factorize()
        basis = mf.W.T
        selected_instances_ids = self._get_ids(data_matrix, basis)
        return selected_instances_ids

    def randomize(self, data_matrix, amount=1):
        min_n_instances = int(data_matrix.shape[0] * 0.1)
        max_n_instances = int(data_matrix.shape[0] * 0.5)
        min_sqrt_n_instances = min(min_n_instances / 2, int(math.sqrt(min_n_instances / 2)))
        max_sqrt_n_instances = min(max_n_instances / 2, int(math.sqrt(max_n_instances / 2)))
        self.n_instances = random.randint(min_sqrt_n_instances, max_sqrt_n_instances)
        return self

    def _get_ids(self, data_matrix, selected):
        diffs = pairwise_distances(data_matrix, selected)
        selected_instances_ids = [i for i, diff in enumerate(diffs) if 0 in diff]
        return selected_instances_ids

# -----------------------------------------------------------------------------


class EqualizingSelector(AbstractSelector):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed clustering via the supplied algorithm and then
    choosing instances uniformly at random from each cluster.
    """

    def __init__(self, n_instances='auto', clustering_algo=None, random_state=1):
        self.n_instances = n_instances
        self.clustering_algo = clustering_algo
        self.random_state = random_state
        random.seed(random_state)

    def fit(self, data_matrix, target=None):
        if self.n_instances == 'auto':
            self.n_instances = self._default_n_instances(data_matrix.shape[0])

    def fit_transform(self, data_matrix, target=None):
        self.fit(data_matrix, target)
        return self.transform(data_matrix, target)

    def transform(self, data_matrix, target=None):
        selected_instances_ids = self.select(data_matrix, target)
        if target is None:
            return data_matrix[selected_instances_ids]
        else:
            return data_matrix[selected_instances_ids], list(np.array(target)[selected_instances_ids])

    def select(self, data_matrix, target=None):
        # extract clusters
        class_ids = self.clustering_algo.fit_predict(data_matrix)
        # select same number per cluster uniformly at random with resampling if small size
        n_classes = len(set(class_ids))
        n_instances_per_cluster = int(self.n_instances / n_classes)
        instances_class_ids = [(c, i) for i, c in enumerate(class_ids)]
        random.shuffle(instances_class_ids)
        class_counter = defaultdict(int)
        selected_instances_ids = []
        for c, i in instances_class_ids:
            class_counter[c] += 1
            if class_counter[c] <= n_instances_per_cluster:
                selected_instances_ids.append(i)
        return selected_instances_ids

    def randomize(self, data_matrix, amount=1):
        min_n_instances = int(data_matrix.shape[0] * 0.1)
        max_n_instances = int(data_matrix.shape[0] * 0.5)
        self.n_instances = random.randint(min_n_instances, max_n_instances)
        # TODO: randomize clustering algorithm
        return self

# -----------------------------------------------------------------------------


class QuickShiftSelector(AbstractSelector):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed finding all parent instances as the nearest neighbor that
    has a higher density. Density is defined as the average kernel value for the instance.
    The n_instances with highest parent-instance norm are returned.
    """

    def __init__(self, n_instances='auto', metric='cosine', **kwds):
        self.n_instances = n_instances
        self.metric = metric
        self.kwds = kwds

    def fit(self, data_matrix, target=None):
        if self.n_instances == 'auto':
            self.n_instances = self._default_n_instances(data_matrix.shape[0])

    def fit_transform(self, data_matrix, target=None):
        self.fit(data_matrix, target)
        return self.transform(data_matrix, target)

    def transform(self, data_matrix, target=None):
        selected_instances_ids = self.select(data_matrix, target)
        if target is None:
            return data_matrix[selected_instances_ids]
        else:
            return data_matrix[selected_instances_ids], list(np.array(target)[selected_instances_ids])

    def select(self, data_matrix, target=None):
        # compute parent relationship
        parent_ids = self.parents(data_matrix, target=target)
        # compute norm of parent-instance vector
        # compute parent vectors
        parents = data_matrix[parent_ids]
        # compute difference
        diffs = np.diag(pairwise_distances(data_matrix, Y=parents))
        # sort from largest distance to smallest
        parent_distance_sorted_ids = list(np.argsort(-diffs))
        selected_instances_ids = []
        # add root (i.e. instance with distance 0 from parent)
        selected_instances_ids = [parent_distance_sorted_ids[-1]] + \
            parent_distance_sorted_ids[:self.n_instances - 1]
        return selected_instances_ids

    def parents(self, data_matrix, target=None):
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
                j = kernel_matrix_sorted[i, jj]
                if jj > 0:
                    j_density = d
                    # if the density of the neighbor is higher than the density of the instance assign parent
                    if j_density > i_density:
                        parent_ids[i] = j
                        break
        return parent_ids

    def randomize(self, data_matrix, amount=1):
        min_n_instances = int(data_matrix.shape[0] * 0.1)
        max_n_instances = int(data_matrix.shape[0] * 0.5)
        self.n_instances = random.randint(min_n_instances, max_n_instances)
        # TODO: randomize metric
        return self

# -----------------------------------------------------------------------------


class DensitySelector(AbstractSelector):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed sampling according to the instance density.
    The density is computed as the average kernel.
    """

    def __init__(self, n_instances='auto', metric='cosine', **kwds):
        self.n_instances = n_instances
        self.metric = metric
        self.kwds = kwds

    def fit(self, data_matrix, target=None):
        if self.n_instances == 'auto':
            self.n_instances = self._default_n_instances(data_matrix.shape[0])

    def fit_transform(self, data_matrix, target=None):
        self.fit(data_matrix, target)
        return self.transform(data_matrix, target)

    def transform(self, data_matrix, target=None):
        selected_instances_ids = self.select(data_matrix, target)
        if target is None:
            return data_matrix[selected_instances_ids]
        else:
            return data_matrix[selected_instances_ids], list(np.array(target)[selected_instances_ids])

    def select(self, data_matrix, target=None):
        kernel_matrix = pairwise_kernels(data_matrix, metric=self.metric, **self.kwds)
        # compute instance density as average pairwise similarity
        densities = np.sum(kernel_matrix, 0)
        # normalize to obtain probabilities
        probabilities = densities / np.sum(densities)
        # select instances according to their probability
        selected_instances_ids = [self._sample(probabilities) for i in range(self.n_instances)]
        return selected_instances_ids

    def randomize(self, data_matrix, amount=1):
        min_n_instances = int(data_matrix.shape[0] * 0.1)
        max_n_instances = int(data_matrix.shape[0] * 0.5)
        self.n_instances = random.randint(min_n_instances, max_n_instances)
        # TODO: randomize metric
        return self

    def _sample(self, probabilities):
        target_prob = random.random()
        prob_accumulator = 0
        for i, p in enumerate(probabilities):
            prob_accumulator += p
            if target_prob < prob_accumulator:
                return i
        # at last return the id of last element
        return len(probabilities) - 1

# -----------------------------------------------------------------------------
