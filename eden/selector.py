import numpy as np
import pymf
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict
import random
import logging
logger = logging.getLogger(__name__)

# TODO: make a onion selector, where one specifies the n_instances and the n_peels
# one removes progressively the outermost n_instances for n_peel times and then returns the n_instances
# that are outermost


# -------------------------------------------------------------------------------------------------

def get_ids(data_matrix, selected):
    diffs = pairwise_distances(data_matrix, selected)
    selected_instances_ids = [i for i, diff in enumerate(diffs) if 0 in diff]
    return selected_instances_ids

# -------------------------------------------------------------------------------------------------


class Projector(object):

    """
    Takes a selector and returns the distance from each selected instance as a transformation.
    """

    def __init__(self, selector, metric='cosine', **kwds):
        self.selector = selector
        self.metric = metric
        self.kwds = kwds

    def fit(self, data_matrix, targets=None):
        self.selected_instances = self.selector.transform(data_matrix)

    def fit_transform(self, data_matrix, targets=None):
        self.fit(data_matrix, targets)
        return self.transform(data_matrix, targets)

    def transform(self, data_matrix, targets=None):
        if self.selected_instances is None:
            raise Exception('transform must be used after fit')
        data_matrix_out = pairwise_kernels(data_matrix,
                                           Y=self.selected_instances,
                                           metric=self.metric,
                                           **self.kwds)
        return data_matrix_out

# -------------------------------------------------------------------------------------------------


class CompositeSelector(object):

    """
    Takes a list of selectors and returns the disjoint union of all selections.
    """

    def __init__(self, selectors):
        self.selectors = selectors

    def fit(self, data_matrix, targets=None):
        pass

    def fit_transform(self, data_matrix, targets=None):
        return self.transform(data_matrix, targets)

    def transform(self, data_matrix, targets=None):
        if targets:
            data_matrix_out_list = []
            targets_out_list = []
            for selector in self.selectors:
                data_matrix_out, targets_out = selector.transform(data_matrix, targets)
                data_matrix_out_list.append(data_matrix_out)
                targets_out_list.append(targets_out)
            return np.vstack(data_matrix_out_list), np.hstack(targets_out_list)
        else:
            data_matrix_out_list = []
            for selector in self.selectors:
                data_matrix_out = selector.transform(data_matrix)
                data_matrix_out_list.append(data_matrix_out)
            return np.vstack(data_matrix_out_list)

# -------------------------------------------------------------------------------------------------


class IdentitySelector(object):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed returning the same instances.
    """

    def __init__(self, n_instances=None, random_state=None):
        # Note: n_instances is just a placeholder
        self.n_instances = None
        self.random_state = None

    def fit(self, data_matrix, targets=None):
        pass

    def fit_transform(self, data_matrix, targets=None):
        return data_matrix

    def transform(self, data_matrix, targets=None):
        if targets is None:
            return data_matrix
        else:
            return data_matrix, targets

# -------------------------------------------------------------------------------------------------


class NullSelector(object):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection returns no instances.
    """

    def __init__(self, n_instances=None, random_state=None):
        # Note: n_instances is just a placeholder
        self.n_instances = None
        self.random_state = None

    def fit(self, data_matrix, targets=None):
        pass

    def fit_transform(self, data_matrix, targets=None):
        return None

    def transform(self, data_matrix, targets=None):
        if targets is None:
            return None
        else:
            return None, None

# -------------------------------------------------------------------------------------------------


class RandomSelector(object):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed uniformly at random.
    """

    def __init__(self, n_instances=10, random_state=1):
        self.n_instances = n_instances
        self.random_state = random_state
        random.seed(random_state)

    def fit(self, data_matrix, targets=None):
        pass

    def fit_transform(self, data_matrix, targets=None):
        return self.transform(data_matrix, targets)

    def transform(self, data_matrix, targets=None):
        selected_instances_ids = self.select(data_matrix, targets)
        if targets is None:
            return data_matrix[selected_instances_ids]
        else:
            return data_matrix[selected_instances_ids], targets[selected_instances_ids]

    def select(self, data_matrix, targets=None):
        n_instances = data_matrix.shape[0]
        selected_instances_ids = list(range(n_instances))
        random.shuffle(selected_instances_ids)
        selected_instances_ids = sorted(selected_instances_ids[:self.n_instances])
        return selected_instances_ids

# -------------------------------------------------------------------------------------------------


class SparseSelector(object):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed choosing instances that maximizes instances pairwise difference.
    """

    def __init__(self, n_instances=10, metric='euclidean', random_state=1):
        self.n_instances = n_instances
        self.metric = metric
        self.random_state = random_state
        random.seed(random_state)

    def fit(self, data_matrix, targets=None):
        pass

    def fit_transform(self, data_matrix, targets=None):
        return self.transform(data_matrix, targets)

    def transform(self, data_matrix, targets=None):
        selected_instances_ids = self.select(data_matrix, targets)
        if targets is None:
            return data_matrix[selected_instances_ids]
        else:
            return data_matrix[selected_instances_ids], targets[selected_instances_ids]

    def select(self, data_matrix, targets=None):
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

# -------------------------------------------------------------------------------------------------


class MaxVolSelector(object):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed choosing instances that maximizes the volume of the convex hull.
    """

    def __init__(self, n_instances=10, random_state=1):
        self.n_instances = n_instances
        self.random_state = random_state
        random.seed(random_state)

    def fit(self, data_matrix, targets=None):
        pass

    def fit_transform(self, data_matrix, targets=None):
        return self.transform(data_matrix, targets)

    def transform(self, data_matrix, targets=None):
        selected_instances_ids = self.select(data_matrix, targets)
        if targets is None:
            return data_matrix[selected_instances_ids]
        else:
            return data_matrix[selected_instances_ids], targets[selected_instances_ids]

    def select(self, data_matrix, targets=None):
        mf = pymf.SIVM(data_matrix.T, num_bases=self.n_instances)
        mf.factorize()
        basis = mf.W.T
        selected_instances_ids = get_ids(data_matrix, basis)
        return selected_instances_ids
# -------------------------------------------------------------------------------------------------


class EqualizingSelector(object):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed clustering via the supplied algorithm and then
    choosing instances uniformly at random from each cluster.
    """

    def __init__(self, n_instances=10, clustering_algo=None, random_state=1):
        self.n_instances = n_instances
        self.clustering_algo = clustering_algo
        self.random_state = random_state
        random.seed(random_state)

    def fit(self, data_matrix, targets=None):
        pass

    def fit_transform(self, data_matrix, targets=None):
        return self.transform(data_matrix, targets)

    def transform(self, data_matrix, targets=None):
        selected_instances_ids = self.select(data_matrix, targets)
        if targets is None:
            return data_matrix[selected_instances_ids]
        else:
            return data_matrix[selected_instances_ids], targets[selected_instances_ids]

    def select(self, data_matrix, targets=None):
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

# -------------------------------------------------------------------------------------------------


class QuickShiftSelector(object):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed finding all parent instances as the nearest neighbor that
    has a higher density. Density is defined as the average kernel value for the instance.
    The n_instances with highest parent-instance norm are returned.
    """

    def __init__(self, n_instances=10, metric='cosine', **kwds):
        self.n_instances = n_instances
        self.metric = metric
        self.kwds = kwds

    def fit(self, data_matrix, targets=None):
        pass

    def fit_transform(self, data_matrix, targets=None):
        self.fit(data_matrix, targets)
        return self.transform(data_matrix, targets)

    def transform(self, data_matrix, targets=None):
        selected_instances_ids = self.select(data_matrix, targets)
        if targets is None:
            return data_matrix[selected_instances_ids]
        else:
            return data_matrix[selected_instances_ids], targets[selected_instances_ids]

    def select(self, data_matrix, targets=None):
        n_instances = data_matrix.shape[0]
        kernel_matrix = pairwise_kernels(data_matrix, metric=self.metric, **self.kwds)
        # compute instance density as average pairwise similarity
        density = np.sum(kernel_matrix, 0) / n_instances
        # compute list of nearest neighbors
        kernel_matrix_sorted = np.argsort(-kernel_matrix)
        # make matrix of densities ordered by nearest neighbor
        density_matrix = density[kernel_matrix_sorted]
        # compute parent relationship
        parent_ids = self.parents(density_matrix, kernel_matrix_sorted)
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

    def parents(self, density_matrix=None, kernel_matrix_sorted=None):
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

# -------------------------------------------------------------------------------------------------
