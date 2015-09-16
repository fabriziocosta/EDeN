from collections import defaultdict
import random
import logging
import math
from copy import deepcopy
import numpy as np

import pymf
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import roc_auc_score
# from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier

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
        """Fit all selectors on the same samples.

        Parameters
        ----------
        data_matrix : array-like, shape = (n_samples, n_features)
            Samples.

        target : array-like, shape = (n_samples)

        Note
        ----
        The target array can be used to store the ids of the samples.
        The selection operation is applied to the samples and the congruent targets.

        Returns
        -------
        self
        """
        for selector in self.selectors:
            selector.fit(data_matrix, target)
        return self

    def fit_transform(self, data_matrix, target=None):
        """Fit all selectors on the same samples and then transform, i.e. select
        a subset of samples and return the reduced data matrix.

        Parameters
        ----------
        data_matrix : array-like, shape = (n_samples, n_features)
            Samples.

        target : array-like, shape = (n_samples)
            Vector containing additional information about the samples.

        Note
        ----
        The target array can be used to store the ids of the samples.
        The selection operation is applied to the samples and the congruent targets.

        Returns
        -------
        data_matrix : array-like, shape = (n_samples_new, n_features)
            Subset of samples.
        """
        self.fit(data_matrix, target)
        return self.transform(data_matrix, target)

    def transform(self, data_matrix, target=None):
        """Select a subset of samples and return the reduced data matrix.

        Parameters
        ----------
        data_matrix : array-like, shape = (n_samples, n_features)
            Samples.

        target : array-like, shape = (n_samples)
            Vector containing additional information about the samples.

        Note
        ----
        The target array can be used to store the ids of the samples.
        The selection operation is applied to the samples and the congruent targets.

        Returns
        -------
        data_matrix : array-like, shape = (n_samples_new, n_features)
            Subset of samples.
        """
        data_matrix_out_list = []
        for selector in self.selectors:
            data_matrix_out = selector.transform(data_matrix)
            data_matrix_out_list.append(data_matrix_out)
        data_matrix_out = np.vstack(data_matrix_out_list)

        if target is not None:
            self.selected_targets = []
            for selector in self.selectors:
                self.selected_targets.append(selector.selected_targets)

        return data_matrix_out

    def randomize(self, data_matrix, amount=1.0):
        """Set all the (hyper) parameters of the method to a random value.
        A configuration is created that is in the neighborhood of the current configuration,
        where the size of the neighborhood is parametrized by the variable 'amount'.

        Parameters
        ----------
        data_matrix : array-like, shape = (n_samples, n_features)
            Samples.

        amount : float (default 1.0)
            The size of the neighborhood of the parameters' configuration.
            A value of 0 means no change, while a value of 1.0 means that the new configuration
            can be anywhere in the parameter space.

        Returns
        -------
        self
        """
        for selector in self.selectors:
            selector.randomize(data_matrix)
        return self

    def optimize(self, data_matrix, target=None, score_func=None, n_iter=20):
        """Set the values of the (hyper) parameters so that the score_func achieves maximal value
        on data_matrix.

        Parameters
        ----------
        data_matrix : array-like, shape = (n_samples, n_features)
            Samples.

        target : array-like, shape = (n_samples)
            Vector containing additional information about the samples.

        score_func : callable
            Function to compute the score to maximize.
        """
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

    def randomize(self, data_matrix, amount=1.0):
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
        self.selected_targets = None
        return None

    def randomize(self, data_matrix, amount=1.0):
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
        self.selected_targets = None
        return None

    def randomize(self, data_matrix, amount=1.0):
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
        if target is not None:
            self.selected_targets = list(np.array(target)[selected_instances_ids])
        else:
            self.selected_targets = None
        return data_matrix[selected_instances_ids]

    def select(self, data_matrix, target=None):
        n_instances = data_matrix.shape[0]
        selected_instances_ids = list(range(n_instances))
        random.shuffle(selected_instances_ids)
        selected_instances_ids = sorted(selected_instances_ids[:self.n_instances])
        return selected_instances_ids

    def randomize(self, data_matrix, amount=1.0):
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
        if target is not None:
            self.selected_targets = list(np.array(target)[selected_instances_ids])
        else:
            self.selected_targets = None
        return data_matrix[selected_instances_ids]

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

    def randomize(self, data_matrix, amount=1.0):
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
        if target is not None:
            self.selected_targets = list(np.array(target)[selected_instances_ids])
        else:
            self.selected_targets = None
        return data_matrix[selected_instances_ids]

    def select(self, data_matrix, target=None):
        mf = pymf.SIVM(data_matrix.T, num_bases=self.n_instances)
        mf.factorize()
        basis = mf.W.T
        selected_instances_ids = self._get_ids(data_matrix, basis)
        return selected_instances_ids

    def randomize(self, data_matrix, amount=1.0):
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
        if target is not None:
            self.selected_targets = list(np.array(target)[selected_instances_ids])
        else:
            self.selected_targets = None
        return data_matrix[selected_instances_ids]

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

    def randomize(self, data_matrix, amount=1.0):
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
        if target is not None:
            self.selected_targets = list(np.array(target)[selected_instances_ids])
        else:
            self.selected_targets = None
        return data_matrix[selected_instances_ids]

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

    def randomize(self, data_matrix, amount=1.0):
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

    def __init__(self, n_instances='auto', randomized=False, metric='cosine', **kwds):
        self.n_instances = n_instances
        self.randomized = randomized
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
        if target is not None:
            self.selected_targets = list(np.array(target)[selected_instances_ids])
        else:
            self.selected_targets = None
        return data_matrix[selected_instances_ids]

    def select(self, data_matrix, target=None):
        # select most dense instances
        probabilities = self._probability_func(data_matrix, target)
        if self.randomized:
            # select instances according to their probability
            selected_instances_ids = [self._sample(probabilities) for i in range(self.n_instances)]
        else:
            # select the instances with highest probability
            selected_instances_ids = sorted([(prob, i) for i, prob in enumerate(probabilities)], reverse=True)
            selected_instances_ids = [i for prob, i in selected_instances_ids[:self.n_instances]]
        return selected_instances_ids

    def _probability_func(self, data_matrix, target=None):
        kernel_matrix = pairwise_kernels(data_matrix, metric=self.metric, **self.kwds)
        # compute instance density as average pairwise similarity
        densities = np.sum(kernel_matrix, 0)
        # normalize to obtain probabilities
        probabilities = densities / np.sum(densities)
        return probabilities

    def randomize(self, data_matrix, amount=1.0):
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


class KNNDecisionSurfaceSelector(AbstractSelector):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed sampling according to the inverse ratio score of class
    in the neighborhood of each sample.
    The ratio score of class is computed as the ratio between the abundance of instances
    with a given class in a k-neighborhood.
    """

    def __init__(self, n_instances='auto', randomized=False, n_neighbors=10, metric='cosine', **kwds):
        self.n_instances = n_instances
        self.randomized = randomized
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.kwds = kwds

    def fit(self, data_matrix, target=None):
        if self.n_instances == 'auto':
            self.n_instances = self._default_n_instances(data_matrix.shape[0])

    def fit_transform(self, data_matrix, target=None):
        self.fit(data_matrix, target)
        return self.transform(data_matrix, target)

    def transform(self, data_matrix, target=None):
        assert(target is not None), 'target cannot be None'
        selected_instances_ids = self.select(data_matrix, target)
        self.selected_targets = list(np.array(target)[selected_instances_ids])
        return data_matrix[selected_instances_ids]

    def select(self, data_matrix, target=None):
        # select maximally ambiguous or unpredictable instances
        probabilities = self._probability_func(data_matrix, target)
        if self.randomized:
            # select instances according to their probability
            selected_instances_ids = [self._sample(probabilities) for i in range(self.n_instances)]
        else:
            # select the instances with highest probability
            selected_instances_ids = sorted([(prob, i) for i, prob in enumerate(probabilities)], reverse=True)
            selected_instances_ids = [i for prob, i in selected_instances_ids[:self.n_instances]]
        return selected_instances_ids

    def _probability_func(self, data_matrix, target=None):
        _max_density_const = 100
        # compute the knn probabilities
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(data_matrix, target)
        probs_lists = neigh.predict_proba(data_matrix)
        # compute the max probability value for each instance
        inverse_densities = [max(probs_list) for probs_list in probs_lists]
        # compute inverse of probability as density score
        densities = [1 / d if d != 0 else _max_density_const for d in inverse_densities]
        # upper bound values to _max_density_const
        densities = [d if d < _max_density_const else _max_density_const for d in densities]
        # normalize to obtain probabilities
        probabilities = densities / np.sum(densities)
        return probabilities

    def randomize(self, data_matrix, amount=1.0):
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


class KernelDecisionSurfaceSelector(AbstractSelector):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed sampling according to the inverse of a scoring function.
    """

    def __init__(self, n_instances='auto',
                 score_func=roc_auc_score,
                 randomized=False,
                 metric='cosine', **kwds):
        self.n_instances = n_instances
        self.score_func = score_func
        self.randomized = randomized
        self.metric = metric
        self.kwds = kwds

    def fit(self, data_matrix, target=None):
        if self.n_instances == 'auto':
            self.n_instances = self._default_n_instances(data_matrix.shape[0])

    def fit_transform(self, data_matrix, target=None):
        self.fit(data_matrix, target)
        return self.transform(data_matrix, target)

    def transform(self, data_matrix, target=None):
        assert(target is not None), 'target cannot be None'
        selected_instances_ids = self.select(data_matrix, target)
        self.selected_targets = list(np.array(target)[selected_instances_ids])
        return data_matrix[selected_instances_ids]

    def select(self, data_matrix, target=None):
        # select maximally ambiguous or unpredictable instances
        probabilities = self._probability_func(data_matrix, target)
        if self.randomized:
            # select instances according to their probability
            selected_instances_ids = [self._sample(probabilities) for i in range(self.n_instances)]
        else:
            # select the instances with highest probability
            selected_instances_ids = sorted([(prob, i) for i, prob in enumerate(probabilities)], reverse=True)
            selected_instances_ids = [i for prob, i in selected_instances_ids[:self.n_instances]]
        return selected_instances_ids

    def _probability_func(self, data_matrix, target=None):
        kernel_matrix = pairwise_kernels(data_matrix, metric=self.metric, **self.kwds)
        # compute list of the ids sorted by similarity
        kernel_matrix_ids_sorted = np.argsort(-kernel_matrix)
        # compute the similarity of each instance in sorted order
        kernel_matrix_sorted = np.sort(-kernel_matrix)
        target_vals = list(set(target))
        scores = []
        for i in range(data_matrix.shape[0]):
            # get list of targets sorted by the instance similarity
            y_true = target[kernel_matrix_ids_sorted[i]]
            current_target = target[i]
            # if current target is the 0 class, i.e. the negative class, then invert the target
            if current_target == target_vals[0]:
                y_true = self._flip_targets(y_true, target_vals)
            y_scores = kernel_matrix_sorted[i]
            score = self.score_func(y_true, y_scores)
            scores.append(score)
        scores = np.array(scores)
        # normalize to obtain probabilities
        probabilities = scores / np.sum(scores)
        return probabilities

    def _flip_targets(self, target, target_vals):
        target_out = []
        for t in target:
            if t == target_vals[0]:
                target_out.append(target_vals[0])
            else:
                target_out.append(target_vals[1])
        return target_out

    def randomize(self, data_matrix, amount=1.0):
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


class DecisionSurfaceSelector(AbstractSelector):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed sampling according to the inverse of the ROC score.
    """

    def __init__(self, n_instances='auto',
                 estimator=SGDClassifier(average=True, class_weight='auto', shuffle=True, n_jobs=-1),
                 randomized=False):
        self.n_instances = n_instances
        self.estimator = estimator
        self.randomized = randomized

    def fit(self, data_matrix, target=None):
        if self.n_instances == 'auto':
            self.n_instances = self._default_n_instances(data_matrix.shape[0])

    def fit_transform(self, data_matrix, target=None):
        self.fit(data_matrix, target)
        return self.transform(data_matrix, target)

    def transform(self, data_matrix, target=None):
        assert(target is not None), 'target cannot be None'
        selected_instances_ids = self.select(data_matrix, target)
        self.selected_targets = list(np.array(target)[selected_instances_ids])
        return data_matrix[selected_instances_ids]

    def select(self, data_matrix, target=None):
        # select maximally ambiguous or unpredictable instances
        probabilities = self._probability_func(data_matrix, target)
        if self.randomized:
            # select instances according to their probability
            selected_instances_ids = [self._sample(probabilities) for i in range(self.n_instances)]
        else:
            # select the instances with highest probability
            selected_instances_ids = sorted([(prob, i) for i, prob in enumerate(probabilities)], reverse=True)
            selected_instances_ids = [i for prob, i in selected_instances_ids[:self.n_instances]]
        return selected_instances_ids

    def _logistic_func(self, x, l=1, k=1, x0=0):
        return float(l) / (1 + math.exp(- k * (x - x0)))

    def _probability_func(self, data_matrix, target=None):
        _max_const = 100
        self.estimator.fit(data_matrix, target)
        confidence_lists = self.estimator.decision_function(data_matrix)
        if target.ndim > 1:
            # compute the max confidence value for each instance
            max_confidences = [max(conf_list) for conf_list in confidence_lists]
        else:
            max_confidences = confidence_lists
        # compute inverse of the logistic of the max confidence as density score
        densities = [1 / self._logistic_func(x) if x != 0 else _max_const for x in max_confidences]
        # upper bound values to _max_const
        densities = [d if d < _max_const else _max_const for d in densities]
        # normalize to obtain probabilities
        probabilities = densities / np.sum(densities)
        return probabilities

    def randomize(self, data_matrix, amount=1.0):
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
