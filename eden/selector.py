#!/usr/bin/env python

from collections import defaultdict
import random
import logging
import math
from copy import deepcopy
import numpy as np
from scipy import sparse

import pymf
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from scipy.stats import entropy
from sklearn.random_projection import SparseRandomProjection

from eden.util import serialize_dict

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------


class CompositeSelector(object):

    """
    Takes a list of selectors and returns the disjoint union of all selections.
    """

    def __init__(self, selectors):
        self.selectors = selectors

    def __repr__(self):
        serial = []
        serial.append('CompositeSelector')
        serial.append('selectors [%d]:' % len(self.selectors))
        for i, selector in enumerate(self.selectors):
            if len(self.selectors) > 1:
                serial.append('%d/%d  ' % (i + 1, len(self.selectors)))
            serial.append(str(selector))
        return '\n'.join(serial)

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
        return self.fit(data_matrix, target).transform(data_matrix, target)

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
        for selector in self.selectors:
            data_matrix_out = selector.transform(data_matrix, target=target)

        self._collect_selected_instances_ids()
        self._collect_target()
        data_matrix_out = self._collect_data_matrix(data_matrix)

        return data_matrix_out

    def _collect_data_matrix(self, data_matrix):
        if isinstance(data_matrix, sparse.csr_matrix):
            data_matrix_out = data_matrix[self.selected_instances_ids, :]
        else:
            data_matrix_out = data_matrix[self.selected_instances_ids]
        return data_matrix_out

    def _collect_selected_instances_ids(self):
        selected_instances_ids = [np.array(selector.selected_instances_ids)
                                  for selector in self.selectors
                                  if selector.selected_instances_ids is not None]
        self.selected_instances_ids = np.hstack(selected_instances_ids)

    def _collect_target(self):
        selected_targets = [np.array(selector.selected_targets).reshape(-1, 1)
                            for selector in self.selectors
                            if selector.selected_targets is not None]
        if selected_targets:
            self.selected_targets = np.vstack(selected_targets).reshape(-1, 1)
        else:
            self.selected_targets = None

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
            selector.randomize(data_matrix, amount=amount)
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
        score, index, obj_dict = max(self._optimize(self, data_matrix, target, score_func, n_iter))
        self.__dict__.update(obj_dict)
        self.score = score

    def _optimize(self, data_matrix, target=None, score_func=None, n_iter=None):
        for i in range(n_iter):
            self.randomize(data_matrix)
            data_matrix_out = self.fit_transform(data_matrix, target)
            score = score_func(data_matrix, data_matrix_out)
            yield (score, i, deepcopy(self.__dict__))

# -----------------------------------------------------------------------------


class AbstractSelector(object):

    """Interface declaration for the Selector classes."""

    def _auto_n_instances(self, data_size):
        # TODO [fabrizio]: reconstruct Gram matrix using approximation
        # find optimal num points for a reconstruction with small error
        # trade off with a cost C (hyperparameter) to pay per point so not to choose all points
        min_n = 3
        max_n = 2 * int(math.sqrt(data_size))
        n_instances = random.randint(min_n, max_n)
        return n_instances

    def __repr__(self):
        serial = []
        serial.append(self.name)
        serial.append('n_instances: %d' % (self.n_instances))
        serial.append('random_state: %d' % (self.random_state))
        return '\n'.join(serial)

    def fit(self, data_matrix, target=None):
        return self

    def fit_transform(self, data_matrix, target=None):
        return self.fit(data_matrix, target).transform(data_matrix, target)

    def transform(self, data_matrix, target=None):
        self.selected_instances_ids = self.select(data_matrix, target)
        if target is not None:
            self.selected_targets = np.array(target)[self.selected_instances_ids]
        else:
            self.selected_targets = None
        return data_matrix[self.selected_instances_ids]

    def select(self, data_matrix, target=None):
        raise NotImplementedError("Should have implemented this")

    def randomize(self, data_matrix, amount=1.0):
        raise NotImplementedError("Should have implemented this")

# -----------------------------------------------------------------------------


class AllSelector(AbstractSelector):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed choosing all instances as landmarks.
    """

    def __init__(self, n_instances=None, random_state=1):
        self.name = 'AllSelector'
        self.n_instances = n_instances
        self.random_state = random_state

    def __repr__(self):
        serial = []
        serial.append(self.name)
        if self.n_instances:
            serial.append('n_instances: %d' % (self.n_instances))
        else:
            serial.append('n_instances: all')
        serial.append('random_state: %d' % (self.random_state))
        return '\n'.join(serial)

    def select(self, data_matrix, target=None):
        self.n_instances = data_matrix.shape[0]
        selected_instances_ids = list(range(self.n_instances))
        return selected_instances_ids

    # -----------------------------------------------------------------------------


class SparseSelector(AbstractSelector):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed choosing instances that maximizes instances pairwise difference.
    """

    def __init__(self, n_instances=20, random_state=1, metric='rbf', **kwds):
        self.name = 'SparseSelector'
        self.n_instances = n_instances
        self.metric = metric
        self.kwds = kwds
        self.random_state = random_state

    def __repr__(self):
        serial = []
        serial.append(self.name)
        serial.append('n_instances: %d' % (self.n_instances))
        serial.append('metric: %s' % (self.metric))
        if self.kwds is None or len(self.kwds) == 0:
            pass
        else:
            serial.append('params:')
            serial.append(serialize_dict(self.kwds))
        serial.append('random_state: %d' % (self.random_state))
        return '\n'.join(serial)

    def select(self, data_matrix, target=None):
        # extract difference matrix
        kernel_matrix = pairwise_kernels(data_matrix, metric=self.metric, **self.kwds)
        # set minimum value
        m = - 1
        # set diagonal to 0 to remove self similarity
        np.fill_diagonal(kernel_matrix, 0)
        # iterate size - k times, i.e. until only k instances are left
        for t in range(data_matrix.shape[0] - self.n_instances):
            # find pairs with largest kernel
            (max_i, max_j) = np.unravel_index(np.argmax(kernel_matrix), kernel_matrix.shape)
            # choose one instance at random
            if random.random() > 0.5:
                id = max_i
            else:
                id = max_j
            # remove instance with highest score by setting all its pairwise similarity to min value
            kernel_matrix[id, :] = m
            kernel_matrix[:, id] = m
        # extract surviving elements, i.e. element that have 0 on the diagonal
        selected_instances_ids = np.array(
            [i for i, x in enumerate(np.diag(kernel_matrix)) if x == 0])
        return selected_instances_ids

    def randomize(self, data_matrix, amount=1.0):
        random.seed(self.random_state)
        self.n_instances = self._auto_n_instances(data_matrix.shape[0])
        self.metric = 'rbf'
        self.kwds = {'gamma': random.choice([10 ** x for x in range(-3, 3)])}
        self.random_state = self.random_state ^ random.randint(1, 1e9)

# -----------------------------------------------------------------------------


class MaxVolSelector(AbstractSelector):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed choosing instances that maximizes the volume of the convex hull.
    """

    def __init__(self, n_instances=20, random_state=1):
        self.name = 'MaxVolSelector'
        self.n_instances = n_instances
        self.random_state = random_state
        random.seed(random_state)

    def __repr__(self):
        serial = []
        serial.append(self.name)
        serial.append('n_instances: %d' % (self.n_instances))
        serial.append('random_state: %d' % (self.random_state))
        return '\n'.join(serial)

    def select(self, data_matrix, target=None):
        if sparse.issparse(data_matrix):
            data_matrix = SparseRandomProjection().fit_transform(data_matrix).toarray()
        mf = pymf.SIVM(data_matrix.T, num_bases=self.n_instances)
        mf.factorize()
        basis = mf.W.T
        selected_instances_ids = self._get_ids(data_matrix, basis)
        return selected_instances_ids

    def _get_ids(self, data_matrix, selected):
        diffs = pairwise_distances(data_matrix, selected)
        selected_instances_ids = [i for i, diff in enumerate(diffs) if 0 in diff]
        return selected_instances_ids

    def randomize(self, data_matrix, amount=1.0):
        random.seed(self.random_state)
        self.n_instances = self._auto_n_instances(data_matrix.shape[0])
        self.random_state = self.random_state ^ random.randint(1, 1e9)

# -----------------------------------------------------------------------------


class OnionSelector(AbstractSelector):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed choosing instances that maximizes the volume of the convex hull,
    removing them from the set, iterating the procedure n_layers times and then returning
    the instances forming the convex hull of the remaining dataset.
    """

    def __init__(self, n_instances=20, n_layers=1, random_state=1):
        self.name = 'OnionSelector'
        self.n_instances = n_instances
        self.n_layers = n_layers
        self.random_state = random_state
        random.seed(random_state)

    def __repr__(self):
        serial = []
        serial.append(self.name)
        serial.append('n_instances: %d' % (self.n_instances))
        serial.append('n_layers: %d' % (self.n_layers))
        serial.append('random_state: %d' % (self.random_state))
        return '\n'.join(serial)

    def transform(self, data_matrix, target=None):
        if sparse.issparse(data_matrix):
            data_matrix = SparseRandomProjection().fit_transform(data_matrix).toarray()
        current_data_matrix = data_matrix
        current_target = target
        self.selected_instances_ids = np.array(range(data_matrix.shape[0]))
        self.selected_targets = None
        for i in range(self.n_layers):
            selected_instances_ids = self.select_layer(current_data_matrix)
            # remove selected instances from data matrix
            ids = set(range(current_data_matrix.shape[0]))
            remaining_ids = list(ids.difference(selected_instances_ids))
            if len(remaining_ids) == 0:
                break
            remaining_ids = np.array(remaining_ids)
            current_data_matrix = current_data_matrix[remaining_ids]
            self.selected_instances_ids = self.selected_instances_ids[remaining_ids]
            if current_target is not None:
                current_target = current_target[remaining_ids]
        selected_instances_ids = self.select_layer(current_data_matrix)
        if current_target is not None:
            self.selected_targets = current_target[selected_instances_ids]
        self.selected_instances_ids = self.selected_instances_ids[selected_instances_ids]
        return current_data_matrix[selected_instances_ids]

    def select_layer(self, data_matrix):
        mf = pymf.SIVM(data_matrix.T, num_bases=self.n_instances)
        mf.factorize()
        basis = mf.W.T
        selected_instances_ids = self._get_ids(data_matrix, basis)
        return selected_instances_ids

    def _get_ids(self, data_matrix, selected):
        diffs = pairwise_distances(data_matrix, selected)
        selected_instances_ids = [i for i, diff in enumerate(diffs) if 0 in diff]
        return selected_instances_ids

    def randomize(self, data_matrix, amount=1.0):
        random.seed(self.random_state)
        self.n_instances = self._auto_n_instances(data_matrix.shape[0])
        max_n_layers = min(5, data_matrix.shape[0] / (self.n_instances + 1))
        self.n_layers = random.randint(1, max_n_layers)
        self.random_state = self.random_state ^ random.randint(1, 1e9)

# -----------------------------------------------------------------------------


class EqualizingSelector(AbstractSelector):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed clustering via the supplied algorithm and then
    choosing instances uniformly at random from each cluster.
    """

    def __init__(self, n_instances=20, clustering_algo=None, random_state=1, **kwds):
        self.name = 'EqualizingSelector'
        self.n_instances = n_instances
        self.clustering_algo = clustering_algo(**kwds)
        self.kwds = kwds
        self.random_state = random_state
        random.seed(random_state)

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
        random.seed(self.random_state)
        self.n_instances = self._auto_n_instances(data_matrix.shape[0])
        self.random_state = self.random_state ^ random.randint(1, 1e9)
        # TODO: randomize clustering algorithm

# -----------------------------------------------------------------------------


class QuickShiftSelector(AbstractSelector):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed finding all parent instances as the nearest neighbor that
    has a higher density. Density is defined as the average kernel value for the instance.
    The n_instances with highest parent-instance norm are returned.
    """

    def __init__(self, n_instances=20, random_state=1, metric='rbf', **kwds):
        self.name = 'QuickShiftSelector'
        self.n_instances = n_instances
        self.random_state = random_state
        self.metric = metric
        self.kwds = kwds

    def __repr__(self):
        serial = []
        serial.append(self.name)
        serial.append('n_instances: %d' % (self.n_instances))
        serial.append('metric: %s' % (self.metric))
        if self.kwds is None or len(self.kwds) == 0:
            pass
        else:
            serial.append('params:')
            serial.append(serialize_dict(self.kwds))
        serial.append('random_state: %d' % (self.random_state))
        return '\n'.join(serial)

    def select(self, data_matrix, target=None):
        # compute parent relationship
        self.parent_ids = self.parents(data_matrix, target=target)
        # compute norm of parent-instance vector
        # compute parent vectors
        parents = data_matrix[self.parent_ids]
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
        # compute instance density as 1 over average pairwise distance
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
        random.seed(self.random_state)
        self.n_instances = self._auto_n_instances(data_matrix.shape[0])
        self.metric = 'rbf'
        self.kwds = {'gamma': random.choice([10 ** x for x in range(-3, 3)])}
        self.random_state = self.random_state ^ random.randint(1, 1e9)

# -----------------------------------------------------------------------------


class DensitySelector(AbstractSelector):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed sorting according to the instance density and
    taking instances uniformly distributed in the sorted list (i.e. from
    the instance with maximal density to the instance with the lowest density).
    The density is computed as the average kernel.
    """

    def __init__(self, n_instances=20, percentile=0.75, random_state=1, metric='rbf', **kwds):
        self.name = 'DensitySelector'
        self.n_instances = n_instances
        self.percentile = percentile
        self.random_state = random_state
        self.metric = metric
        self.kwds = kwds
        self.selected_targets = None

    def __repr__(self):
        serial = []
        serial.append(self.name)
        serial.append('n_instances: %d' % (self.n_instances))
        serial.append('percentile: %.2f' % (self.percentile))
        serial.append('metric: %s' % (self.metric))
        if self.kwds is None or len(self.kwds) == 0:
            pass
        else:
            serial.append('params:')
            serial.append(serialize_dict(self.kwds))
        serial.append('random_state: %d' % (self.random_state))
        return '\n'.join(serial)

    def select(self, data_matrix, target=None):
        # select most dense instances
        densities = self._density_func(data_matrix, target)
        # select the instances for equally spaced intervals in the list sorted by density
        sorted_instances_ids = sorted([(prob, i) for i, prob in enumerate(densities)], reverse=True)
        n_instances_tot = int(data_matrix.shape[0] * (1 - self.percentile))
        step = max(1, int(n_instances_tot / self.n_instances))
        selected_instances_ids = [i for prob, i in sorted_instances_ids[:n_instances_tot:step]]
        return selected_instances_ids

    def _density_func(self, data_matrix, target=None):
        kernel_matrix = pairwise_kernels(data_matrix, metric=self.metric, **self.kwds)
        # compute instance density as average pairwise similarity
        densities = np.mean(kernel_matrix, 0)
        return densities

    def randomize(self, data_matrix, amount=1.0):
        random.seed(self.random_state)
        self.n_instances = self._auto_n_instances(data_matrix.shape[0])
        self.percentile = random.uniform(0.5, 0.9)
        self.metric = 'rbf'
        self.kwds = {'gamma': random.choice([10 ** x for x in range(-3, 3)])}
        self.random_state = self.random_state ^ random.randint(1, 1e9)

# -----------------------------------------------------------------------------


class DecisionSurfaceSelector(AbstractSelector):

    """
    Transform a set of sparse high dimensional vectors to a smaller set.
    Selection is performed sampling according to the inverse of the ROC score.
    """

    def __init__(self, n_instances=20,
                 estimator=SGDClassifier(average=True, class_weight='balanced', shuffle=True),
                 random_state=1,
                 randomized=False):
        self.name = 'DecisionSurfaceSelector'
        self.n_instances = n_instances
        self.estimator = estimator
        self.random_state = random_state
        self.randomized = randomized

    def __repr__(self):
        serial = []
        serial.append(self.name)
        serial.append('n_instances: %d' % (self.n_instances))
        serial.append('estimator: %s' % str(self.estimator))
        serial.append('random_state: %d' % (self.random_state))
        return '\n'.join(serial)

    def _logistic_function(self, xs, ell=1, k=1, x0=0):
        _max_x = 100
        _min_x = -_max_x
        vals = []
        for x in xs:
            if x > _max_x:
                x = _max_x
            if x < _min_x:
                x = _min_x
            val = ell / (1 + math.exp(-k * (x - x0)))
            vals.append(val)
        return np.array(vals).reshape(1, -1)

    def select(self, data_matrix, target=None):
        if target is None:
            raise Exception('target cannot be None')
        # select maximally ambiguous or unpredictable instances
        probabilities = self._probability_func(data_matrix, target)
        self.scores = probabilities
        if self.randomized:
            # select instances according to their probability
            selected_instances_ids = [self._sample(probabilities) for i in range(self.n_instances)]
        else:
            # select the instances with highest probability
            selected_instances_ids = sorted([(prob, i) for i, prob in enumerate(probabilities)], reverse=True)
            selected_instances_ids = [i for prob, i in selected_instances_ids[:self.n_instances]]
        return selected_instances_ids

    def _probability_func(self, data_matrix, target=None):
        # select maximally ambiguous (max entropy) or unpredictable instances
        self.estimator.fit(data_matrix, target)
        decision_function = getattr(self.estimator, "decision_function", None)
        if decision_function and callable(decision_function):
            preds = self.estimator.decision_function(data_matrix)
            x0 = 0
        else:
            predict_proba = getattr(self.estimator, "predict_proba", None)
            if predict_proba and callable(predict_proba):
                preds = self.estimator.predict_proba(data_matrix)
                x0 = 0.5
            else:
                raise Exception('Estimator seems to lack a decision_function or predict_proba method.')
        # Note: use -entropy to sort from max entropy to min entropy
        entropies = []
        for p in preds:
            ps = self._logistic_function(p, x0=x0)
            nps = normalize(ps, norm='l1').reshape(-1, 1)
            e = - entropy(nps)
            entropies.append(e)
        # normalize to obtain probabilities
        probabilities = entropies / sum(entropies)
        return probabilities

    def randomize(self, data_matrix, amount=1.0):
        random.seed(self.random_state)
        self.n_instances = self._auto_n_instances(data_matrix.shape[0])
        algo = random.choice(['SGDClassifier', 'KNeighborsClassifier'])
        kwds = dict()
        if algo == 'SGDClassifier':
            self.estimator = SGDClassifier(average=True, class_weight='balanced', shuffle=True)
            kwds = dict(n_iter=random.randint(5, 200),
                        penalty=random.choice(['l1', 'l2', 'elasticnet']),
                        l1_ratio=random.uniform(0.1, 0.9),
                        loss=random.choice(['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']),
                        power_t=random.uniform(0.1, 1),
                        alpha=random.choice([10 ** x for x in range(-8, 0)]),
                        eta0=random.choice([10 ** x for x in range(-4, -1)]),
                        learning_rate=random.choice(["invscaling", "constant", "optimal"]))
        if algo == 'KNeighborsClassifier':
            self.estimator = KNeighborsClassifier()
            kwds = dict(n_neighbors=random.randint(3, 100))
        self.estimator.set_params(**kwds)
        self.random_state = self.random_state ^ random.randint(1, 1e9)
