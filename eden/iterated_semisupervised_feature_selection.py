import random
import logging
import numpy as np

from sklearn.semi_supervised import LabelSpreading
from sklearn.feature_selection import RFECV
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def semisupervised_target(target=None,
                          unknown_fraction=None,
                          known_fraction=None,
                          random_state=1):
    """Simulates partial knowledge on the targets by randomly masking
    targets with the value -1.

    Parameters
    ----------
    target : array-like, shape = (n_samples)
        Class labels.

    unknown_fraction : float
        Fraction of desired unknown labels.

    known_fraction : float
        Fraction of desired known labels.

    random_state : int
        Seed for the random number generator.

    Returns
    -------
    target : array-like, shape = (n_samples)
        Class labels using -1 for the unknown class.
    """
    if unknown_fraction is not None and known_fraction is not None:
        if unknown_fraction != 1 - known_fraction:
            raise Exception('unknown_fraction and known_fraction are inconsistent. bailing out')
    target = LabelEncoder().fit_transform(target)

    if known_fraction is not None:
        unknown_fraction = 1 - known_fraction

    if unknown_fraction == 0:
        return target
    elif unknown_fraction == 1:
        return None
    else:
        label_ids = [1] * int(len(target) * unknown_fraction) + \
            [0] * int(len(target) * (1 - unknown_fraction))
        random.seed(random_state)
        random.shuffle(label_ids)
        random_unlabeled_points = np.where(label_ids)
        labels = np.copy(target)
        labels[random_unlabeled_points] = -1
        return labels


class IteratedSemiSupervisedFeatureSelection(object):

    """Feature selection estimator that uses an iterated approach in a semisueprvised setting.


    Parameters
    ----------
    estimator: scikit-learn estimator (default SGDClassifier)
        Estimator used in the recursive feature elimination algorithm.

    n_iter : int
        The maximum number of iterations.

    min_feature_ratio : float
        The ratio between the initial number of features and the number
        of features after the selection.
    """

    def __init__(self, estimator=SGDClassifier(average=True, shuffle=True, penalty='elasticnet'),
                 n_iter=30,
                 min_feature_ratio=0.1):
        self.estimator = estimator
        self.n_iter = n_iter
        self.min_feature_ratio = min_feature_ratio
        self.selectors = []

    def fit(self, data_matrix=None, target=None):
        """Fit the estimator on the samples.

        Parameters
        ----------
        data_matrix : array-like, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        self
        """
        n_features_orig = data_matrix.shape[1]
        for i in range(self.n_iter):
            n_features_input = data_matrix.shape[1]
            target = self._semi_supervised_learning(data_matrix, target)
            data_matrix = self._feature_selection(data_matrix, target)
            n_features_output = data_matrix.shape[1]
            if self._terminate(n_features_orig, n_features_input, n_features_output):
                # remove last selector since it does not satisfy conditions
                self.selectors.pop(-1)
                break
        return self

    def _terminate(self, n_features_orig, n_features_input, n_features_output):
        if n_features_output == n_features_input:
            return True
        if n_features_output < n_features_orig * self.min_feature_ratio:
            return True
        if n_features_output < 3:
            return True
        return False

    def transform(self, data_matrix=None):
        """Reduce the data matrix to the features selected in the fit phase.

        Parameters
        ----------
        data_matrix : array, shape = (n_samples, n_features)
          Samples.

        Returns
        -------
        data_matrix : array, shape = (n_samples, n_features_new)
            Transformed array.
        """
        for selector in self.selectors:
            data_matrix = selector.transform(data_matrix)
        return data_matrix

    def fit_transform(self, data_matrix=None, target=None):
        """Fit the estimator on the samples and reduce the data matrix to
        the selected features.

        Iterate semi supervised label spreading and feature selection:
        due to feature selection the metric space changes and with it so does
        the result of label spreading.

        Parameters
        ----------
        data_matrix : array-like, shape = (n_samples, n_features)
            Samples.

        target : array-like, shape = (n_samples)

        Returns
        -------
        data_matrix : array, shape = (n_samples, n_features_new)
            Transformed array.
        """
        data_matrix_ = data_matrix.copy()
        self.fit(data_matrix, target)
        return self.transform(data_matrix_)

    def _semi_supervised_learning(self, data_matrix, target):
        semi_supervised_estimator = LabelSpreading(kernel='knn', n_neighbors=5)
        semi_supervised_estimator.fit(data_matrix, target)
        predicted_target = semi_supervised_estimator.predict(data_matrix)
        return self._clamp(target, predicted_target)

    def _clamp(self, target, predicted_target):
        extended_target = []
        for pred_label, label in zip(predicted_target, target):
            if label != -1 and pred_label != label:
                extended_target.append(label)
            else:
                extended_target.append(pred_label)
        return np.array(extended_target)

    def _feature_selection(self, data_matrix, target):
        # perform recursive feature elimination
        selector = RFECV(self.estimator, step=0.1, cv=10)
        data_matrix = selector.fit_transform(data_matrix, target)
        self.selectors.append(selector)
        return data_matrix
