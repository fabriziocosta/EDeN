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

    random_state : int (default 1)
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
        labels = []
        known_fraction = 0.2
        class_set = set(target)
        class_samples = {}
        for c in class_set:
            cs = [id for id in range(len(target)) if target[id] == c]
            n_desired = int(max(1, len(cs) * known_fraction))
            class_samples[c] = random.sample(cs, n_desired)
        for id in range(len(target)):
            if id in class_samples[target[id]]:
                val = target[id]
            else:
                val = -1
            labels.append(val)
        return np.array(labels).reshape(-1, 1)


class IteratedSemiSupervisedFeatureSelection(object):

    """Feature selection estimator that uses an iterated approach in a semi-supervised setting.


    Parameters
    ----------
    estimator: scikit-learn estimator (default SGDClassifier)
        Estimator used in the recursive feature elimination algorithm.

    n_iter : int (default 30)
        The maximum number of iterations.

    min_feature_ratio : float (default 0.1)
        The ratio between the initial number of features and the number
        of features after the selection.
    """

    def __init__(self,
                 estimator=SGDClassifier(average=True, shuffle=True, penalty='elasticnet'),
                 n_iter=30,
                 step=.1,
                 cv=5,
                 min_feature_ratio=0.1,
                 n_neighbors=5):
        self.estimator = estimator
        self.n_iter = n_iter
        self.step = step
        self.cv = cv
        self.min_feature_ratio = min_feature_ratio
        self.n_neighbors = n_neighbors
        self.feature_selectors = []

    def fit(self, data_matrix=None, target=None):
        """Fit the estimator on the samples.

        Parameters
        ----------
        data_matrix : array-like, shape = (n_samples, n_features)
            Samples.

        target : array-like, shape = (n_samples, )
            Array containing partial class information (use -1 to indicate unknown class).

        Returns
        -------
        self
        """
        n_features_orig = data_matrix.shape[1]
        for i in range(self.n_iter):
            n_features_input = data_matrix.shape[1]
            target_new = self._semi_supervised_learning(data_matrix, target)
            if len(set(target_new)) < 2:
                # remove last feature_selector since it does not satisfy conditions
                self.feature_selectors.pop(-1)
                break
            data_matrix = self._feature_selection(data_matrix, target_new)
            n_features_output = data_matrix.shape[1]
            if self._terminate(n_features_orig, n_features_input, n_features_output):
                if len(self.feature_selectors) > 0:
                    # remove last feature_selector since it does not satisfy conditions
                    self.feature_selectors.pop(-1)
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
        data_matrix_new = data_matrix.copy()
        for feature_selector in self.feature_selectors:
            data_matrix_new = feature_selector.transform(data_matrix_new)
        return data_matrix_new

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

        target : array-like, shape = (n_samples, )
            Array containing class information.

        Returns
        -------
        data_matrix : array, shape = (n_samples, n_features_new)
            Transformed array.
        """
        data_matrix_copy = data_matrix.copy()
        self.fit(data_matrix, target)
        return self.transform(data_matrix_copy)

    def _semi_supervised_learning(self, data_matrix, target):
        n_classes = len(set(target))
        # if there are too few classes (e.g. less than -1 and at least 2 other classes)
        # then just bail out and return the original target
        # otherwise one cannot meaningfully spread the information of only one class
        if n_classes > 2:
            semi_supervised_estimator = LabelSpreading(kernel='knn', n_neighbors=self.n_neighbors)
            semi_supervised_estimator.fit(data_matrix, target)
            predicted_target = semi_supervised_estimator.predict(data_matrix)
            predicted_target = self._clamp(target, predicted_target)
            predicted_target = predicted_target.T.tolist()[0]
            return predicted_target
        else:
            return target

    def _clamp(self, target, predicted_target):
        extended_target = []
        for pred_label, label in zip(predicted_target, target):
            if label != -1 and pred_label != label:
                extended_target.append(label)
            else:
                extended_target.append(pred_label)
        return np.array(extended_target).reshape(-1, 1)

    def _feature_selection(self, data_matrix, target):
        try:
            # perform recursive feature elimination
            step = max(int(data_matrix.shape[1] * self.step), 1)
            feature_selector = RFECV(self.estimator, step=step, cv=self.cv)
            data_matrix_out = feature_selector.fit_transform(data_matrix, target)
            self.feature_selectors.append(feature_selector)
            return data_matrix_out
        except Exception as e:
            logger.debug(e)
            return data_matrix
