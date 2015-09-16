import logging

from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)


class AutoCluster(object):

    """Clustering estimator that selects the best option from a list of estimators.

    Cluster analysis or clustering is the task of grouping a set of objects in such a way
    that objects in the same group (called a cluster) are more similar (in some sense or another)
    to each other than to those in other groups (clusters).


    Parameters
    ----------
    estimators : list(estimator)
        List of scikit-learn clustering estimators.

    score_func : callable, default silhouette_score
        Function that scores the quality of a cluster label assignment
        given the label assignment and the samples.

    Attributes
    ----------
    score : float
        The value returned by the score_func.

    predictions : array-like, (n_features,)
        The list of class identifiers.

    estimator : scikit-learn estimator
        The clustering estimator that returned the best score.

    n_clusters : int
        The optimal number of clusters according to the score_func.
    """

    def __init__(self, estimators=[AgglomerativeClustering(linkage='ward'),
                                   AgglomerativeClustering(linkage='complete'),
                                   AgglomerativeClustering(linkage='average'),
                                   MiniBatchKMeans()],
                 score_func=silhouette_score):
        self.estimators = estimators
        self.score_func = score_func

    def set_params(self, *args, **kwargs):
        """Set the parameters of all estimators."""
        for estimator in self.estimators:
            estimator.set_params(*args, **kwargs)
        return self

    def fit(self, data_matrix):
        """Fit all clustering estimators on the same samples.

        Parameters
        ----------
        data_matrix : array-like, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        self
        """
        for estimator in self.estimators:
            estimator.fit(data_matrix)
        return self

    def predict(self, data_matrix):
        """Predict cluster labels for samples in data_matrix using the best estimator.

        Parameters
        ----------
        data_matrix : array, shape = (n_samples, n_features)
          Samples.

        Returns
        -------
        predictions : array, shape = (n_samples,)
            Predicted cluster label per sample.

        Notes
        -----
        The predictions returned are from the estimator that obtains the maximum value for score_func.
        """
        self.score, self.predictions, self.estimator = max(self._predict(data_matrix))
        return self.predictions

    def _predict(self, data_matrix):
        for estimator in self.estimators:
            predictions = estimator.predict(data_matrix)
            score = self.score_func(data_matrix, predictions)
            yield (score, predictions, estimator)

    def fit_predict(self, data_matrix):
        """Fit to data, then predict it.

        Parameters
        ----------
        data_matrix : array, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        predictions : array, shape = (n_samples,)
            Predicted cluster label per sample.

        Notes
        -----
        The predictions returned are from the estimator that obtains the maximum value for score_func.
        """
        self.fit(data_matrix)
        return self.predict(data_matrix)

    def optimize(self, data_matrix, max_n_clusters=20):
        """Select the optimal number of clusters according to score_func.

        Parameters
        ----------
        data_matrix : array-like, shape = [n_samples, n_features]
            Samples.

        max_n_clusters : int, default=20
            Maximum number of clusters.

        Returns
        -------
            self

        Notes
        -----
            After execution the following attributes are set:

            score: the value returned by the score_func

            predictions: the list of class labels

            estimator: the clustering estimator that returned the best score

            n_clusters: the optimal number of clusters according to the score_func
        """
        self.score, \
            self.predictions, \
            self.estimator, \
            self.n_clusters = max(self._optimize(data_matrix, max_n_clusters))
        return self

    def _optimize(self, data_matrix, max_n_clusters=None):
        for n_clusters in max_n_clusters:
            self.set_params(n_clusters=n_clusters)
            predictions = self.predict(data_matrix)
            yield (self.score, predictions, self.estimator, n_clusters)
