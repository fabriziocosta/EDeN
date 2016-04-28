#!/usr/bin/env python
"""Provides wrappers to clustering algorithms."""

from eden.util import vectorize
from eden.graph import Vectorizer
from sklearn.base import BaseEstimator, ClusterMixin

import logging
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------


class ClustererWrapper(BaseEstimator, ClusterMixin):
    """Clusterer."""

    def __init__(self, program=None):
        """Construct."""
        self.program = program
        self.vectorizer = Vectorizer()
        self.params_vectorize = dict()

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method.

        Returns
        -------
        self
        """
        # finds parameters for the vectorizer as those that contain "__"
        params_vectorizer = dict()
        params_clusterer = dict()
        for param in params:
            if "vectorizer__" in param:
                key = param.split('__')[1]
                val = params[param]
                params_vectorizer[key] = val
            elif "vectorize__" in param:
                key = param.split('__')[1]
                val = params[param]
                self.params_vectorize[key] = val
            else:
                params_clusterer[param] = params[param]
        self.program.set_params(**params_clusterer)
        self.vectorizer.set_params(**params_vectorizer)
        return self

    def fit_predict(self, graphs):
        """fit_predict."""
        try:
            data_matrix = vectorize(graphs,
                                    vectorizer=self.vectorizer,
                                    **self.params_vectorize)
            predictions = self.program.fit_predict(data_matrix)
            return predictions
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)


# ------------------------------------------------------------------------------


class ExplicitClusterer(BaseEstimator, ClusterMixin):
    """ExplicitClusterer."""

    def __init__(self, attribute=None):
        """Construct."""
        self.attribute = attribute

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method.

        Returns
        -------
        self
        """
        for param in params:
            self.__dict__[param] = params[param]
        return self

    def fit_predict(self, graphs):
        """fit_predict."""
        try:
            for graph in graphs:
                prediction = graph.graph.get(self.attribute, None)
                yield prediction
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)
