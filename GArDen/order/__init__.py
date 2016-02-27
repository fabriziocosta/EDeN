#!/usr/bin/env python
"""Provides wrappers to clustering algorithms."""

from eden.util import vectorize
from eden.graph import Vectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from itertools import tee

import logging
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------


class OrdererWrapper(BaseEstimator, ClassifierMixin):
    """Orderer."""

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
        params_orderer = dict()
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
                params_orderer[param] = params[param]
        self.program.set_params(**params_orderer)
        self.vectorizer.set_params(**params_vectorizer)
        return self

    def decision_function(self, graphs):
        """decision_function."""
        try:
            graphs, graphs_ = tee(graphs)
            data_matrix = vectorize(graphs_,
                                    vectorizer=self.vectorizer,
                                    **self.params_vectorize)
            scores = self.program.decision_function(data_matrix)
            return scores
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)
