#!/usr/bin/env python
"""Provides wrappers to predictive algorithms."""

from eden.util import vectorize, describe, is_iterable
from eden.graph import Vectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.sparse import vstack
import numpy as np
from itertools import tee, izip
from collections import defaultdict

import logging
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------


class ClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Classifier."""

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

    def fit(self, graphs):
        """fit."""
        try:
            data_matrix, y = self._make_data_matrix(graphs)
            estimator = self.program.fit(data_matrix, y)
            return estimator
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def predict(self, graphs):
        """fit."""
        try:
            graphs, graphs_ = tee(graphs)
            data_matrix = vectorize(graphs_,
                                    vectorizer=self.vectorizer,
                                    **self.params_vectorize)
            predictions = self.program.predict(data_matrix)
            scores = self.program.decision_function(data_matrix)
            prediction_partition = defaultdict(list)
            for score, prediction, graph in \
                    sorted(izip(scores, predictions, graphs), reverse=True):
                element = dict(score=score, graph=graph)
                prediction_partition[prediction].append(element)
            return prediction_partition
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _make_data_matrix(self, graphs):
        if len(graphs) == 2 and \
                is_iterable(graphs[0]) and is_iterable(graphs[1]):
            # binary classification case
            positive_data_matrix = vectorize(graphs[0],
                                             vectorizer=self.vectorizer,
                                             **self.params_vectorize)
            logger.debug('Positive data: %s' %
                         describe(positive_data_matrix))
            negative_data_matrix = vectorize(graphs[1],
                                             vectorizer=self.vectorizer,
                                             **self.params_vectorize)
            logger.debug('Negative data: %s' %
                         describe(negative_data_matrix))
            yp = [1] * positive_data_matrix.shape[0]
            yn = [-1] * negative_data_matrix.shape[0]
            y = np.array(yp + yn)
            data_matrix = vstack(
                [positive_data_matrix, negative_data_matrix], format="csr")
            return data_matrix, y
        elif len(graphs) == 1 and is_iterable(graphs[0]):
            # case of single class
            positive_data_matrix = vectorize(graphs[0],
                                             vectorizer=self.vectorizer,
                                             **self.params_vectorize)
            logger.debug('Positive data: %s' %
                         describe(positive_data_matrix))
            negative_data_matrix = positive_data_matrix.multiply(-1)
            logger.debug('Negative data: %s' %
                         describe(negative_data_matrix))
            yp = [1] * positive_data_matrix.shape[0]
            yn = [-1] * negative_data_matrix.shape[0]
            y = np.array(yp + yn)
            data_matrix = vstack(
                [positive_data_matrix, negative_data_matrix], format="csr")
            return data_matrix, y
        else:
            raise Exception('Unknown prediction task')
