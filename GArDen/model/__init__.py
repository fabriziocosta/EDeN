#!/usr/bin/env python
"""Provides wrappers to predictive algorithms."""

from eden.util import vectorize
from eden.graph import Vectorizer
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from scipy.sparse import vstack
import numpy as np
from itertools import tee, izip
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import SGDClassifier, SGDRegressor

import logging
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------


class TransformerWrapper(BaseEstimator, ClassifierMixin):
    """TransformerWrapper."""

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
            self.program.fit(graphs)
            return self
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def transform(self, graphs):
        """predict."""
        try:
            for graph in graphs:
                transformed_graph = self._transform(graph)
                yield transformed_graph
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _transform(self, graph):
        return graph

# ------------------------------------------------------------------------------


class KNNWrapper(BaseEstimator, ClassifierMixin):
    """KNNWrapper."""

    def __init__(self, program=NearestNeighbors(n_neighbors=2)):
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
            self.graphs = list(graphs)
            data_matrix = vectorize(self.graphs,
                                    vectorizer=self.vectorizer,
                                    **self.params_vectorize)
            self.program = self.program.fit(data_matrix)
            return self
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def predict(self, graphs):
        """predict."""
        try:
            graphs, graphs_ = tee(graphs)
            data_matrix = vectorize(graphs_,
                                    vectorizer=self.vectorizer,
                                    **self.params_vectorize)
            distances, indices = self.program.kneighbors(data_matrix)
            for knn_dists, knn_ids, graph in izip(distances, indices, graphs):
                neighbor_graphs = []
                for knn_id in knn_ids:
                    neighbor_graphs.append(self.graphs[knn_id])
                graph.graph['neighbors'] = neighbor_graphs
                graph.graph['ids'] = knn_ids
                graph.graph['distances'] = knn_dists
                yield graph
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)


# ------------------------------------------------------------------------------


class ClassifierWrapper(BaseEstimator, ClassifierMixin):
    """Classifier."""

    def __init__(self,
                 program=SGDClassifier(average=True,
                                       class_weight='balanced',
                                       shuffle=True)):
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
            graphs, graphs_ = tee(graphs)
            data_matrix = vectorize(graphs_,
                                    vectorizer=self.vectorizer,
                                    **self.params_vectorize)
            y = self._extract_targets(graphs)
            # manage case for single class learning
            if len(set(y)) == 1:
                # make negative data matrix
                negative_data_matrix = data_matrix.multiply(-1)
                # make targets
                y = list(y)
                y_neg = [-1] * len(y)
                # concatenate elements
                data_matrix = vstack(
                    [data_matrix, negative_data_matrix], format="csr")
                y = y + y_neg
                y = np.ravel(y)
            self.program = self.program.fit(data_matrix, y)
            return self
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def predict(self, graphs):
        """predict."""
        try:
            graphs, graphs_ = tee(graphs)
            data_matrix = vectorize(graphs_,
                                    vectorizer=self.vectorizer,
                                    **self.params_vectorize)
            predictions = self.program.predict(data_matrix)
            scores = self.program.decision_function(data_matrix)
            for score, prediction, graph in izip(scores, predictions, graphs):
                graph.graph['prediction'] = prediction
                graph.graph['score'] = score
                yield graph
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _extract_targets(self, graphs):
        y = []
        for graph in graphs:
            if graph.graph.get('target', None) is not None:
                y.append(graph.graph['target'])
            else:
                raise Exception('Missing the attribute "target" \
                    in graph dictionary!')
        y = np.ravel(y)
        return y

# ------------------------------------------------------------------------------


class RegressorWrapper(BaseEstimator, RegressorMixin):
    """Regressor."""

    def __init__(self,
                 program=SGDRegressor(average=True, shuffle=True)):
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
            graphs, graphs_ = tee(graphs)
            data_matrix = vectorize(graphs_,
                                    vectorizer=self.vectorizer,
                                    **self.params_vectorize)
            y = self._extract_targets(graphs)
            self.program = self.program.fit(data_matrix, y)
            return self
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def predict(self, graphs):
        """predict."""
        try:
            graphs, graphs_ = tee(graphs)
            data_matrix = vectorize(graphs_,
                                    vectorizer=self.vectorizer,
                                    **self.params_vectorize)
            predictions = self.program.predict(data_matrix)
            for prediction, graph in izip(predictions, graphs):
                graph.graph['prediction'] = prediction
                graph.graph['score'] = prediction
                yield graph
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _extract_targets(self, graphs):
        y = []
        for graph in graphs:
            if graph.graph.get('target', None) is not None:
                y.append(graph.graph['target'])
            else:
                raise Exception('Missing the attribute "target" \
                    in graph dictionary!')
        y = np.ravel(y)
        return y
