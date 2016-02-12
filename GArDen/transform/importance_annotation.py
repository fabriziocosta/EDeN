#!/usr/bin/env python
"""Provides annotation of importance of nodes."""

from eden.graph import Vectorizer

from sklearn.base import BaseEstimator, ClassifierMixin

import logging
logger = logging.getLogger(__name__)


class AnnotateImportance(BaseEstimator, ClassifierMixin):
    """Annotate minimal cycles."""

    def __init__(self,
                 estimator=None):
        """Construct."""
        self.estimator = estimator
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
        for param in params:
            if "vectorizer__" in param:
                key = param.split('__')[1]
                val = params[param]
                params_vectorizer[key] = val
        self.vectorizer.set_params(**params_vectorizer)
        return self

    def transform(self, graphs):
        """Transform."""
        try:
            annotated_graphs = self.vectorizer.annotate(
                graphs,
                estimator=self.estimator,
                reweight=1.0,
                relabel=False)
            for graph in annotated_graphs:
                yield graph
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)