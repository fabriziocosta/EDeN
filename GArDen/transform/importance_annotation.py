#!/usr/bin/env python
"""Provides annotation of importance of nodes."""

from eden.graph import Vectorizer

from sklearn.base import BaseEstimator, ClassifierMixin

import logging
logger = logging.getLogger(__name__)


class AnnotateImportance(BaseEstimator, ClassifierMixin):
    """Annotate minimal cycles."""

    def __init__(self,
                 program=None,
                 relabel=False,
                 reweight=1.0):
        """Construct."""
        self.program = program
        self.relabel = relabel
        self.reweight = reweight
        self.vectorizer = Vectorizer()
        self.params_vectorize = dict()

    def set_params(self, **params):
        """Set the parameters of this program.

        The method.

        Returns
        -------
        self
        """
        # finds parameters for the vectorizer as those that contain "__"
        params_vectorizer = dict()
        params_program = dict()
        for param in params:
            if "vectorizer__" in param:
                key = param.split('__')[1]
                val = params[param]
                params_vectorizer[key] = val
            else:
                params_program[param] = params[param]
        self.program.set_params(**params_program)
        self.vectorizer.set_params(**params_vectorizer)
        return self

    def transform(self, graphs):
        """Transform."""
        try:
            annotated_graphs = self.vectorizer.annotate(
                graphs,
                estimator=self.program,
                reweight=self.reweight,
                relabel=self.relabel)
            for graph in annotated_graphs:
                yield graph
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)
