#!/usr/bin/env python
"""Provides annotation of importance of nodes."""

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

import logging
logger = logging.getLogger(__name__)

# TODO: parameters for other metrics


class DensityEstimator(BaseEstimator, ClassifierMixin):
    """Compute density of each instance."""

    def __init__(self, metric='rbf', gamma=0.1, reverse=True):
        """Construct."""
        self.reverse = reverse
        self.metric = metric
        self.gamma = gamma

    def decision_function(self, data_matrix):
        """decision_function."""
        try:
            kernel_matrix = pairwise_kernels(data_matrix,
                                             metric=self.metric,
                                             gamma=self.gamma)
            data_size = kernel_matrix.shape[0]
            density = np.sum(kernel_matrix, 0) / data_size
            if self.reverse is True:
                return density * -1
            else:
                return density
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

# TODO: nearest neighbor iterator
