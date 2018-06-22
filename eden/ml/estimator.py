#!/usr/bin/env python
"""Provides scikit interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from eden.graph import Vectorizer
from eden.util import timeit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Perceptron
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import multiprocessing as mp
from eden.ml.estimator_utils import balance, subsample, paired_shuffle
import random
import logging

logger = logging.getLogger()


class EdenEstimator(BaseEstimator, ClassifierMixin):
    """Build an estimator for graphs."""

    def __init__(self, r=3, d=8, nbits=16, discrete=True,
                 balance=False, subsample_size=200, ratio=2,
                 normalization=False, inner_normalization=False,
                 penalty='elasticnet'):
        """construct."""
        self.set_params(r, d, nbits, discrete, balance, subsample_size,
                        ratio, normalization, inner_normalization,
                        penalty)

    def set_params(self, r=3, d=8, nbits=16, discrete=True,
                   balance=False, subsample_size=200, ratio=2,
                   normalization=False, inner_normalization=False,
                   penalty='elasticnet'):
        """setter."""
        self.r = r
        self.d = d
        self.nbits = nbits
        self.normalization = normalization
        self.inner_normalization = inner_normalization
        self.discrete = discrete
        self.balance = balance
        self.subsample_size = subsample_size
        self.ratio = ratio
        if penalty == 'perceptron':
            self.model = Perceptron(max_iter=5, tol=None)
        else:
            self.model = SGDClassifier(
                average=True, class_weight='balanced', shuffle=True,
                penalty=penalty, max_iter=5, tol=None)
        self.vectorizer = Vectorizer(
            r=self.r, d=self.d,
            normalization=self.normalization,
            inner_normalization=self.inner_normalization,
            discrete=self.discrete,
            nbits=self.nbits)
        return self

    def transform(self, graphs):
        """transform."""
        x = self.vectorizer.transform(graphs)
        return x

    @timeit
    def kernel_matrix(self, graphs):
        """kernel_matrix."""
        x = self.transform(graphs)
        return metrics.pairwise.pairwise_kernels(x, metric='linear')

    def fit(self, graphs, targets, randomize=True):
        """fit."""
        if self.balance:
            if randomize:
                bal_graphs, bal_targets = balance(
                    graphs, targets, None, ratio=self.ratio)
            else:
                samp_graphs, samp_targets = subsample(
                    graphs, targets, subsample_size=self.subsample_size)
                x = self.transform(samp_graphs)
                self.model.fit(x, samp_targets)
                bal_graphs, bal_targets = balance(
                    graphs, targets, self, ratio=self.ratio)
            size = len(bal_targets)
            logger.debug('Dataset size=%d' % (size))
            x = self.transform(bal_graphs)
            self.model = self.model.fit(x, bal_targets)
        else:
            x = self.transform(graphs)
            self.model = self.model.fit(x, targets)
        return self

    def predict(self, graphs):
        """predict."""
        x = self.transform(graphs)
        preds = self.model.predict(x)
        return preds

    def decision_function(self, graphs):
        """decision_function."""
        x = self.transform(graphs)
        preds = self.model.decision_function(x)
        return preds

    @timeit
    def cross_val_score(self, graphs, targets,
                        scoring='roc_auc', cv=5):
        """cross_val_score."""
        x = self.transform(graphs)
        scores = cross_val_score(
            self.model, x, targets, cv=cv, scoring=scoring)
        return scores

    @timeit
    def cross_val_predict(self, graphs, targets, cv=5):
        """cross_val_score."""
        x = self.transform(graphs)
        scores = cross_val_predict(
            self.model, x, targets, cv=cv, method='decision_function')
        return scores

    @timeit
    def cluster(self, graphs, n_clusters=16):
        """cluster."""
        x = self.transform(graphs)
        clust_est = MiniBatchKMeans(n_clusters=n_clusters)
        cluster_ids = clust_est.fit_predict(x)
        return cluster_ids

    @timeit
    def model_selection(self, graphs, targets,
                        n_iter=30, subsample_size=None):
        """model_selection_randomized."""
        param_distr = {"r": list(range(1, 5)), "d": list(range(0, 10))}
        if subsample_size:
            graphs, targets = subsample(
                graphs, targets, subsample_size=subsample_size)

        pool = mp.Pool()
        scores = pool.map(_eval, [(graphs, targets, param_distr)] * n_iter)
        pool.close()
        pool.join()

        best_params = max(scores)[1]
        logger.debug("Best parameters:\n%s" % (best_params))
        self = EdenEstimator(**best_params)
        return self

    @timeit
    def learning_curve(self, graphs, targets,
                       cv=5, n_steps=10, start_fraction=0.1):
        """learning_curve."""
        graphs, targets = paired_shuffle(graphs, targets)
        x = self.transform(graphs)
        train_sizes = np.linspace(start_fraction, 1.0, n_steps)
        scoring = 'roc_auc'
        train_sizes, train_scores, test_scores = learning_curve(
            self.model, x, targets,
            cv=cv, train_sizes=train_sizes,
            scoring=scoring)
        return train_sizes, train_scores, test_scores

    @timeit
    def bias_variance_decomposition(self, graphs, targets,
                                    cv=5, n_bootstraps=10):
        """bias_variance_decomposition."""
        x = self.transform(graphs)
        score_list = []
        for i in range(n_bootstraps):
            scores = cross_val_score(
                self.model, x, targets, cv=cv)
            score_list.append(scores)
        score_list = np.array(score_list)
        mean_scores = np.mean(score_list, axis=1)
        std_scores = np.std(score_list, axis=1)
        return mean_scores, std_scores


class EdenRegressor(BaseEstimator, RegressorMixin):
    """Build a regressor for graphs."""

    def __init__(self, r=3, d=8, nbits=16, discrete=True,
                 normalization=True, inner_normalization=True,
                 penalty='elasticnet', loss='squared_loss'):
        """construct."""
        self.set_params(r, d, nbits, discrete,
                        normalization, inner_normalization,
                        penalty, loss)

    def set_params(self, r=3, d=8, nbits=16, discrete=True,
                   normalization=True, inner_normalization=True,
                   penalty='elasticnet', loss='squared_loss'):
        """setter."""
        self.r = r
        self.d = d
        self.nbits = nbits
        self.normalization = normalization
        self.inner_normalization = inner_normalization
        self.discrete = discrete
        self.model = SGDRegressor(
            loss=loss, penalty=penalty,
            average=True, shuffle=True,
            max_iter=5, tol=None)
        self.vectorizer = Vectorizer(
            r=self.r, d=self.d,
            normalization=self.normalization,
            inner_normalization=self.inner_normalization,
            discrete=self.discrete,
            nbits=self.nbits)
        return self

    def transform(self, graphs):
        """transform."""
        x = self.vectorizer.transform(graphs)
        return x

    @timeit
    def kernel_matrix(self, graphs):
        """kernel_matrix."""
        x = self.transform(graphs)
        return metrics.pairwise.pairwise_kernels(x, metric='linear')

    def fit(self, graphs, targets, randomize=True):
        """fit."""
        x = self.transform(graphs)
        self.model = self.model.fit(x, targets)
        return self

    def predict(self, graphs):
        """predict."""
        x = self.transform(graphs)
        preds = self.model.predict(x)
        return preds

    def decision_function(self, graphs):
        """decision_function."""
        return self.predict(graphs)


def _sample_params(param_distr):
    params = dict()
    for key in param_distr:
        params[key] = random.choice(param_distr[key])
    return params


def _eval_params(graphs, targets, param_distr):
    # sample parameters
    params = _sample_params(param_distr)
    # create model with those parameters
    est = EdenEstimator(**params)
    # run a cross_val_score
    scores = est.cross_val_score(graphs, targets)
    # return average
    return np.mean(scores), params


def _eval(data):
    return _eval_params(*data)
