#!/usr/bin/env python

import numpy as np
from scipy.sparse import vstack
from sklearn import cross_validation
import random
import joblib
from eden.graph import Vectorizer
from sklearn.linear_model import SGDClassifier
from itertools import tee


class EDeNModel(object):

    def __init__(self, pre_processor=None, vectorizer=Vectorizer(complexity=1), estimator=SGDClassifier(class_weight='auto', shuffle=True)):
        self.pre_processor = pre_processor
        self.vectorizer = vectorizer
        self.estimator = estimator
        self.pre_processor_args = None
        self.vectorizer_args = None
        self.estimator_args = None

    def save(self, model_name):
        joblib.dump(self, model_name, compress=1)

    def load(self, obj):
        self.__dict__.update(joblib.load(obj).__dict__)

    def is_iterable(self, test):
        if hasattr(test, '__iter__'):
            return True
        else:
            return False

    def _data_matrix(self, iterable, n_jobs=1):
        assert(self.is_iterable(iterable)), 'Not iterable'
        iterator = self.pre_processor(iterable, **self.pre_processor_args)
        self.vectorizer.set_params(**self.vectorizer_args)
        X = self.vectorizer.transform(iterator, n_jobs=n_jobs)
        return X

    def _data_matrices(self, iterable_pos, iterable_neg, n_jobs=1):
        assert(self.is_iterable(iterable_pos)
               and self.is_iterable(iterable_neg)), 'Not iterable'
        self.vectorizer.set_params(**self.vectorizer_args)
        iterator_pos = self.pre_processor(
            iterable_pos, **self.pre_processor_args)
        Xpos = self.vectorizer.transform(iterator_pos, n_jobs=n_jobs)
        iterator_neg = self.pre_processor(
            iterable_neg, **self.pre_processor_args)
        Xneg = self.vectorizer.transform(iterator_neg, n_jobs=n_jobs)
        yp = [1] * Xpos.shape[0]
        yn = [-1] * Xneg.shape[0]
        y = np.array(yp + yn)
        X = vstack([Xpos, Xneg], format="csr")
        return X, y

    def _sample(self, parameters):
        parameters_sample = dict()
        for parameter in parameters:
            values = parameters[parameter]
            value = random.choice(values)
            parameters_sample[parameter] = value
        return parameters_sample

    def optimize(self, iterable_pos, iterable_neg, verbose=True, n_jobs=1, n_iter=20, pre_processor_parameters=dict(), vectorizer_paramters=dict(), estimator_parameters=dict(), cv=10, scoring='roc_auc'):
        best_pre_processor_args_ = dict()
        best_vectorizer_args_ = dict()
        best_estimator_args_ = dict()
        best_score_ = 0
        for i in range(n_iter):
            # sample paramters randomly
            self.pre_processor_args = self._sample(pre_processor_parameters)
            self.vectorizer_args = self._sample(vectorizer_paramters)
            self.estimator_args = self._sample(estimator_parameters)
            # copy the iterators for later re-use
            iterable_pos, iterable_pos_ = tee(iterable_pos)
            iterable_neg, iterable_neg_ = tee(iterable_neg)
            self.estimator.set_params(**self.estimator_args)
            X, y = self._data_matrices(
                iterable_pos_, iterable_neg_, n_jobs=n_jobs)
            scores = cross_validation.cross_val_score(
                self.estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
            # consider as score the mean-std for a conservative and robust
            # estimate
            score = np.mean(scores) - np.std(scores)
            # update the best confirguration
            if best_score_ <= score:
                best_score_ = score
                best_pre_processor_args_ = self.pre_processor_args
                best_vectorizer_args_ = self.vectorizer_args
                best_estimator_args_ = self.estimator_args
                if verbose:
                    print('Iteration: %d'%i)
                    print('Best score: %f' % score)
                    print('Pre_processor: %s' % self.pre_processor_args)
                    print('Vectorizer: %s' % self.vectorizer_args)
                    print('Estimator: %s' % self.estimator_args)
        # store the best hyperparamter configuration
        self.pre_processor_args = best_pre_processor_args_
        self.vectorizer_args = best_vectorizer_args_
        self.estimator_args = best_estimator_args_
        # fit the estimator using the best hyperparameters
        self.fit(iterable_pos, iterable_neg, n_jobs=n_jobs)

    def fit(self, iterable_pos, iterable_neg, n_jobs=1):
        self.estimator.set_params(**self.estimator_args)
        X, y = self._data_matrices(iterable_pos, iterable_neg, n_jobs=n_jobs)
        self.estimator.fit(X, y)

    def predict(self, iterable, n_jobs=1):
        X = self._data_matrix(iterable, n_jobs=n_jobs)
        return self.estimator.predict(X)

    def decision_function(self, iterable, n_jobs=1):
        X = self._data_matrix(iterable, n_jobs=n_jobs)
        return self.estimator.decision_function(X)

    def estimate(self, iterable_pos, iterable_neg, n_jobs=1, cv=10):
        X, y = self._data_matrices(iterable_pos, iterable_neg, n_jobs=n_jobs)
        print 'Classifier:'
        print self.estimator
        print '-' * 80
        print 'Predictive performance:'
        # assess the generalization capacity of the model via a 10-fold cross
        # validation
        for scoring in ['accuracy', 'precision', 'recall', 'f1', 'average_precision', 'roc_auc']:
            scores = cross_validation.cross_val_score(
                self.estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
            print('%20s: %.3f +- %.3f' %
                  (scoring, np.mean(scores), np.std(scores)))
        print '-' * 80
