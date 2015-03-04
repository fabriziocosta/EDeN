#!/usr/bin/env python

import numpy as np
from scipy.sparse import vstack
from sklearn import cross_validation
import random
import time
import datetime
import joblib
import pprint
from sklearn.linear_model import SGDClassifier
from itertools import tee
from sklearn.metrics import precision_recall_curve, roc_curve
from eden.util import fit_estimator, selection_iterator
from eden.util.util import report_base_statistics
from eden.graph import Vectorizer


class BinaryClassificationModel(object):

    def __init__(self, pre_processor=None, vectorizer=Vectorizer(complexity=1), estimator=SGDClassifier(class_weight='auto', shuffle=True), random_seed=1):
        self.pre_processor = pre_processor
        self.vectorizer = vectorizer
        self.estimator = estimator
        self.pre_processor_args = None
        self.vectorizer_args = None
        self.estimator_args = None
        random.seed(random_seed)

    def save(self, model_name):
        joblib.dump(self, model_name, compress=1)

    def load(self, obj):
        self.__dict__.update(joblib.load(obj).__dict__)

    def get_pre_processor():
        return self.pre_processor

    def get_vectorizer():
        return self.vectorizer

    def get_estimator():
        return self.estimator

    def _is_iterable(self, test):
        if hasattr(test, '__iter__'):
            return True
        else:
            return False

    def _data_matrix(self, iterable, n_jobs=1):
        assert(self._is_iterable(iterable)), 'Not iterable'
        iterator = self.pre_processor(iterable, **self.pre_processor_args)
        self.vectorizer.set_params(**self.vectorizer_args)
        X = self.vectorizer.transform(iterator, n_jobs=n_jobs)
        return X

    def _data_matrices(self, iterable_pos, iterable_neg, n_jobs=1):
        assert(self._is_iterable(iterable_pos)
               and self._is_iterable(iterable_neg)), 'Not iterable'
        self.vectorizer.set_params(**self.vectorizer_args)
        iterator_pos = self.pre_processor(
            iterable_pos, **self.pre_processor_args)
        Xpos = self.vectorizer.transform(iterator_pos, n_jobs=n_jobs)
        iterator_neg = self.pre_processor(
            iterable_neg, **self.pre_processor_args)
        Xneg = self.vectorizer.transform(iterator_neg, n_jobs=n_jobs)
        return self._assemble_data_matrix(Xpos, Xneg)

    def _assemble_data_matrix(self, Xpos, Xneg):
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

    def optimize(self, iterable_pos, iterable_neg,
                 n_iter=20,
                 pre_processor_parameters=dict(),
                 vectorizer_parameters=dict(),
                 estimator_parameters=dict(),
                 verbose=True,
                 n_jobs=1,
                 cv=10,
                 scoring='roc_auc'):
        def print_args():
            print('Current parameters:')
            print('Pre_processor:')
            pprint.pprint(self.pre_processor_args)
            print('Vectorizer:')
            pprint.pprint(self.vectorizer_args)
            print('Estimator:')
            pprint.pprint(self.estimator_args)
        if verbose:
            print('Parameters range:')
            print('Pre_processor:')
            pprint.pprint(pre_processor_parameters)
            print('Vectorizer:')
            pprint.pprint(vectorizer_parameters)
            print('Estimator:')
            pprint.pprint(estimator_parameters)
        best_pre_processor_args_ = dict()
        best_vectorizer_args_ = dict()
        best_estimator_args_ = dict()
        best_score_ = best_score_mean_ = best_score_std_ = 0
        start = time.time()
        mean_len_pre_processor_parameters = np.mean(
            [len(pre_processor_parameters[p]) for p in pre_processor_parameters])
        mean_len_vectorizer_parameters = np.mean(
            [len(vectorizer_parameters[p]) for p in vectorizer_parameters])
        for i in range(n_iter):
            try:
                self.estimator_args = self._sample(estimator_parameters)
                self.estimator.set_params(n_jobs=n_jobs, **self.estimator_args)
                # build data matrix only the first time or if needed e.g. because
                # there are more choices in the paramter settings for the
                # pre_processor or the vectorizer
                if i == 0 or mean_len_pre_processor_parameters != 1 or mean_len_vectorizer_parameters != 1:
                    # sample paramters randomly
                    self.pre_processor_args = self._sample(
                        pre_processor_parameters)
                    self.vectorizer_args = self._sample(vectorizer_parameters)
                    # copy the iterators for later re-use
                    iterable_pos, iterable_pos_ = tee(iterable_pos)
                    iterable_neg, iterable_neg_ = tee(iterable_neg)
                    X, y = self._data_matrices(
                        iterable_pos_, iterable_neg_, n_jobs=n_jobs)
                scores = cross_validation.cross_val_score(
                    self.estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
            except Exception as e:
                if verbose:
                    print('Failed iteration: %d/%d (at %.1f sec; %s)' %
                          (i + 1, n_iter, time.time() - start, str(datetime.timedelta(seconds=(time.time() - start)))))
                    print e.__doc__
                    print e.message
                    print('Failed with the following setting:')
                    print_args()
                    print '...continuing'
            else:
                # consider as score the mean-std for a robust estimate
                score_mean = np.mean(scores)
                score_std = np.std(scores)
                score = score_mean - score_std
                # update the best confirguration
                if best_score_ < score:
                    best_score_ = score
                    best_score_mean_ = score_mean
                    best_score_std_ = score_std
                    best_pre_processor_args_ = self.pre_processor_args
                    best_vectorizer_args_ = self.vectorizer_args
                    best_estimator_args_ = self.estimator_args
                    if verbose:
                        print
                        print('Iteration: %d/%d (at %.1f sec; %s)' %
                          (i + 1, n_iter, time.time() - start, str(datetime.timedelta(seconds=(time.time() - start)))))
                        print('Best score (%s): %f (%f +- %f)' %
                              (scoring, best_score_, best_score_mean_, best_score_std_))
                        print_args()
                        print 'Instances: %d ; Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1],  X.getnnz() / X.shape[0])
                        print report_base_statistics(y)
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

    def estimate(self, iterable_pos, iterable_neg, n_jobs=1, cv=10, plots=False):
        iterable_pos, iterable_pos_ = tee(iterable_pos)
        iterable_pos, iterable_pos_ = tee(iterable_pos)
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


class ActiveLearningBinaryClassificationModel(BinaryClassificationModel):

    def __init__(self, pre_processor=None, vectorizer=Vectorizer(complexity=1), estimator=SGDClassifier(class_weight='auto', shuffle=True), random_seed=1):
        self.pre_processor = pre_processor
        self.vectorizer = vectorizer
        self.estimator = estimator
        self.pre_processor_args = None
        self.vectorizer_args = None
        self.estimator_args = None
        random.seed(random_seed)

    def optimize(self, iterable_pos, iterable_neg,
                 n_active_learning_iterations=2,
                 size_positive=-1,
                 size_negative=-1,
                 lower_bound_threshold_positive=-1,
                 upper_bound_threshold_positive=1,
                 lower_bound_threshold_negative=-1,
                 upper_bound_threshold_negative=1,
                 n_iter=20,
                 pre_processor_parameters=dict(),
                 vectorizer_parameters=dict(),
                 estimator_parameters=dict(),
                 verbose=True,
                 n_jobs=1,
                 cv=10,
                 scoring='roc_auc'):
        def print_args():
            print('Current parameters:')
            print('Pre_processor:')
            pprint.pprint(self.pre_processor_args)
            print('Vectorizer:')
            pprint.pprint(self.vectorizer_args)
            print('Estimator:')
            pprint.pprint(self.estimator_args)
        if verbose:
            print('Parameters range:')
            print('Pre_processor:')
            pprint.pprint(pre_processor_parameters)
            print('Vectorizer:')
            pprint.pprint(vectorizer_parameters)
            print('Estimator:')
            pprint.pprint(estimator_parameters)
        best_pre_processor_args_ = dict()
        best_vectorizer_args_ = dict()
        best_estimator_args_ = dict()
        best_score_ = best_score_mean_ = best_score_std_ = 0
        start = time.time()
        mean_len_pre_processor_parameters = np.mean(
            [len(pre_processor_parameters[p]) for p in pre_processor_parameters])
        mean_len_vectorizer_parameters = np.mean(
            [len(vectorizer_parameters[p]) for p in vectorizer_parameters])
        for i in range(n_iter):
            try:
                self.estimator_args = self._sample(estimator_parameters)
                self.estimator.set_params(n_jobs=n_jobs, **self.estimator_args)
                # build data matrix only the first time or if needed e.g. because
                # there are more choices in the paramter settings for the
                # pre_processor or the vectorizer
                if i == 0 or mean_len_pre_processor_parameters != 1 or mean_len_vectorizer_parameters != 1:
                    # sample paramters randomly
                    self.pre_processor_args = self._sample(
                        pre_processor_parameters)
                    self.vectorizer_args = self._sample(vectorizer_parameters)
                    # copy the iterators for later re-use
                    iterable_pos, iterable_pos_ = tee(iterable_pos)
                    iterable_neg, iterable_neg_ = tee(iterable_neg)
                    X, y = self._self_training_data_matrices(iterable_pos_, iterable_neg_,
                                                             n_active_learning_iterations=n_active_learning_iterations,
                                                             size_positive=size_positive,
                                                             size_negative=size_negative,
                                                             lower_bound_threshold_positive=lower_bound_threshold_positive,
                                                             upper_bound_threshold_positive=upper_bound_threshold_positive,
                                                             lower_bound_threshold_negative=lower_bound_threshold_negative,
                                                             upper_bound_threshold_negative=upper_bound_threshold_negative,
                                                             n_jobs=n_jobs)
                scores = cross_validation.cross_val_score(
                    self.estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
            except Exception as e:
                if verbose:
                    print('Failed iteration: %d/%d (at %.1f sec; %s)' %
                          (i + 1, n_iter, time.time() - start, str(datetime.timedelta(seconds=(time.time() - start)))))
                    print e.__doc__
                    print e.message
                    print('Failed with the following setting:')
                    print_args()
                    print '...continuing'
            else:
                # consider as score the mean-std for a robust estimate
                score_mean = np.mean(scores)
                score_std = np.std(scores)
                score = score_mean - score_std
                # update the best confirguration
                if best_score_ < score:
                    best_score_ = score
                    best_score_mean_ = score_mean
                    best_score_std_ = score_std
                    best_pre_processor_args_ = self.pre_processor_args
                    best_vectorizer_args_ = self.vectorizer_args
                    best_estimator_args_ = self.estimator_args
                    if verbose:
                        print
                        print('Iteration: %d/%d (at %.1f sec; %s)' %
                          (i + 1, n_iter, time.time() - start, str(datetime.timedelta(seconds=(time.time() - start)))))
                        print('Best score (%s): %f (%f +- %f)' %
                              (scoring, best_score_, best_score_mean_, best_score_std_))
                        print_args()
                        print 'Instances: %d ; Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1],  X.getnnz() / X.shape[0])
                        print report_base_statistics(y)
        # store the best hyperparamter configuration
        self.pre_processor_args = best_pre_processor_args_
        self.vectorizer_args = best_vectorizer_args_
        self.estimator_args = best_estimator_args_
        # fit the estimator using the best hyperparameters
        self.fit(iterable_pos, iterable_neg, n_jobs=n_jobs)

    def _self_training_data_matrices(self, iterable_pos, iterable_neg,
                                     n_active_learning_iterations=2,
                                     size_positive=-1,
                                     size_negative=100,
                                     lower_bound_threshold_positive=-1,
                                     upper_bound_threshold_positive=1,
                                     lower_bound_threshold_negative=-1,
                                     upper_bound_threshold_negative=1,
                                     n_jobs=1):
        # select the initial ids simply as the first occurrences
        positive_ids = range(size_positive)
        negative_ids = range(size_negative)
        # iterate: select instances according to current model and create novel
        # data matrix to fit the model in next round
        for i in range(n_active_learning_iterations):
            # make data matrix on selected instances
            # if this is the first iteration or we need to select positives
            if i == 0 or size_positive != -1:
                iterable_pos, iterable_pos_, iterable_pos__ = tee(
                    iterable_pos, 3)
                if size_positive == -1:  # if we take all positives
                    Xpos = self._data_matrix(iterable_pos_, n_jobs=n_jobs)
                else:  # otherwise use selection
                    Xpos = self._data_matrix(
                        selection_iterator(iterable_pos_, positive_ids), n_jobs=n_jobs)
            # if this is the first iteration or we need to select negatives
            if i == 0 or size_negative != -1:
                iterable_neg, iterable_neg_, iterable_neg__ = tee(
                    iterable_neg, 3)
                if size_negative == -1:  # if we take all negatives
                    Xneg = self._data_matrix(iterable_neg_, n_jobs=n_jobs)
                else:  # otherwise use selection
                    Xneg = self._data_matrix(
                        selection_iterator(iterable_neg_, negative_ids), n_jobs=n_jobs)
            # assemble data matrix
            X, y = self._assemble_data_matrix(Xpos, Xneg)
            # stop the fitting procedure at the last-1 iteration and return X,y
            if i == n_active_learning_iterations - 1:
                break
            # fit the estimator on selected instances
            self.estimator.fit(X, y)
            # use the trained estimator to select the next instances
            if size_positive != -1:
                positive_ids = self._bounded_selection(
                    iterable_pos__, size=size_positive, lower_bound_threshold=lower_bound_threshold_positive, upper_bound_threshold=upper_bound_threshold_positive)
            if size_negative != -1:
                negative_ids = self._bounded_selection(
                    iterable_neg__, size=size_negative, lower_bound_threshold=lower_bound_threshold_negative, upper_bound_threshold=upper_bound_threshold_negative)
        return X, y

    def _bounded_selection(self, iterable, size=None, lower_bound_threshold=None, upper_bound_threshold=None):
        #transform row data to graphs
        iterable = self.pre_processor(iterable)
        #use estimator to predict margin for instances in an out-of-core fashion
        predictions = self.vectorizer.predict(iterable, self.estimator)
        ids = list()
        for i, prediction in enumerate(predictions):
            if prediction >= float(lower_bound_threshold) and prediction <= float(upper_bound_threshold):
                ids.append(i)
        if len(ids) == 0:
            raise Exception('No instances found that satisfy constraints')
        # keep a random sample of num_neg difficult cases
        random.shuffle(ids)
        ids = ids[:size]
        return ids
