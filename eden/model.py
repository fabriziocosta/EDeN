#!/usr/bin/env python

import numpy as np
from scipy.sparse import vstack
from sklearn import cross_validation
import random
import time
import datetime
import joblib
import pprint
import copy
from collections import defaultdict
from sklearn.linear_model import SGDClassifier
from itertools import tee, izip
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from eden.util import fit_estimator, selection_iterator, is_iterable, report_base_statistics
from eden.util import vectorize, mp_pre_process
from eden.util import serialize_dict
from eden.graph import Vectorizer

import logging
logger = logging.getLogger(__name__)


class ActiveLearningBinaryClassificationModel(object):

    def __init__(self, pre_processor=None,
                 vectorizer=Vectorizer(complexity=1),
                 estimator=SGDClassifier(class_weight='auto', shuffle=True),
                 fit_vectorizer=False,
                 n_jobs=8,
                 n_blocks=8,
                 block_size=None,
                 pre_processor_n_jobs=1,
                 pre_processor_n_blocks=8,
                 pre_processor_block_size=None,
                 description=None,
                 random_state=1):
        self.pre_processor = copy.deepcopy(pre_processor)
        self.vectorizer = copy.deepcopy(vectorizer)
        self.estimator = copy.deepcopy(estimator)
        self.pre_processor_args = None
        self.vectorizer_args = None
        self.estimator_args = None
        self.description = description
        self.fit_vectorizer = fit_vectorizer
        self.n_jobs = n_jobs
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.pre_processor_n_jobs = pre_processor_n_jobs
        self.pre_processor_n_blocks = pre_processor_n_blocks
        self.pre_processor_block_size = pre_processor_block_size
        random.seed(random_state)

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

    def _data_matrix(self, iterable, fit_vectorizer=False):
        assert(is_iterable(iterable)), 'Not iterable'
        graphs = mp_pre_process(iterable,
                                pre_processor=self.pre_processor,
                                pre_processor_args=self.pre_processor_args,
                                n_blocks=self.pre_processor_n_blocks,
                                block_size=self.pre_processor_block_size,
                                n_jobs=self.pre_processor_n_jobs)
        graphs, graphs_ = tee(graphs)
        self.vectorizer.set_params(**self.vectorizer_args)
        if fit_vectorizer:
            self.vectorizer.fit(graphs_)
        X = vectorize(graphs, vectorizer=self.vectorizer, n_jobs=self.n_jobs, n_blocks=self.n_blocks)
        return X

    def _data_matrices(self, iterable_pos, iterable_neg, fit_vectorizer=False):
        Xpos = self._data_matrix(iterable_pos, fit_vectorizer=fit_vectorizer)
        Xneg = self._data_matrix(iterable_neg, fit_vectorizer=False)
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

    def fit(self, iterable_pos, iterable_neg):
        self.estimator.set_params(**self.estimator_args)
        X, y = self._data_matrices(iterable_pos, iterable_neg, fit_vectorizer=self.fit_vectorizer)
        self.estimator.fit(X, y)

    def predict(self, iterable):
        X = self._data_matrix(iterable)
        return self.estimator.predict(X)

    def decision_function(self, iterable):
        X = self._data_matrix(iterable)
        return self.estimator.decision_function(X)

    def get_info(self, iterable, key='id'):
        iterable_graph = self.pre_processor(iterable, **self.pre_processor_args)
        for graph in iterable_graph:
            yield graph.graph.get(key, 'N/A')

    def info(self, iterable, key='id'):
        iterable, iterable_ = tee(iterable)
        X = self._data_matrix(iterable)
        info_iterable = self.get_info(iterable_, key=key)
        for margin, graph_info in izip(self.estimator.decision_function(X), info_iterable):
            yield margin, graph_info

    def estimate(self, iterable_pos, iterable_neg):
        X, y = self._data_matrices(iterable_pos, iterable_neg, fit_vectorizer=False)
        margins = self.estimator.decision_function(X)
        predictions = self.estimator.predict(X)
        apr = average_precision_score(y, margins)
        roc = roc_auc_score(y, margins)

        #output results
        text = []
        text.append('\nClassifier:')
        text.append('%s' % self.estimator)
        text.append('\nData:')
        text.append('Instances: %d ; Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1],  X.getnnz() / X.shape[0]))
        text.append('\nPredictive performace estimate:')
        text.append('%s' % classification_report(y, predictions))
        text.append('APR: %.3f' % apr)
        text.append('ROC: %.3f' % roc)
        logger.info('\n'.join(text))
        return apr, roc

    def get_parameters(self):
        text = []
        text.append('\n\tModel parameters:')
        text.append('\nPre_processor:')
        text.append(serialize_dict(self.pre_processor_args))
        text.append('\nVectorizer:')
        text.append(serialize_dict(self.vectorizer_args))
        text.append('\nEstimator:')
        text.append(serialize_dict(self.estimator_args))
        return '\n'.join(text)

    def optimize(self, iterable_pos, iterable_neg,
                 model_name='model',
                 n_active_learning_iterations=0,
                 size_positive=-1,
                 size_negative=-1,
                 lower_bound_threshold_positive=-1,
                 upper_bound_threshold_positive=1,
                 lower_bound_threshold_negative=-1,
                 upper_bound_threshold_negative=1,
                 n_iter=20,
                 max_total_time=-1,
                 pre_processor_parameters=dict(),
                 vectorizer_parameters=dict(),
                 estimator_parameters=dict(),
                 cv=10,
                 scoring='roc_auc',
                 score_func=lambda u, s: u - s,
                 two_steps_optimization=True):
    
        def _get_parameters_range():
            text = []
            text.append('\n\n\tParameters range:')
            text.append('\nPre_processor:')
            text.append(serialize_dict(pre_processor_parameters))
            text.append('\nVectorizer:')
            text.append(serialize_dict(vectorizer_parameters))
            text.append('\nEstimator:')
            text.append(serialize_dict(estimator_parameters))
            return '\n'.join(text)

        logger.debug(_get_parameters_range())
        # init
        best_pre_processor_ = None
        best_vectorizer_ = None
        best_estimator_ = None
        best_pre_processor_args_ = dict()
        best_vectorizer_args_ = dict()
        best_estimator_args_ = dict()
        best_pre_processor_parameters_ = defaultdict(list)
        best_vectorizer_parameters_ = defaultdict(list)
        best_estimator_parameters_ = defaultdict(list)
        best_score_ = best_score_mean_ = best_score_std_ = 0
        start = time.time()
        if len(pre_processor_parameters) == 0:
            mean_len_pre_processor_parameters = 0
        else:
            mean_len_pre_processor_parameters = np.mean([len(pre_processor_parameters[p]) for p in pre_processor_parameters])
        if len(vectorizer_parameters) == 0:
            mean_len_vectorizer_parameters = 0
        else:
            mean_len_vectorizer_parameters = np.mean([len(vectorizer_parameters[p]) for p in vectorizer_parameters])
        if (mean_len_pre_processor_parameters == 1 or mean_len_pre_processor_parameters == 0) and (mean_len_vectorizer_parameters == 1 or mean_len_vectorizer_parameters == 0):
            data_matrix_is_stable = True
        else:
            data_matrix_is_stable = False
        # main iteration
        for i in range(n_iter):
            if max_total_time != -1:
                if time.time() - start > max_total_time:
                    logger.warning('Reached max time: %s' % (str(datetime.timedelta(seconds=(time.time() - start)))))
                    break
            try:
                # after n_iter/2 iterations, replace the parameter lists with only those values that have been found to increase the performance
                if i == int(n_iter / 2) and two_steps_optimization == True:
                    if len(best_pre_processor_parameters_) > 0:
                        pre_processor_parameters = dict(best_pre_processor_parameters_)
                    if len(best_vectorizer_parameters_) > 0:
                        vectorizer_parameters = dict(best_vectorizer_parameters_)
                    if len(best_estimator_parameters_) > 0:
                        estimator_parameters = dict(best_estimator_parameters_)
                    logger.debug(_get_parameters_range())

                self.estimator_args = self._sample(estimator_parameters)
                self.estimator.set_params(**self.estimator_args)
                # build data matrix only the first time or if needed e.g. because
                # there are more choices in the paramter settings for the
                # pre_processor or the vectorizer
                if i == 0 or data_matrix_is_stable == False:
                    # sample paramters randomly
                    self.pre_processor_args = self._sample(pre_processor_parameters)
                    self.vectorizer_args = self._sample(vectorizer_parameters)
                    # copy the iterators for later re-use
                    iterable_pos, iterable_pos_ = tee(iterable_pos)
                    iterable_neg, iterable_neg_ = tee(iterable_neg)
                    if n_active_learning_iterations == 0:  # if no active learning mode, just produce data matrix
                        X, y = self._data_matrices(iterable_pos_, iterable_neg_, fit_vectorizer=self.fit_vectorizer)
                    else:  # otherwise use the active learning strategy
                        X, y = self._active_learning_data_matrices(iterable_pos_, iterable_neg_,
                                                                   n_active_learning_iterations=n_active_learning_iterations,
                                                                   size_positive=size_positive,
                                                                   size_negative=size_negative,
                                                                   lower_bound_threshold_positive=lower_bound_threshold_positive,
                                                                   upper_bound_threshold_positive=upper_bound_threshold_positive,
                                                                   lower_bound_threshold_negative=lower_bound_threshold_negative,
                                                                   upper_bound_threshold_negative=upper_bound_threshold_negative)
                scores = cross_validation.cross_val_score(self.estimator, X, y, cv=cv, scoring=scoring, n_jobs=self.n_jobs)
            except Exception as e:
                text = []
                text.append('\nFailed iteration: %d/%d (at %.1f sec; %s)' %
                            (i + 1, n_iter, time.time() - start, str(datetime.timedelta(seconds=(time.time() - start)))))
                text.append(e.__doc__)
                text.append(e.message)
                text.append('Failed with the following setting:')
                text.append(self.get_parameters())
                text.append('...continuing')
                logger.debug('\n'.join(text))
            else:
                # consider as score the mean-std for a robust estimate of predictive performance
                score_mean = np.mean(scores)
                score_std = np.std(scores)
                score = score_func(score_mean, score_std)
                logger.debug('iteration: %d/%d score (%s): %.3f (%.3f +- %.3f)'%(i+1, n_iter, scoring, score, score_mean, score_std))
                # update the best confirguration
                if best_score_ < score:
                    # fit the estimator since the cross_validation estimate does not set the estimator parametrs
                    self.estimator.fit(X, y)
                    self.save(model_name)
                    best_score_ = score
                    best_score_mean_ = score_mean
                    best_score_std_ = score_std
                    best_pre_processor_ = copy.deepcopy(self.pre_processor)
                    best_vectorizer_ = copy.deepcopy(self.vectorizer)
                    best_estimator_ = copy.deepcopy(self.estimator)
                    best_pre_processor_args_ = copy.deepcopy(self.pre_processor_args)
                    best_vectorizer_args_ = copy.deepcopy(self.vectorizer_args)
                    best_estimator_args_ = copy.deepcopy(self.estimator_args)
                    if i > 0:  # if we improve over the very first iteration, then...
                        # add parameter to list of best parameters
                        for key in self.pre_processor_args:
                            best_pre_processor_parameters_[key].append(self.pre_processor_args[key])
                        for key in self.vectorizer_args:
                            best_vectorizer_parameters_[key].append(self.vectorizer_args[key])
                        for key in self.estimator_args:
                            best_estimator_parameters_[key].append(self.estimator_args[key])
                    text = []
                    text.append('\n\n\tIteration: %d/%d (after %.1f sec; %s)' %
                                (i + 1, n_iter, time.time() - start, str(datetime.timedelta(seconds=(time.time() - start)))))
                    text.append('Best score (%s): %.3f (%.3f +- %.3f)' % (scoring, best_score_, best_score_mean_, best_score_std_))
                    text.append('\nData:')
                    text.append('Instances: %d ; Features: %d with an avg of %d features per instance' %
                                (X.shape[0], X.shape[1],  X.getnnz() / X.shape[0]))
                    text.append(report_base_statistics(y))
                    text.append(self.get_parameters())
                    logger.info('\n'.join(text))
        # store the best hyperparamter configuration
        self.pre_processor_args = copy.deepcopy(best_pre_processor_args_)
        self.vectorizer_args = copy.deepcopy(best_vectorizer_args_)
        self.estimator_args = copy.deepcopy(best_estimator_args_)
        # store the best machines
        self.pre_processor = copy.deepcopy(best_pre_processor_)
        self.vectorizer = copy.deepcopy(best_vectorizer_)
        self.estimator = copy.deepcopy(best_estimator_)
        # save to disk
        logger.info('Saved current best model in %s' % model_name)
        self.save(model_name)

    def _active_learning_data_matrices(self, iterable_pos, iterable_neg,
                                       n_active_learning_iterations=2,
                                       size_positive=-1,
                                       size_negative=100,
                                       lower_bound_threshold_positive=-1,
                                       upper_bound_threshold_positive=1,
                                       lower_bound_threshold_negative=-1,
                                       upper_bound_threshold_negative=1):
        # select the initial ids simply as the first occurrences
        if size_positive != -1:
            positive_ids = range(size_positive)
        if size_negative != -1:
            negative_ids = range(size_negative)
        # iterate: select instances according to current model and create novel data matrix to fit the model in next round
        for i in range(n_active_learning_iterations):
            # make data matrix on selected instances
            # if this is the first iteration or we need to select positives
            if i == 0 or size_positive != -1:
                iterable_pos, iterable_pos_, iterable_pos__ = tee(iterable_pos, 3)
                if size_positive == -1:  # if we take all positives
                    Xpos = self._data_matrix(iterable_pos_, fit_vectorizer=self.fit_vectorizer)
                else:  # otherwise use selection
                    Xpos = self._data_matrix(selection_iterator(iterable_pos_, positive_ids), fit_vectorizer=self.fit_vectorizer)
            # if this is the first iteration or we need to select negatives
            if i == 0 or size_negative != -1:
                iterable_neg, iterable_neg_, iterable_neg__ = tee(iterable_neg, 3)
                if size_negative == -1:  # if we take all negatives
                    Xneg = self._data_matrix(iterable_neg_, fit_vectorizer=False)
                else:  # otherwise use selection
                    Xneg = self._data_matrix(selection_iterator(iterable_neg_, negative_ids), fit_vectorizer=False)
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
        # transform row data to graphs
        iterable = self.pre_processor(iterable)
        # use estimator to predict margin for instances in an out-of-core fashion
        predictions = self.vectorizer.predict(iterable, self.estimator)
        ids = list()
        for i, prediction in enumerate(predictions):
            if prediction >= float(lower_bound_threshold) and prediction <= float(upper_bound_threshold):
                ids.append(i)
        if len(ids) == 0:
            raise Exception('No instances found that satisfy constraints')
        # keep a random sample of interesting instances
        random.shuffle(ids)
        ids = ids[:size]
        return ids
