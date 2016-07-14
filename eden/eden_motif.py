#!/usr/bin/env python

import logging
from eden.util import configure_logging
import multiprocessing as mp
from collections import defaultdict
from eden import apply_async
import numpy as np
from scipy.sparse import vstack
from eden.util.iterated_maximum_subarray import compute_max_subarrays_sequence
from itertools import izip
import time
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import MiniBatchKMeans
from eden.sequence import Vectorizer

logger = logging.getLogger(__name__)
configure_logging(logger, verbosity=2)


def serial_pre_process(iterable, vectorizer=None):
    data_matrix = vectorizer.transform(iterable)
    return data_matrix


def chunks(iterable, n):
    iterable = iter(iterable)
    while True:
        items = []
        for i in range(n):
            it = iterable.next()
            items.append(it)
        yield items


def multiprocess_vectorize(pos_iterable,
                           vectorizer=None,
                           pos_block_size=100,
                           n_jobs=-1):
    start_time = time.time()
    if n_jobs == -1:
        pool = mp.Pool()
    else:
        pool = mp.Pool(n_jobs)

    pos_results = [apply_async(
        pool, serial_pre_process,
        args=(seqs, vectorizer))
        for seqs in chunks(pos_iterable, pos_block_size)]
    logging.debug('Setup %.2f secs' % (time.time() - start_time))

    start_time = time.time()
    matrices = []
    for i, p in enumerate(pos_results):
        loc_start_time = time.time()
        pos_data_matrix = p.get()
        matrices += pos_data_matrix
        logging.debug('%d %s (%.2f secs) (delta: %.2f)' % (i, pos_data_matrix.shape, time.time() - start_time, time.time() - loc_start_time))

    pool.close()
    pool.join()
    data_matrix = vstack(matrices)
    return data_matrix


def multiprocess_fit(pos_iterable, neg_iterable,
                     vectorizer=None,
                     estimator=None,
                     pos_block_size=100,
                     neg_block_size=100,
                     n_jobs=-1):
    start_time = time.time()
    classes = np.array([1, -1])
    if n_jobs == -1:
        pool = mp.Pool()
    else:
        pool = mp.Pool(n_jobs)

    pos_results = [apply_async(
        pool, serial_pre_process,
        args=(seqs, vectorizer))
        for seqs in chunks(pos_iterable, pos_block_size)]
    neg_results = [apply_async(
        pool, serial_pre_process,
        args=(seqs, vectorizer))
        for seqs in chunks(neg_iterable, neg_block_size)]
    logging.debug('Setup %.2f secs' % (time.time() - start_time))

    start_time = time.time()
    for i, (p, n) in enumerate(izip(pos_results, neg_results)):
        loc_start_time = time.time()
        pos_data_matrix = p.get()
        y = [1] * pos_data_matrix.shape[0]
        neg_data_matrix = n.get()
        y += [-1] * neg_data_matrix.shape[0]
        y = np.array(y)
        data_matrix = vstack([pos_data_matrix, neg_data_matrix])
        estimator.partial_fit(data_matrix, y, classes=classes)
        logging.debug('%d %s (%.2f secs) (delta: %.2f)' % (i, data_matrix.shape, time.time() - start_time, time.time() - loc_start_time))

    pool.close()
    pool.join()

    return estimator


def serial_subarray(iterable, vectorizer=None, estimator=None, min_subarray_size=5, max_subarray_size=10):
    annotated_seqs = vectorizer.annotate(iterable, estimator=estimator)
    subarrays_items = []
    for seq, score in annotated_seqs:
        subarrays = compute_max_subarrays_sequence(seq=seq, score=score,
                                                   min_subarray_size=min_subarray_size,
                                                   max_subarray_size=max_subarray_size)
        subseqs = [subarray['subarray_string'] for subarray in subarrays]
        subarrays_items += subseqs
    return subarrays_items


def multiprocess_subarray(pos_iterable,
                          vectorizer=None,
                          estimator=None,
                          min_subarray_size=5,
                          max_subarray_size=10,
                          pos_block_size=100,
                          n_jobs=-1):
    start_time = time.time()
    if n_jobs == -1:
        pool = mp.Pool()
    else:
        pool = mp.Pool(n_jobs)

    pos_results = [apply_async(
        pool, serial_subarray,
        args=(seqs, vectorizer, estimator, min_subarray_size, max_subarray_size))
        for seqs in chunks(pos_iterable, pos_block_size)]
    logging.debug('Setup %.2f secs' % (time.time() - start_time))
    start_time = time.time()
    subarrays_items = []
    for i, p in enumerate(pos_results):
        loc_start_time = time.time()
        subarrays_item = p.get()
        subarrays_items += subarrays_item
        logging.debug('%d (%.2f secs) (delta: %.2f)' % (i, time.time() - start_time, time.time() - loc_start_time))

    pool.close()
    pool.join()
    return subarrays_items

# ------------------------------------------------------------------------------


class SequenceMotifDecomposer(BaseEstimator, ClassifierMixin):
    """SequenceMotifDecomposer."""

    def __init__(self,
                 complexity=None,
                 n_clusters=10,
                 min_subarray_size=5,
                 max_subarray_size=10,
                 estimator=SGDClassifier(warm_start=True),
                 clusterer=MiniBatchKMeans(),
                 pos_block_size=300,
                 neg_block_size=300,
                 n_jobs=-1):
        """Construct."""
        self.complexity = complexity
        self.n_clusters = n_clusters
        self.min_subarray_size = min_subarray_size
        self.max_subarray_size = max_subarray_size
        self.pos_block_size = pos_block_size
        self.neg_block_size = neg_block_size
        self.n_jobs = n_jobs
        self.vectorizer = Vectorizer(complexity=complexity, auto_weights=True, nbits=15)
        self.estimator = estimator
        self.clusterer = clusterer

    def fit(self, pos_seqs=None, neg_seqs=None):
        """fit."""
        try:
            self.model = multiprocess_fit(pos_seqs, neg_seqs,
                                          vectorizer=self.vectorizer,
                                          estimator=self.estimator,
                                          pos_block_size=self.pos_block_size,
                                          neg_block_size=self.neg_block_size,
                                          n_jobs=self.n_jobs)
            return self
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def predict(self, pos_seqs=None):
        """predict."""
        try:
            subarrays_items = multiprocess_subarray(
                pos_seqs,
                vectorizer=self.vectorizer,
                estimator=self.model,
                min_subarray_size=self.min_subarray_size,
                max_subarray_size=self.max_subarray_size,
                pos_block_size=self.pos_block_size,
                n_jobs=self.n_jobs)

            data_matrix = multiprocess_vectorize(
                subarrays_items,
                vectorizer=self.vectorizer,
                pos_block_size=self.pos_block_size,
                n_jobs=self.n_jobs)
            self.clusterer.set_params(n_clusters=self.n_clusters)
            preds = self.clusterer.fit_predict(data_matrix)

            self.clusters = defaultdict(list)
            for pred, seq in zip(preds, subarrays_items):
                self.clusters[pred].append(seq)

            return self.clusters
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)
