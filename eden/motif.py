#!/usr/bin/env python

import sys
import os
import random
import re
from time import time, clock
import multiprocessing as mp
import numpy as np
from itertools import tee, chain
from collections import defaultdict

import joblib

from eden import apply_async
from eden.graph import Vectorizer
from eden.path import Vectorizer as PathVectorizer
from eden.util import vectorize, mp_pre_process, compute_intervals
from eden.converter.fasta import sequence_to_eden
from eden.modifier.seq import seq_to_seq, shuffle_modifier
from eden.util import fit
from eden.util.iterated_maximum_subarray import compute_max_subarrays

import esm

import logging
logger = logging.getLogger('root.%s' % (__name__))


class SequenceMotif(object):

    def __init__(self,
                 min_subarray_size=7,
                 max_subarray_size=10,
                 min_motif_count=1,
                 min_cluster_size=1,
                 training_size=None,
                 negative_ratio=2,
                 shuffle_order=2,
                 n_iter_search=1,
                 complexity=4,
                 nbits=20,
                 algorithm='DBSCAN',
                 n_clusters=4,
                 eps=0.3,
                 threshold=0.2,
                 branching_factor=50,
                 min_samples=3,
                 n_blocks=2,
                 block_size=None,
                 n_jobs=8,
                 random_state=1):
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.n_jobs = n_jobs
        self.training_size = training_size
        self.n_iter_search = n_iter_search
        self.complexity = complexity
        self.nbits = nbits
        # init vectorizer
        self.vectorizer = Vectorizer(complexity=self.complexity, nbits=self.nbits)
        self.seq_vectorizer = PathVectorizer(complexity=self.complexity, nbits=self.nbits)
        self.negative_ratio = negative_ratio
        self.shuffle_order = shuffle_order
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.eps = eps
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.min_samples = min_samples
        self.min_subarray_size = min_subarray_size
        self.max_subarray_size = max_subarray_size
        self.min_motif_count = min_motif_count
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        random.seed(random_state)

        self.motives_db = defaultdict(list)
        self.motives = []
        self.clusters = defaultdict(list)
        self.cluster_model = []

    def _serial_graph_motif(self, seqs, placeholder=None):
        # make graphs
        iterable = sequence_to_eden(seqs)
        # use node importance and 'position' attribute to identify max_subarrays of a specific size
        graphs = self.vectorizer.annotate(iterable, estimator=self.estimator)
        # use compute_max_subarrays to return an iterator over motives
        motives = []
        for graph in graphs:
            subarrays = compute_max_subarrays(graph=graph, min_subarray_size=self.min_subarray_size, max_subarray_size=self.max_subarray_size)
            for subarray in subarrays:
                motives.append(subarray['subarray_string'])
        return motives

    def _multiprocess_graph_motif(self, seqs):
        size = len(seqs)
        intervals = compute_intervals(size=size, n_blocks=self.n_blocks, block_size=self.block_size)
        if self.n_jobs == -1:
            pool = mp.Pool()
        else:
            pool = mp.Pool(processes=self.n_jobs)
        results = [apply_async(pool, self._serial_graph_motif, args=(seqs[start:end], True)) for start, end in intervals]
        output = [p.get() for p in results]
        return list(chain(*output))

    def _motif_finder(self, seqs):
        if self.n_jobs > 1 or self.n_jobs == -1:
            return self._multiprocess_graph_motif(seqs)
        else:
            return self._serial_graph_motif(seqs)

    def save(self, model_name):
        joblib.dump(self, model_name, compress=1)

    def load(self, obj):
        self.__dict__.update(joblib.load(obj).__dict__)

    def _fit_predictive_model(self, seqs):
        # duplicate iterator
        pos_seqs, pos_seqs_ = tee(seqs)
        pos_graphs = mp_pre_process(pos_seqs, pre_processor=sequence_to_eden, n_blocks=self.n_blocks, n_jobs=self.n_jobs)
        # shuffle seqs to obtain negatives
        neg_seqs = seq_to_seq(pos_seqs_, modifier=shuffle_modifier, times=self.negative_ratio, order=self.shuffle_order)
        neg_graphs = mp_pre_process(neg_seqs, pre_processor=sequence_to_eden, n_blocks=self.n_blocks, n_jobs=self.n_jobs)
        # fit discriminative estimator
        self.estimator = fit(pos_graphs, neg_graphs,
                             vectorizer=self.vectorizer,
                             n_iter_search=self.n_iter_search,
                             n_blocks=self.n_blocks,
                             block_size=self.block_size,
                             n_jobs=self.n_jobs,
                             random_state=self.random_state)

    def _cluster(self, seqs):
        X = vectorize(seqs, vectorizer=self.seq_vectorizer, n_blocks=self.n_blocks, n_jobs=self.n_jobs)
        if self.algorithm == 'MiniBatchKMeans':
            from sklearn.cluster import MiniBatchKMeans
            clustering_algorithm = MiniBatchKMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        elif self.algorithm == 'DBSCAN':
            from sklearn.cluster import DBSCAN
            clustering_algorithm = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        elif self.algorithm == 'Birch':
            from sklearn.cluster import Birch
            clustering_algorithm = Birch(threshold=self.threshold, n_clusters=self.n_clusters, branching_factor=self.branching_factor)
        else:
            raise Exception('Unknown algorithm: %s' % self.algorithm)
        predictions = clustering_algorithm.fit_predict(X)
        # collect instance ids per cluster id
        for i in range(len(predictions)):
            self.clusters[predictions[i]] += [i]

    def _filter(self):
        # transform self.clusters that contains only the ids of the motives to
        # clustered_motives that contains the actual sequences
        new_sequential_cluster_id = -1
        clustered_motives = defaultdict(list)
        for cluster_id in self.clusters:
            if cluster_id != -1:
                if len(self.clusters[cluster_id]) >= self.min_cluster_size:
                    clustered_seqs = []
                    new_sequential_cluster_id += 1
                    for motif_id in self.clusters[cluster_id]:
                        clustered_motives[new_sequential_cluster_id].append(self.motives[motif_id])
        motives_db = defaultdict(list)
        # extract motif count within a cluster
        for cluster_id in clustered_motives:
            # consider only non identical motives
            motif_set = set(clustered_motives[cluster_id])
            for motif_i in motif_set:
                # count occurrences of each motif in cluster
                count = 0
                for motif_j in clustered_motives[cluster_id]:
                    if motif_i == motif_j:
                        count += 1
                # create dict with motives and their counts
                # if counts are above a threshold
                if count >= self.min_motif_count:
                    motives_db[cluster_id].append((count, motif_i))
        # transform cluster ids to incremental ids
        incremental_id = 0
        for cluster_id in motives_db:
            if len(motives_db[cluster_id]) >= self.min_cluster_size:
                self.motives_db[incremental_id] = motives_db[cluster_id]
                incremental_id += 1

    def fit(self, seqs):
        start = time()
        if self.training_size is None:
            training_seqs = seqs
        else:
            training_seqs = random.sample(seqs, self.training_size)
        self._fit_predictive_model(training_seqs)
        end = time()
        logger.info('model induction: %d positive instances %d secs' % (len(training_seqs), (end - start)))

        start = time()
        self.motives = self._motif_finder(seqs)
        end = time()
        logger.info('motives extraction: %d motives %d secs' % (len(self.motives), end - start))

        start = time()
        self._cluster(self.motives)
        end = time()
        logger.info('motives clustering: %d clusters %d secs' % (len(self.clusters), end - start))

        start = time()
        self._filter()
        end = time()
        n_motives = sum(len(self.motives_db[cid]) for cid in self.motives_db)
        n_clusters = len(self.motives_db)
        logger.info('after filtering: %d motives %d clusters %d secs' % (n_motives, n_clusters, (end - start)))

        start = time()
        # create models
        for cluster_id in self.motives_db:
            motives = [motif for count, motif in self.motives_db[cluster_id]]
            index = esm.Index()
            for motif in motives:
                index.enter(motif)
            index.fix()
            self.cluster_model = index
        end = time()
        logger.info('motif model construction: %d secs' % (end - start))

    def _build_regex(self, seqs):
        regex = ''
        for m in seqs:
            regex += '|' + m
        regex = regex[1:]
        return regex

    def _cluster_hit(self, seq, cluster_id):
        motives = [motif for count, motif in self.motives_db[cluster_id]]
        pattern = self._build_regex(motives)
        for m in re.finditer(pattern, seq):
            if m:
                yield (m.start(), m.end())
            else:
                yield None

    def predict(self, seqs, return_list=False):
        # returns for each instance a list with the cluster ids that have a hit
        # if  return_list=False then just return 1 if there is at least one hit from one cluster
        for header, seq in seqs:
            cluster_hits = []
            for cluster_id in self.motives_db:
                hits = self._cluster_hit(seq, cluster_id)
                if len(list(hits)):
                    cluster_hits.append(cluster_id)
            if return_list == False:
                if len(cluster_hits):
                    yield 1
                else:
                    yield 0
            else:
                yield cluster_hits

    def transform(self, seqs, return_match=False):
        # transform = transform an instance to a dense vector with features as cluster ID and entries 0/1 if a motif is found,
        # if 'return_match' argument is True, then write a pair with (start position,end position)  in the entry instead of 0/1
        num = len(self.motives_db)
        for header, seq in seqs:
            cluster_hits = [0] * num
            for cluster_id in self.motives_db:
                hits = self._cluster_hit(seq, cluster_id)
                hits = list(hits)
                if return_match == False:
                    if len(hits):
                        cluster_hits[cluster_id] = 1
                else:
                    cluster_hits[cluster_id] = hits
            yield cluster_hits
