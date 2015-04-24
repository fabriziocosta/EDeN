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

import argparse
import logging
import logging.handlers
from eden.util import setup
from eden.util import save_output

from eden.graph import Vectorizer
from eden.path import Vectorizer as PathVectorizer
from eden import vectorize
from eden.util import mp_pre_process
from eden.converter.fasta import sequence_to_eden
from eden.modifier.seq import seq_to_seq, shuffle_modifier
from eden.util import fit
from eden.util.iterated_maximum_subarray import compute_max_subarrays


def serial_graph_motif(seqs, vectorizer=None, estimator=None, min_subarray_size=5, max_subarray_size=18):
    # make graphs
    iterable = sequence_to_eden(seqs)
    # use node importance and 'position' attribute to identify max_subarrays of a specific size
    graphs = vectorizer.annotate(iterable, estimator=estimator)
    # use compute_max_subarrays to return an iterator over motives
    motives = []
    for graph in graphs:
        subarrays = compute_max_subarrays(graph=graph, min_subarray_size=min_subarray_size, max_subarray_size=max_subarray_size)
        for subarray in subarrays:
            motives.append(subarray['subarray_string'])
    return motives


def multiprocess_graph_motif(seqs, vectorizer=None, estimator=None, min_subarray_size=5, max_subarray_size=18, n_blocks=5, n_jobs=8):
    size = len(seqs)
    block_size = size / n_blocks
    if n_jobs == -1:
        pool = mp.Pool()
    else:
        pool = mp.Pool(processes=n_jobs)
    results = [pool.apply_async(serial_graph_motif, args=(
        seqs[s * block_size:(s + 1) * block_size], vectorizer, estimator, min_subarray_size, max_subarray_size)) for s in range(n_blocks - 1)]
    output = [p.get() for p in results]
    return list(chain(*output))


def motif_finder(seqs, vectorizer=None, estimator=None, min_subarray_size=5, max_subarray_size=18, n_blocks=5, n_jobs=8):
    if n_jobs > 1 or n_jobs == -1:
        return multiprocess_graph_motif(seqs, vectorizer=vectorizer, estimator=estimator, min_subarray_size=min_subarray_size, max_subarray_size=max_subarray_size, n_blocks=n_blocks, n_jobs=n_jobs)
    else:
        return serial_graph_motif(seqs, vectorizer=vectorizer, estimator=estimator, min_subarray_size=min_subarray_size, max_subarray_size=max_subarray_size)


class SequenceMotif(object):

    def __init__(self,
                 min_subarray_size=7,
                 max_subarray_size=10,
                 min_motif_count=1,
                 min_cluster_size=1,
                 training_size=None,
                 negative_ratio=2,
                 n_iter_search=1,
                 complexity=4,
                 nbits=20,
                 algorithm='DBSCAN',
                 n_clusters=4,
                 eps=0.3,
                 threshold=0.2,
                 branching_factor=50,
                 min_samples=3,
                 verbosity=0,
                 n_blocks=2,
                 n_jobs=8,
                 random_seed=1):
        self.n_blocks = n_blocks
        self.n_jobs = n_jobs
        self.training_size = training_size
        self.n_iter_search = n_iter_search
        self.complexity = complexity
        self.nbits = nbits
        # init vectorizer
        self.vectorizer = Vectorizer(complexity=self.complexity, nbits=self.nbits)
        self.seq_vectorizer = PathVectorizer(complexity=self.complexity, nbits=self.nbits)
        self.negative_ratio = negative_ratio
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
        self.verbosity = verbosity
        random.seed(random_seed)

        self.motives_db = defaultdict(list)
        self.motives = []
        self.clusters = defaultdict(list)

    def fit_predictive_model(self, seqs):
        # duplicate iterator
        pos_seqs, pos_seqs_ = tee(seqs)
        pos_graphs = mp_pre_process(pos_seqs, pre_processor=sequence_to_eden, n_blocks=self.n_blocks, n_jobs=self.n_jobs)
        # shuffle seqs to obtain negatives
        neg_seqs = seq_to_seq(pos_seqs_, modifier=shuffle_modifier, times=self.negative_ratio, order=2)
        neg_graphs = mp_pre_process(neg_seqs, pre_processor=sequence_to_eden, n_blocks=self.n_blocks, n_jobs=self.n_jobs)
        # fit discriminative estimator
        self.estimator = fit(pos_graphs, neg_graphs, vectorizer=self.vectorizer,
                             n_iter_search=self.n_iter_search, n_blocks=self.n_blocks, n_jobs=self.n_jobs)

    def cluster(self, seqs):
        X = vectorize(seqs, vectorizer=self.seq_vectorizer, n_blocks=self.n_blocks, n_jobs=self.n_jobs)
        if self.algorithm == 'MiniBatchKMeans':
            from sklearn.cluster import MiniBatchKMeans
            clustering_algorithm = MiniBatchKMeans(n_clusters=self.n_clusters)
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

    def filter(self):
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
            if len(motives_db[cluster_id]) > 0:
                self.motives_db[incremental_id] = motives_db[cluster_id]
                incremental_id += 1

    def fit(self, seqs):
        start = time()
        if self.training_size is None:
            training_seqs = seqs
        else:
            training_seqs = random.sample(seqs, self.training_size)
        self.fit_predictive_model(training_seqs)
        end = time()
        if self.verbosity > 0:
            print('model induction: %d secs' % (end - start))

        start = time()
        self.motives = motif_finder(seqs,
                                    vectorizer=self.vectorizer,
                                    estimator=self.estimator,
                                    min_subarray_size=self.min_subarray_size,
                                    max_subarray_size=self.max_subarray_size,
                                    n_jobs=self.n_jobs)
        end = time()
        if self.verbosity > 0:
            print('motives extraction: %d motives %d secs' % (len(self.motives), end - start))

        start = time()
        self.cluster(self.motives)
        self.filter()
        end = time()
        if self.verbosity > 0:
            print('motives clustering: %d clusters %d secs' % (len(self.clusters), end - start))

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
