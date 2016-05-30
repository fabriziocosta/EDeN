#!/usr/bin/env python

import random
from time import time
import multiprocessing as mp
from itertools import tee, chain
from collections import defaultdict

import joblib

from eden import apply_async
from eden.graph import Vectorizer
from eden.sequence import Vectorizer as SeqVectorizer
from eden.util import vectorize, mp_pre_process, compute_intervals
from eden.converter.fasta import sequence_to_eden
from eden.modifier.seq import seq_to_seq, shuffle_modifier
from eden.util import fit
from eden.util.iterated_maximum_subarray import compute_max_subarrays
from eden.util.iterated_maximum_subarray import extract_sequence_and_score

import esm

import logging
logger = logging.getLogger(__name__)


class SequenceMotif(object):

    def __init__(self,
                 min_subarray_size=7,
                 max_subarray_size=10,
                 min_motif_count=1,
                 min_cluster_size=1,
                 training_size=None,
                 negative_ratio=1,
                 shuffle_order=2,
                 n_iter_search=1,
                 complexity=4,
                 radius=None,
                 distance=None,
                 nbits=20,
                 clustering_algorithm=None,
                 n_jobs=4,
                 n_blocks=8,
                 block_size=None,
                 pre_processor_n_jobs=4,
                 pre_processor_n_blocks=8,
                 pre_processor_block_size=None,
                 random_state=1):
        self.n_jobs = n_jobs
        self.n_blocks = n_blocks
        self.block_size = block_size
        self.pre_processor_n_jobs = pre_processor_n_jobs
        self.pre_processor_n_blocks = pre_processor_n_blocks
        self.pre_processor_block_size = pre_processor_block_size
        self.training_size = training_size
        self.n_iter_search = n_iter_search
        self.complexity = complexity
        self.nbits = nbits
        # init vectorizer
        self.vectorizer = Vectorizer(complexity=self.complexity,
                                     r=radius, d=distance,
                                     nbits=self.nbits)
        self.seq_vectorizer = SeqVectorizer(complexity=self.complexity,
                                            r=radius, d=distance,
                                            nbits=self.nbits)
        self.negative_ratio = negative_ratio
        self.shuffle_order = shuffle_order
        self.clustering_algorithm = clustering_algorithm
        self.min_subarray_size = min_subarray_size
        self.max_subarray_size = max_subarray_size
        self.min_motif_count = min_motif_count
        self.min_cluster_size = min_cluster_size
        self.random_state = random_state
        random.seed(random_state)

        self.motives_db = defaultdict(list)
        self.motives = []
        self.clusters = defaultdict(list)
        self.cluster_models = []
        self.importances = []

    def save(self, model_name):
        self.clustering_algorithm = None  # NOTE: some algorithms cannot be pickled
        joblib.dump(self, model_name, compress=1)

    def load(self, obj):
        self.__dict__.update(joblib.load(obj).__dict__)
        self._build_cluster_models()

    def fit(self, seqs, neg_seqs=None):
        """
        Builds a discriminative estimator.
        Identifies the maximal subarrays in the data.
        Clusters them with the clustering algorithm provided in the initialization phase.
        For each cluster builds a fast sequence search model (Aho Corasick data structure).
        """
        start = time()
        if self.training_size is None:
            training_seqs = seqs
        else:
            training_seqs = random.sample(seqs, self.training_size)
        self._fit_predictive_model(training_seqs, neg_seqs=neg_seqs)
        end = time()
        logger.info('model induction: %d positive instances %d s' % (len(training_seqs), (end - start)))

        start = time()
        self.motives = self._motif_finder(seqs)
        end = time()
        logger.info('motives extraction: %d motives in %ds' % (len(self.motives), end - start))

        start = time()
        self._cluster(self.motives, clustering_algorithm=self.clustering_algorithm)
        end = time()
        logger.info('motives clustering: %d clusters in %ds' % (len(self.clusters), end - start))

        start = time()
        self._filter()
        end = time()
        n_motives = sum(len(self.motives_db[cid]) for cid in self.motives_db)
        n_clusters = len(self.motives_db)
        logger.info('after filtering: %d motives %d clusters in %ds' % (n_motives, n_clusters, (end - start)))

        start = time()
        # create models
        self._build_cluster_models()
        end = time()
        logger.info('motif model construction in %ds' % (end - start))

        start = time()
        # update motives counts
        self._update_counts(seqs)
        end = time()
        logger.info('updated motif counts in %ds' % (end - start))

    def info(self):
        text = []
        for cluster_id in self.motives_db:
            num_hits = len(self.cluster_hits[cluster_id])
            frac_num_hits = num_hits / float(self.dataset_size)
            text.append('Cluster: %s #%d (%.3f)' % (cluster_id, num_hits, frac_num_hits))
            for count, motif in sorted(self.motives_db[cluster_id], reverse=True):
                text.append('%s #%d' % (motif, count))
            text.append('')
        return text

    def _update_counts(self, seqs):
        self.dataset_size = len(seqs)
        cluster_hits = defaultdict(set)
        motives_db = defaultdict(list)
        for cluster_id in self.motives_db:
            motives = [motif for count, motif in self.motives_db[cluster_id]]
            motif_dict = {}
            for motif in motives:
                counter = 0
                for header, seq in seqs:
                    if motif in seq:
                        counter += 1
                        cluster_hits[cluster_id].add(header)
                motif_dict[motif] = counter
            # remove implied motives
            motif_dict_copy = motif_dict.copy()
            for motif_i in motif_dict:
                for motif_j in motif_dict:
                    if motif_dict[motif_i] == motif_dict[motif_j] and \
                            len(motif_j) < len(motif_i) and motif_j in motif_i:
                        if motif_j in motif_dict_copy:
                            motif_dict_copy.pop(motif_j)
            for motif in motif_dict_copy:
                motives_db[cluster_id].append((motif_dict[motif], motif))
        self.motives_db = motives_db
        self.cluster_hits = cluster_hits

    def fit_predict(self, seqs, return_list=False):
        self.fit(seqs)
        for prediction in self.predict(seqs, return_list=return_list):
            yield prediction

    def fit_transform(self, seqs, return_match=False):
        self.fit(seqs)
        for prediction in self.transform(seqs, return_match=return_match):
            yield prediction

    def predict(self, seqs, return_list=False):
        """Returns for each instance a list with the cluster ids that have a hit
        if  return_list=False then just return 1 if there is at least one hit from one cluster."""
        for header, seq in seqs:
            cluster_hits = []
            for cluster_id in self.motives_db:
                hits = list(self._cluster_hit(seq, cluster_id))
                if len(hits):
                    begin, end = min(hits)
                    cluster_hits.append((begin, cluster_id))
            if return_list is False:
                if len(cluster_hits):
                    yield len(cluster_hits)
                else:
                    yield 0
            else:
                yield [cluster_id for pos, cluster_id in sorted(cluster_hits)]

    def transform(self, seqs, return_match=False):
        """Transform an instance to a dense vector with features as cluster ID and entries 0/1 if a motif is found,
        if 'return_match' argument is True, then write a pair with (start position,end position)  in the entry
        instead of 0/1"""
        num = len(self.motives_db)
        for header, seq in seqs:
            cluster_hits = [0] * num
            for cluster_id in self.motives_db:
                hits = self._cluster_hit(seq, cluster_id)
                hits = list(hits)
                if return_match is False:
                    if len(hits):
                        cluster_hits[cluster_id] = 1
                else:
                    cluster_hits[cluster_id] = hits
            yield cluster_hits

    def _serial_graph_motif(self, seqs, placeholder=None):
        # make graphs
        iterable = sequence_to_eden(seqs)
        # use node importance and 'position' attribute to identify max_subarrays of a specific size
        graphs = list(self.vectorizer.annotate(iterable, estimator=self.estimator))

        for graph in graphs:
            seq, score = extract_sequence_and_score(graph)
            self.importances.append((seq, score))

        # use compute_max_subarrays to return an iterator over motives
        motives = []
        for graph in graphs:
            subarrays = compute_max_subarrays(graph=graph,
                                              min_subarray_size=self.min_subarray_size,
                                              max_subarray_size=self.max_subarray_size)
            if subarrays:
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
        results = [apply_async(pool, self._serial_graph_motif, args=(seqs[start:end], True))
                   for start, end in intervals]
        output = [p.get() for p in results]
        return list(chain(*output))

    def _motif_finder(self, seqs):
        if self.n_jobs > 1 or self.n_jobs == -1:
            return self._multiprocess_graph_motif(seqs)
        else:
            return self._serial_graph_motif(seqs)

    def _fit_predictive_model(self, seqs, neg_seqs=None):
        # duplicate iterator
        pos_seqs, pos_seqs_ = tee(seqs)
        pos_graphs = mp_pre_process(pos_seqs, pre_processor=sequence_to_eden,
                                    n_blocks=self.pre_processor_n_blocks,
                                    block_size=self.pre_processor_block_size,
                                    n_jobs=self.pre_processor_n_jobs)
        if neg_seqs is None:
            # shuffle seqs to obtain negatives
            neg_seqs = seq_to_seq(pos_seqs_,
                                  modifier=shuffle_modifier,
                                  times=self.negative_ratio,
                                  order=self.shuffle_order)
        neg_graphs = mp_pre_process(neg_seqs, pre_processor=sequence_to_eden,
                                    n_blocks=self.pre_processor_n_blocks,
                                    block_size=self.pre_processor_block_size,
                                    n_jobs=self.pre_processor_n_jobs)
        # fit discriminative estimator
        self.estimator = fit(pos_graphs, neg_graphs,
                             vectorizer=self.vectorizer,
                             n_iter_search=self.n_iter_search,
                             n_jobs=self.n_jobs,
                             n_blocks=self.n_blocks,
                             block_size=self.block_size,
                             random_state=self.random_state)

    def _cluster(self, seqs, clustering_algorithm=None):
        data_matrix = vectorize(seqs,
                                vectorizer=self.seq_vectorizer,
                                n_blocks=self.n_blocks,
                                block_size=self.block_size,
                                n_jobs=self.n_jobs)
        predictions = clustering_algorithm.fit_predict(data_matrix)
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

    def _build_cluster_models(self):
        self.cluster_models = []
        for cluster_id in self.motives_db:
            motives = [motif for count, motif in self.motives_db[cluster_id]]
            cluster_model = esm.Index()
            for motif in motives:
                cluster_model.enter(motif)
            cluster_model.fix()
            self.cluster_models.append(cluster_model)

    def _cluster_hit(self, seq, cluster_id):
        for ((start, end), motif) in self.cluster_models[cluster_id].query(seq):
            yield (start, end)
