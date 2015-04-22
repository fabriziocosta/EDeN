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
                 max_subarray_size=9,
                 min_motif_count=1,
                 min_cluster_size=1,
                 training_size=None,
                 negative_ratio=2,
                 n_iter_search=1,
                 complexity=3,
                 nbits=14,
                 algorithm='DBSCAN',
                 n_clusters=4,
                 eps=0.3,
                 threshold=0.2,
                 branching_factor=50,
                 min_samples=3,
                 verbosity=0,
                 n_blocks=2,
                 n_jobs=2,
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
                    self.motives_db[cluster_id].append((count, motif_i))

    def fit(self, seqs):
        start = time()
        if self.training_size is None:
            training_seqs = seqs
        else:
            training_seqs = random.sample(seqs, self.training_size)
        self.fit_predictive_model(training_seqs)
        end = time()
        logger.info('model induction: %d secs' % (end - start))

        start = time()
        self.motives = motif_finder(seqs,
                                    vectorizer=self.vectorizer,
                                    estimator=self.estimator,
                                    min_subarray_size=self.min_subarray_size,
                                    max_subarray_size=self.max_subarray_size,
                                    n_jobs=self.n_jobs)
        end = time()
        logger.info('motives extraction: %d motives %d secs' % (len(self.motives), end - start))

        start = time()
        self.cluster(self.motives)
        self.filter()
        end = time()
        logger.info('motives clustering: %d clusters %d secs' % (len(self.clusters), end - start))

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


def main(args):
    from eden.converter.fasta import fasta_to_sequence
    seqs = fasta_to_sequence(args.input_file)
    seqs = list(seqs)
    sequence_motif = SequenceMotif(training_size=args.training_size,
                           verbosity=args.verbosity,
                           algorithm=args.algorithm,
                           min_subarray_size=args.min_subarray_size,
                           max_subarray_size=args.max_subarray_size,
                           min_motif_count=args.min_motif_count,
                           min_cluster_size=args.min_cluster_size,
                           n_clusters=args.n_clusters,
                           eps=args.eps,
                           min_samples=args.min_samples,
                           threshold=args.threshold,
                           branching_factor=args.branching_factor,
                           negative_ratio=args.negative_ratio,
                           n_iter_search=args.n_iter_search,
                           nbits=args.nbits,
                           complexity=args.complexity,
                           n_blocks=args.n_blocks,
                           n_jobs=args.n_jobs)
    sequence_motif.fit(seqs)

    # output motives
    text = []
    for cluster_id in sequence_motif.motives_db:
        text.append("# %d" % cluster_id)
        for count, motif in sorted(sequence_motif.motives_db[cluster_id], reverse=True):
            text.append("%s %d" % (motif, count))
        text.append("")
    save_output(text=text, output_dir_path=args.output_dir_path, out_file_name='motifs', logger=logger)

    # output occurrences of cluster hit in sequences
    predictions = sequence_motif.predict(seqs, return_list=True)
    text = []
    for j, p in enumerate(predictions):
        line = ""
        for i in range(len(p)):
            line += "%d " % p[i]
        if line:
            line = str(j) + "\t" + line
            text.append(line)
    save_output(text=text, output_dir_path=args.output_dir_path, out_file_name='sequences_cluster_id_hit', logger=logger)

    # output occurrences of motives in sequences
    predictions = sequence_motif.transform(seqs, return_match=True)
    text = []
    for j, p in enumerate(predictions):
        line = ""
        for i in range(len(p)):
            if len(p[i]):
                line += "%d:%s " % (i, p[i])
        if line:
            line = str(j) + "\t" + line
            text.append(line)
    save_output(text=text, output_dir_path=args.output_dir_path, out_file_name='sequences_cluster_match_position', logger=logger)

    # save state of motif finder
    if args.log_full_state:
        logger.debug(sequence_motif.__dict__)
    else:
        logger.debug(sequence_motif.estimator)
        logger.debug(sequence_motif.vectorizer)
        logger.debug(sequence_motif.seq_vectorizer)


if __name__ == "__main__":
    description = """
    Explicit Decomposition with Neighborhood (EDeN) utility program.
    Motif finder driver. Offers: fit, predict and transform.
    """
    epilog = """
    Cite: 
    Costa, Fabrizio, and Kurt De Grave, 'Fast neighborhood subgraph pairwise distance kernel', Proceedings of the 26th International Conference on Machine Learning. 2010.
    """
    start_time = time()
    parser = argparse.ArgumentParser(description=description, epilog=epilog, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input-file",
                        dest="input_file",
                        help="Path to a FASTA file.",
                        required=True)
    parser.add_argument("-o", "--output-dir",
                        dest="output_dir_path",
                        help="Path to output directory.",
                        default="out")
    parser.add_argument("-a", "--algorithm",  choices=["MiniBatchKMeans", "DBSCAN", "Birch"],
                        help="""
                        Type of clustering algorithm for extracted motives. 
                        For deatails see: 
                        http://scikit-learn.org/stable/modules/clustering.html 
                        """,
                        default="DBSCAN")
    parser.add_argument("-B", "--nbits",
                        type=int,
                        help="Number of bits used to express the graph kernel features. A value of 20 corresponds to 2**20=1 million possible features.",
                        default=20)
    parser.add_argument("-C", "--complexity",
                        type=int,
                        help="Size of the generalization of k-mers for graphs.",
                        default=4)
    parser.add_argument("-t", "--training-size",
                        dest="training_size",
                        type=int,
                        help="Size of the random sequence sample to use for fitting the discriminative model. If None then all instances are used.",
                        default=None)
    parser.add_argument("-n", "--negative-ratio",
                        dest="negative_ratio",
                        type=int,
                        help="Factor multiplying the training-size to obtain the number of negative instances generated by random permutation.",
                        default=2)
    parser.add_argument("-e", "--n-iter-search",
                        dest="n_iter_search",
                        type=int,
                        help="Number of randomly geenrated hyper parameter configurations tried during the discriminative model optimization. A value of 1 implies using the estimator default values.",
                        default=1)
    parser.add_argument("-m", "--min-subarray-size",
                        dest="min_subarray_size",
                        type=int,
                        help="Minimal size in number of nucleotides of the motives to search.",
                        default=7)
    parser.add_argument("-M", "--max-subarray-size",
                        dest="max_subarray_size",
                        type=int,
                        help="Maximal size in number of nucleotides of the motives to search.",
                        default=10)
    parser.add_argument("-c", "--min-motif-count",
                        dest="min_motif_count",
                        type=int,
                        help="Minimal number of occurrences for a motif sequence to be accepted.",
                        default=1)
    parser.add_argument("-s", "--min-cluster-size",
                        dest="min_cluster_size",
                        type=int,
                        help="Minimal number of motif sequences in a cluster to accept the clsuter.",
                        default=1)
    parser.add_argument("-u", "--n-clusters",
                        dest="n_clusters",
                        type=int,
                        help="Number of clusters. In MiniBatchKMeans and Birch clustering algorithms.",
                        default=4)
    parser.add_argument("-E", "--eps",
                        type=float,
                        help="The maximum distance between two samples for them to be considered as in the same neighborhood. In DBSCAN.",
                        default=0.3)
    parser.add_argument("-S", "--min-samples",
                        dest="min_samples",
                        type=int,
                        help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself. In DBSCAN.",
                        default=3)
    parser.add_argument("-T", "--threshold",
                        type=float,
                        help="The radius of the subcluster obtained by merging a new sample and the closest subcluster should be lesser than the threshold. Otherwise a new subcluster is started. In Birch.",
                        default=0.2)
    parser.add_argument("-f", "--branching-factor",
                        dest="branching_factor",
                        type=int,
                        help="Maximum number of CF subclusters in each node. If a new samples enters such that the number of subclusters exceed the branching_factor then the node has to be split. The corresponding parent also has to be split and if the number of subclusters in the parent is greater than the branching factor, then it has to be split recursively. In Birch.",
                        default=3)
    parser.add_argument("-j", "--n-jobs",
                        dest="n_jobs",
                        type=int,
                        help="Number of cores to use in multiprocessing.",
                        default=2)
    parser.add_argument("-b", "--n-blocks",
                        dest="n_blocks",
                        type=int,
                        help="Number of blocks in which to divide the input for the multiprocessing elaboration.",
                        default=2)
    parser.add_argument("-v", "--verbosity",
                        action="count",
                        help="Increase output verbosity")
    parser.add_argument("-l", "--log-full-state",
                        dest="log_full_state",
                        help="If set, log all the internal parameters values and motif database of the motif finder. Warning: it can generate large logging files.",
                        action="store_true")
    args = parser.parse_args()

    logger = setup.logger(logger_name=sys.argv[0], filename="log", verbosity=args.verbosity)
    logger.info('-' * 80)
    logger.info('Program: %s' % sys.argv[0])
    logger.info('Parameters: %s' % args.__dict__)
    try:
        main(args)
    except Exception:
        import datetime
        curr_time = datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
        logger.exception("Program run failed on %s" % curr_time)
    finally:
        end_time = time()
        logger.info('Elapsed time: %.1f sec', end_time - start_time)
