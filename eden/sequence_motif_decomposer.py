#!/usr/bin/env python

"""SequenceMotifDecomposer is a motif finder algorithm.

@author: Fabrizio Costa
@email: costa@informatik.uni-freiburg.de
"""

import logging
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
from StringIO import StringIO
from Bio import SeqIO
from Bio.Align.Applications import MuscleCommandline
from Bio.Alphabet import IUPAC
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from IPython.display import Image, display
from corebio.seq import Alphabet, SeqList
import weblogolib as wbl
from scipy.cluster.hierarchy import linkage
from collections import Counter
from sklearn import metrics


logger = logging.getLogger(__name__)


def serial_pre_process(iterable, vectorizer=None):
    """serial_pre_process."""
    data_matrix = vectorizer.transform(iterable)
    return data_matrix


def chunks(iterable, n):
    """chunks."""
    iterable = iter(iterable)
    while True:
        items = []
        for i in range(n):
            it = iterable.next()
            items.append(it)
        yield items


def multiprocess_vectorize(iterable,
                           vectorizer=None,
                           pos_block_size=100,
                           n_jobs=-1):
    """multiprocess_vectorize."""
    start_time = time.time()
    if n_jobs == -1:
        pool = mp.Pool()
    else:
        pool = mp.Pool(n_jobs)

    results = [apply_async(
        pool, serial_pre_process,
        args=(seqs, vectorizer))
        for seqs in chunks(iterable, pos_block_size)]
    logging.debug('Setup %.2f secs' % (time.time() - start_time))
    logging.debug('Vectorizing')

    start_time = time.time()
    matrices = []
    for i, p in enumerate(results):
        loc_start_time = time.time()
        pos_data_matrix = p.get()
        matrices += pos_data_matrix
        d_time = time.time() - start_time
        d_loc_time = time.time() - loc_start_time
        size = pos_data_matrix.shape
        logging.debug('%d %s (%.2f secs) (delta: %.2f)' %
                      (i, size, d_time, d_loc_time))

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
    """multiprocess_fit."""
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
    logging.debug('Fitting')

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
        d_time = time.time() - start_time
        d_loc_time = time.time() - loc_start_time
        size = pos_data_matrix.shape
        logging.debug('%d %s (%.2f secs) (delta: %.2f)' %
                      (i, size, d_time, d_loc_time))

    pool.close()
    pool.join()

    return estimator


def multiprocess_performance(pos_iterable, neg_iterable,
                             vectorizer=None,
                             estimator=None,
                             pos_block_size=100,
                             neg_block_size=100,
                             n_jobs=-1):
    """multiprocess_performance."""
    start_time = time.time()
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
    logging.debug('Performance evaluation')

    start_time = time.time()
    preds = []
    binary_preds = []
    true_targets = []
    for i, (p, n) in enumerate(izip(pos_results, neg_results)):
        loc_start_time = time.time()
        pos_data_matrix = p.get()
        y = [1] * pos_data_matrix.shape[0]
        neg_data_matrix = n.get()
        y += [-1] * neg_data_matrix.shape[0]
        y = np.array(y)
        true_targets.append(y)
        data_matrix = vstack([pos_data_matrix, neg_data_matrix])
        pred = estimator.decision_function(data_matrix)
        preds.append(pred)
        binary_pred = estimator.predict(data_matrix)
        binary_preds.append(binary_pred)
        d_time = time.time() - start_time
        d_loc_time = time.time() - loc_start_time
        size = pos_data_matrix.shape
        logging.debug('%d %s (%.2f secs) (delta: %.2f)' %
                      (i, size, d_time, d_loc_time))

    pool.close()
    pool.join()
    preds = np.hstack(preds)
    binary_preds = np.hstack(binary_preds)
    true_targets = np.hstack(true_targets)
    return preds, binary_preds, true_targets


def serial_subarray(iterable,
                    vectorizer=None,
                    estimator=None,
                    min_subarray_size=5,
                    max_subarray_size=10):
    """serial_subarray."""
    annotated_seqs = vectorizer.annotate(iterable, estimator=estimator)
    subarrays_items = []
    for (orig_header, orig_seq), (seq, score) in zip(iterable, annotated_seqs):
        subarrays = compute_max_subarrays_sequence(
            seq=seq, score=score,
            min_subarray_size=min_subarray_size,
            max_subarray_size=max_subarray_size,
            margin=1,
            output='all')
        subseqs = []
        for subarray in subarrays:
            seq = subarray['subarray_string']
            begin = subarray['begin']
            end = subarray['end']
            header = orig_header + '*%d:%d' % (begin, end)
            subseq = (header, seq)
            subseqs.append(subseq)
        subarrays_items += subseqs
    return subarrays_items


def multiprocess_subarray(iterable,
                          vectorizer=None,
                          estimator=None,
                          min_subarray_size=5,
                          max_subarray_size=10,
                          block_size=100,
                          n_jobs=-1):
    """multiprocess_subarray."""
    start_time = time.time()
    if n_jobs == -1:
        pool = mp.Pool()
    else:
        pool = mp.Pool(n_jobs)

    results = [apply_async(
        pool, serial_subarray,
        args=(seqs,
              vectorizer,
              estimator,
              min_subarray_size,
              max_subarray_size))
        for seqs in chunks(iterable, block_size)]
    logging.debug('Setup %.2f secs' % (time.time() - start_time))
    logging.debug('Annotating')

    start_time = time.time()
    subarrays_items = []
    for i, p in enumerate(results):
        loc_start_time = time.time()
        subarrays_item = p.get()
        subarrays_items += subarrays_item
        d_time = time.time() - start_time
        d_loc_time = time.time() - loc_start_time
        logging.debug('%d (%.2f secs) (delta: %.2f)' %
                      (i, d_time, d_loc_time))

    pool.close()
    pool.join()
    return subarrays_items


def serial_score(iterable,
                 vectorizer=None,
                 estimator=None):
    """serial_score."""
    annotated_seqs = vectorizer.annotate(iterable, estimator=estimator)
    scores = [score for seq, score in annotated_seqs]
    return scores


def multiprocess_score(iterable,
                       vectorizer=None,
                       estimator=None,
                       block_size=100,
                       n_jobs=-1):
    """multiprocess_score."""
    start_time = time.time()
    if n_jobs == -1:
        pool = mp.Pool()
    else:
        pool = mp.Pool(n_jobs)

    results = [apply_async(
        pool, serial_score,
        args=(seqs,
              vectorizer,
              estimator))
        for seqs in chunks(iterable, block_size)]
    logging.debug('Setup %.2f secs' % (time.time() - start_time))
    logging.debug('Predicting')

    start_time = time.time()
    scores_items = []
    for i, p in enumerate(results):
        loc_start_time = time.time()
        scores = p.get()
        scores_items += scores
        d_time = time.time() - start_time
        d_loc_time = time.time() - loc_start_time
        logging.debug('%d (%.2f secs) (delta: %.2f)' %
                      (i, d_time, d_loc_time))

    pool.close()
    pool.join()
    return scores_items
# ------------------------------------------------------------------------------


def _fasta_to_fasta(lines):
    seq = ""
    for line in lines:
        if line:
            if line[0] == '>':
                if seq:
                    yield seq
                    seq = ""
                line_str = str(line)
                yield line_str.strip()
            else:
                line_str = line.split()
                if line_str:
                    seq += str(line_str[0]).strip()
    if seq:
        yield seq


# ------------------------------------------------------------------------------


class MuscleAlignWrapper(object):
    """A wrapper to perform Muscle Alignment on sequences."""

    def __init__(self,
                 diags=False,
                 maxiters=16,
                 maxhours=None,
                 # TODO: check if this alphabet is required
                 # it over-rides tool.alphabet
                 alphabet='dna',  # ['dna', 'rna', 'protein']
                 ):
        """Initialize an instance."""
        self.diags = diags
        self.maxiters = maxiters
        self.maxhours = maxhours

        if alphabet == 'protein':
            self.alphabet = IUPAC.protein
        elif alphabet == 'rna':
            self.alphabet = IUPAC.unambiguous_rna
        else:
            self.alphabet = IUPAC.unambiguous_dna

    def _seq_to_stdin_fasta(self, seqs):
        # seperating headers
        headers, instances = [list(x) for x in zip(*seqs)]

        instances_seqrecord = []
        for i, j in enumerate(instances):
            instances_seqrecord.append(
                SeqRecord(Seq(j, self.alphabet), id=str(i)))

        handle = StringIO()
        SeqIO.write(instances_seqrecord, handle, "fasta")
        data = handle.getvalue()
        return headers, data

    def _perform_ma(self, data):
        params = {'maxiters': 7}
        if self.diags is True:
            params['diags'] = True
        if self.maxhours is not None:
            params['maxhours'] = self.maxhours

        muscle_cline = MuscleCommandline(**params)
        stdout, stderr = muscle_cline(stdin=data)
        return stdout

    def _fasta_to_seqs(self, headers, stdout):
        out = list(_fasta_to_fasta(stdout.split('\n')))
        motif_seqs = [''] * len(headers)
        for i in range(len(out[:-1]))[::2]:
            id = int(out[i].split(' ')[0].split('>')[1])
            motif_seqs[id] = out[i + 1]

        return zip(headers, motif_seqs)

    def transform(self, seqs=[]):
        """Carry out alignment."""
        headers, data = self._seq_to_stdin_fasta(seqs)
        stdout = self._perform_ma(data)
        aligned_seqs = self._fasta_to_seqs(headers, stdout)
        return aligned_seqs


# ------------------------------------------------------------------------------


class Weblogo(object):
    """A wrapper of weblogolib for creating sequence."""

    def __init__(self,

                 output_format='png',  # ['eps','png','png_print','jpeg']
                 stacks_per_line=40,
                 sequence_type='dna',  # ['protein','dna','rna']
                 ignore_lower_case=False,
                 # ['bits','nats','digits','kT','kJ/mol','kcal/mol','probability']
                 units='bits',
                 first_position=1,
                 logo_range=list(),
                 # composition = 'auto',
                 scale_stack_widths=True,
                 error_bars=True,
                 title='',
                 figure_label='',
                 show_x_axis=True,
                 x_label='',
                 show_y_axis=True,
                 y_label='',
                 y_axis_tic_spacing=1.0,
                 show_ends=False,
                 # ['auto','base','pairing','charge','chemistry','classic','monochrome']
                 color_scheme='classic',
                 resolution=96,
                 fineprint='',
                 ):
        """Initialize an instance."""
        options = wbl.LogoOptions()

        options.stacks_per_line = stacks_per_line
        options.sequence_type = sequence_type
        options.ignore_lower_case = ignore_lower_case
        options.unit_name = units
        options.first_index = first_position
        if logo_range:
            options.logo_start = logo_range[0]
            options.logo_end = logo_range[1]
        options.scale_width = scale_stack_widths
        options.show_errorbars = error_bars
        if title:
            options.title = title
        if figure_label:
            options.logo_label = figure_label
        options.show_xaxis = show_x_axis
        if x_label:
            options.xaxis_label = x_label
        options.show_yaxis = show_y_axis
        if y_label:
            options.yaxis_label = y_label
        options.yaxis_tic_interval = y_axis_tic_spacing
        options.show_ends = show_ends
        options.color_scheme = wbl.std_color_schemes[color_scheme]
        options.resolution = resolution
        if fineprint:
            options.fineprint = fineprint

        self.options = options
        self.output_format = output_format

    def create_logo(self, seqs=[]):
        """Create sequence logo for input sequences."""
        # seperate headers
        headers, instances = [list(x)
                              for x in zip(*seqs)]

        if self.options.sequence_type is 'rna':
            alphabet = Alphabet('ACGU')
        elif self.options.sequence_type is 'protein':
            alphabet = Alphabet('ACDEFGHIKLMNPQRSTVWY')
        else:
            alphabet = Alphabet('AGCT')
        motif_corebio = SeqList(alist=instances, alphabet=alphabet)
        data = wbl.LogoData().from_seqs(motif_corebio)

        format = wbl.LogoFormat(data, self.options)

        if self.output_format == 'png':
            return wbl.png_formatter(data, format)
        elif self.output_format == 'png_print':
            return wbl.png_print_formatter(data, format)
        elif self.output_format == 'jpeg':
            return wbl.jpeg_formatter(data, format)
        else:
            return wbl.eps_formatter(data, format)


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
        self.vectorizer = Vectorizer(complexity=complexity,
                                     auto_weights=True,
                                     nbits=15)
        self.estimator = estimator
        self.clusterer = clusterer
        self.clusterer_is_fit = False

    def fit(self, pos_seqs=None, neg_seqs=None):
        """fit."""
        try:
            self.estimator = multiprocess_fit(
                pos_seqs, neg_seqs,
                vectorizer=self.vectorizer,
                estimator=self.estimator,
                pos_block_size=self.pos_block_size,
                neg_block_size=self.neg_block_size,
                n_jobs=self.n_jobs)
            return self
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def performance(self, pos_seqs=None, neg_seqs=None):
        """performance."""
        try:
            y_pred, y_binary, y_test = multiprocess_performance(
                pos_seqs, neg_seqs,
                vectorizer=self.vectorizer,
                estimator=self.estimator,
                pos_block_size=self.pos_block_size,
                neg_block_size=self.neg_block_size,
                n_jobs=self.n_jobs)
            # confusion matrix
            cm = metrics.confusion_matrix(y_test, y_binary)
            np.set_printoptions(precision=2)
            logger.info('Confusion matrix:')
            logger.info(cm)

            # classification
            logger.info('Classification:')
            logger.info(metrics.classification_report(y_test, y_binary))

            # roc
            logger.info('ROC: %.3f' % (metrics.roc_auc_score(y_test, y_pred)))

        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def predict(self, pos_seqs=None):
        """predict."""
        try:
            subarrays_items = multiprocess_subarray(
                pos_seqs,
                vectorizer=self.vectorizer,
                estimator=self.estimator,
                min_subarray_size=self.min_subarray_size,
                max_subarray_size=self.max_subarray_size,
                block_size=self.pos_block_size,
                n_jobs=self.n_jobs)

            data_matrix = multiprocess_vectorize(
                subarrays_items,
                vectorizer=self.vectorizer,
                pos_block_size=self.pos_block_size,
                n_jobs=self.n_jobs)
            logging.debug('Clustering')
            start_time = time.time()
            self.clusterer.set_params(n_clusters=self.n_clusters)
            if self.clusterer_is_fit:
                preds = self.clusterer.predict(data_matrix)
            else:
                preds = self.clusterer.fit_predict(data_matrix)
                self.clusterer_is_fit = True
            dtime = time.time() - start_time
            logger.debug('...done  in %.2f secs' % (dtime))

            self.clusters = defaultdict(list)
            for pred, seq in zip(preds, subarrays_items):
                self.clusters[pred].append(seq)

            return self.clusters
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def score(self, seqs=None):
        """fit."""
        try:
            for score in multiprocess_score(seqs,
                                            vectorizer=self.vectorizer,
                                            estimator=self.estimator,
                                            block_size=self.pos_block_size,
                                            n_jobs=self.n_jobs):
                yield score
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _order_clusters(self, clusters, complexity=3):
        sep = ' ' * (complexity * 2)
        # join all sequences in a cluster with enough space that
        # kmers dont interfere
        cluster_seqs = []
        for cluster_id in clusters:
            if len(clusters[cluster_id]) > 0:
                seqs = [s for h, s in clusters[cluster_id]]
                seq = sep.join(seqs)
                cluster_seqs.append(seq)

        # vectorize the seqs and compute their gram matrix K
        cluster_vecs = Vectorizer(complexity).transform(cluster_seqs)
        gram_matrix = metrics.pairwise.pairwise_kernels(
            cluster_vecs, metric='linear')
        c = linkage(gram_matrix, method='single')
        orders = []
        for id1, id2 in c[:, 0:2]:
            if id1 < len(cluster_seqs):
                orders.append(int(id1))
            if id2 < len(cluster_seqs):
                orders.append(int(id2))
        return orders

    def _compute_most_freq_seq(self, align_seqs):
        cluster = []
        for h, align_seq in align_seqs:
            str_list = [c for c in align_seq]
            concat_str = np.array(str_list, dtype=np.dtype('a'))
            cluster.append(concat_str)
        cluster = np.vstack(cluster)
        seq = ''
        for i, row in enumerate(cluster.T):
            c = Counter(row)
            k = c.most_common()
            seq += k[0][0]
        return seq

    def _compute_score(self, align_seqs, min_freq=0.8):
        dim = len(align_seqs)
        cluster = []
        for h, align_seq in align_seqs:
            str_list = [c for c in align_seq]
            concat_str = np.array(str_list, dtype=np.dtype('a'))
            cluster.append(concat_str)
        cluster = np.vstack(cluster)
        score = 0
        to_be_removed = []
        for i, row in enumerate(cluster.T):
            c = Counter(row)
            k = c.most_common()
            if k[0][0] == '-':
                to_be_removed.append(i)
                val = k[1][1]
            else:
                val = k[0][1]
            if float(val) / dim >= min_freq:
                score += 1
        trimmed_align_seqs = []
        for h, align_seq in align_seqs:
            trimmed_align_seq = [a for i, a in enumerate(align_seq)
                                 if i not in to_be_removed]
            trimmed_align_seqs.append((h, ''.join(trimmed_align_seq)))
        return score, trimmed_align_seqs

    def compute_motives(self,
                        clusters,
                        min_score=4,
                        min_freq=0.6,
                        min_cluster_size=10):
        """compute_motives."""
        mcs = min_cluster_size
        ma = MuscleAlignWrapper(alphabet='rna')
        start_time = time.time()
        order = self._order_clusters(clusters, complexity=3)
        dtime = time.time() - start_time
        logger.debug('Order determined  in %.2f secs' % (dtime))
        logging.debug('Alignment')

        self.motives = []
        for cluster_id in order:
            start_time = time.time()
            # align with muscle
            align_seqs = ma.transform(seqs=clusters[cluster_id])
            score, trimmed_align_seqs = self._compute_score(align_seqs,
                                                            min_freq=min_freq)
            if score >= min_score and len(align_seqs) > mcs:
                consensus_seq = self._compute_most_freq_seq(trimmed_align_seqs)
                self.motives.append((cluster_id,
                                     consensus_seq,
                                     trimmed_align_seqs,
                                     align_seqs,
                                     clusters[cluster_id]))
                dtime = time.time() - start_time
                logger.debug(
                    'Cluster %d (#%d) (%.2f secs)  score: %d > %.2f' %
                    (cluster_id, len(align_seqs), dtime, score, min_freq))
        return self.motives

    def display_logos(self,
                      motives):
        """display_logos."""
        alphabet = 'rna'
        color_scheme = 'classic'
        wb = Weblogo(output_format='png',
                     sequence_type=alphabet,
                     resolution=200,
                     stacks_per_line=60,
                     units='bits',
                     color_scheme=color_scheme)
        logos = []
        for m in motives:
            cluster_id, consensus_seq, trimmed_align_seqs, align_seqs, seqs = m
            logo_image = wb.create_logo(seqs=trimmed_align_seqs)
            logos.append(logo_image)
            logger.info('Cluster id: %d' % cluster_id)
            logger.info(consensus_seq)
            display(Image(logo_image))
        return logos
