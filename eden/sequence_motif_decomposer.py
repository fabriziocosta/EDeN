#!/usr/bin/env python

"""SequenceMotifDecomposer is a motif finder algorithm.

@author: Fabrizio Costa
@email: costa@informatik.uni-freiburg.de
"""

import logging
import multiprocessing as mp
import os
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
from corebio.seq import Alphabet, SeqList
import weblogolib as wbl
from scipy.cluster.hierarchy import linkage
import regex as re
from collections import Counter
from sklearn import metrics
from eden.util.NeedlemanWunsh import edit_distance
import random
import pylab as plt
import joblib
from scipy.optimize import curve_fit
import multiprocessing


logger = logging.getLogger(__name__)


def sigmoid(x, a, b):
    """sigmoid."""
    return 1 / (1 + np.exp(-(x - a) / b))


class PValueEvaluator(object):
    """Fit a parametrized sigmoid on the empirical cumulative distribution."""

    def __init__(self, random_state=1):
        """Constructor."""
        self.random_state = random_state
        self.a = -4
        self.b = 1

    def ecdf(self, x):
        """Empirical cumulative distribution function."""
        xs = np.sort(x)
        ys = np.arange(1, len(xs) + 1) / float(len(xs))
        return xs, ys

    def fit(self, scores):
        """fit."""
        if scores:
            xs, ys = self.ecdf(scores)
            popt, pcov = curve_fit(sigmoid, xs, ys)
            self.a, self.b = popt
        else:
            logger.debug('Warning: reverting to default values')
        logger.debug('ECDF fit on %d values' % (len(scores)))
        logger.debug('Optimal params: a:%.2f  b:%.2f' % (self.a, self.b))

    def predict(self, value):
        """pvalue."""
        y = sigmoid(value, self.a, self.b)
        p_val = 1 - y
        return p_val


def ecdf(x):
    """Empirical cumulative distribution function."""
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return xs, ys


def letter_regex(k, size, regex_th=0.3):
    """letter_regex."""
    code = []
    for letter, count in k:
        if count / float(size) > regex_th:
            if letter != '-':
                code.append(letter)
    if len(code) == 0:
        code_str = None
    elif len(code) == 1:
        code_str = code[0]
    else:
        code_str = '(' + '|'.join(code) + ')'
    return code_str


def consensus_regex(trimmed_align_seqs, regex_th):
    """consensus_regex."""
    cluster = []
    for h, align_seq in trimmed_align_seqs:
        str_list = [c for c in align_seq]
        concat_str = np.array(str_list, dtype=np.dtype('a'))
        cluster.append(concat_str)
    cluster = np.vstack(cluster)
    size = len(trimmed_align_seqs)
    for i, row in enumerate(cluster.T):
        c = Counter(row)
        k = c.most_common()
    code = ''
    for i, row in enumerate(cluster.T):
        c = Counter(row)
        k = c.most_common()
        l = letter_regex(k, size, regex_th=regex_th)
        if l:
            code += l
    return code


def find_occurrences(needle, haystack):
    """find_occurrences."""
    for h, s in haystack:
        matches = re.findall(needle, s, overlapped=True)
        if len(matches):
            yield 1
        else:
            yield 0


def occurrences(needle, haystack):
    """occurrences."""
    counts = sum(find_occurrences(needle, haystack))
    size = len(haystack)
    return counts, float(counts) / size


def extract_consensus(seqs, motives, regex_th):
    """extract_consensus."""
    for id in motives:
        c_regex = consensus_regex(motives[id]['trimmed_align_seqs'], regex_th)
        counts, freq = occurrences(c_regex, seqs)
        yield freq, id, c_regex, counts, motives[id]['consensus_seq']


def plot_location(needle, haystack,
                  cluster_id=None, nbins=20, size=(17, 2), fname=None):
    """plot_location."""
    locs = []
    for h, s in haystack:
        for match in re.finditer(needle, s):
            s = match.start()
            e = match.end()
            m = s + (e - s) / 2
            locs.append(m)
    plt.figure(figsize=size)
    n, bins, patches = plt.hist(
        locs, nbins, normed=0, facecolor='blue', alpha=0.3)
    plt.grid()
    plt.title(needle)
    plt.xlabel('Position')
    plt.ylabel('Num occurrences')
    if fname:
        plt.draw()
        figname = '%s_loc_%d.png' % (fname, cluster_id)
        plt.savefig(
            figname, bbox_inches='tight', transparent=True, pad_inches=0)
    else:
        figname = None
        plt.show()
    plt.close()
    return figname


def extract_location(needle, haystack):
    """extract_location."""
    locs = []
    for h, s in haystack:
        for match in re.finditer(needle, s):
            s = match.start()
            e = match.end()
            m = s + (e - s) / 2
            locs.append(m)
    if locs:
        avg_loc = np.percentile(locs, 50)
        std_loc = np.percentile(locs, 70) - np.percentile(locs, 30)
    else:
        avg_loc = -1
        std_loc = 0
    return avg_loc, std_loc


def hits(motives, ids=None):
    """hits."""
    for i in ids:
        for h, s in motives[i]['seqs']:
            tokens = h.split('<loc>')
            seq_id = tokens[0]
            begin, end = tokens[1].split(':')
            yield (seq_id, int(begin), int(end), i)


def compute_cooccurence(motives, ids=None):
    """compute_cooccurence."""
    if ids is None:
        ids = [id for id in motives]
    seqs_summary = defaultdict(list)
    for seq_id, begin, end, i in hits(motives, ids=ids):
        seqs_summary[seq_id].append((begin, end, i))

    distances = defaultdict(list)
    size = max(id for id in motives) + 1
    cooccurence_mtx = np.zeros((size, size))
    for seq_id in sorted(seqs_summary):
        cluster_ids = [cluster_id
                       for begin, end, cluster_id in seqs_summary[seq_id]]
        centers = defaultdict(list)
        for begin, end, cluster_id in seqs_summary[seq_id]:
            centers[cluster_id].append(begin + (end - begin) / 2)
        cluster_ids = set(cluster_ids)
        for i in cluster_ids:
            for j in cluster_ids:
                cooccurence_mtx[i, j] += 1
                if i != j:
                    # find closest instance j from  any instance in i
                    d_ij = []
                    for c_i in centers[i]:
                        for c_j in centers[j]:
                            d_ij.append(abs(c_i - c_j))
                    selected_abs = min(d_ij)
                    for c_i in centers[i]:
                        for c_j in centers[j]:
                            if selected_abs == abs(c_i - c_j):
                                selected = c_i - c_j
                    distances[(i, j)].append(selected)
    cooccurence_mtx = np.nan_to_num(cooccurence_mtx)
    orig_cooccurence_mtx = cooccurence_mtx.copy()
    cooccurence_list = []
    for i, row in enumerate(cooccurence_mtx):
        norm = row[i]
        if norm != 0:
            row /= norm
        else:
            row = np.zeros(row.shape)
        row[i] = 0
        cooccurence_list.append(row)
    norm_cooccurence_mtx = np.vstack(cooccurence_list)
    return orig_cooccurence_mtx, norm_cooccurence_mtx, distances


def plot_distance(cluster_id_i,
                  cluster_id_j,
                  regex_i,
                  regex_j,
                  distances,
                  nbins=5,
                  size=(6, 2),
                  fname=None):
    """plot_distance."""
    ds = distances[(cluster_id_i, cluster_id_j)]
    plt.figure(figsize=size)
    n, bins, patches = plt.hist(
        ds, nbins, normed=0, facecolor='green', alpha=0.3)
    plt.grid()
    plt.title('%s vs %s' % (regex_i, regex_j))
    plt.xlabel('Relative position')
    plt.ylabel('Num occurrences')
    if fname:
        plt.draw()
        figname = '%s_dist_%d_vs_%d.png' % (fname, cluster_id_i, cluster_id_j)
        plt.savefig(
            figname, bbox_inches='tight', transparent=True, pad_inches=0)
    else:
        figname = None
        plt.show()
    plt.close()
    return figname


def mean_shift_decomposition(sig, half_windw_size=5):
    """mean_shift_decomposition."""
    sig_len = len(sig)
    for i in range(half_windw_size, sig_len - half_windw_size):
        min_sig = np.min(sig[i - half_windw_size:i + half_windw_size])
        if min_sig == sig[i]:
            yield i


def box_decomposition(sig, half_windw_size=5):
    """box_decomposition."""
    ids = list(mean_shift_decomposition(sig, half_windw_size))
    for i in range(len(ids) - 1):
        start = ids[i]
        end = ids[i + 1]
        width = end - start
        val = sum(sig[start:end])
        yield val, start, end, width


def cumulative_score(seqs, smod):
    """cumulative_score."""
    median_len = np.median([len(s) for h, s in seqs])
    sigs = None
    for scores in smod.score(seqs):
        sig = np.array(scores)
        if len(sig) != median_len:
            logger.debug('Length mismatch: %d != %d' % (len(sig), median_len))
        if sigs is None:
            if len(sig) >= median_len:
                sigs = sig[:median_len]
        else:
            if len(sig) >= median_len:
                sigs = sigs + sig[:median_len]
    sig = np.array(sigs) / float(len(seqs))
    return sig


def trim_seqs(seqs, smod, half_windw_size=7):
    """trim_seqs."""
    sig = cumulative_score(seqs, smod)
    val, start, end, width = max(box_decomposition(sig, half_windw_size))
    logger.debug('val:%.1f beg:%s end:%s width:%s' % (val, start, end, width))
    for h, s in seqs:
        if s[start:end]:
            yield (h, s[start:end])


def plot_cumulative_score(smod,
                          seqs,
                          size=(6, 2),
                          fname=None):
    """plot_cumulative_score."""
    sig = cumulative_score(seqs, smod)
    plt.figure(figsize=size)
    sigp = np.copy(sig)
    sigp[sigp < 0] = 0
    plt.bar(range(len(sigp)), sigp, alpha=0.3, color='g')
    sign = np.copy(sig)
    sign[sign >= 0] = 0
    plt.bar(range(len(sign)), sign, alpha=0.3, color='r')
    plt.grid()
    plt.xlabel('Position')
    plt.ylabel('Importance score')
    if fname:
        plt.draw()
        figname = '%s_importance.png' % (fname)
        plt.savefig(
            figname, bbox_inches='tight', transparent=True, pad_inches=0)
    else:
        figname = None
        plt.show()
    plt.close()
    return figname
# ------------------------------------------------------------------------------


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
    logger.debug('Setup %.2f secs' % (time.time() - start_time))
    logger.debug('Vectorizing')

    start_time = time.time()
    matrices = []
    for i, p in enumerate(results):
        loc_start_time = time.time()
        pos_data_matrix = p.get()
        matrices += pos_data_matrix
        d_time = time.time() - start_time
        d_loc_time = time.time() - loc_start_time
        size = pos_data_matrix.shape
        logger.debug('%d %s (%.2f secs) (delta: %.2f)' %
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
    logger.debug('Setup %.2f secs' % (time.time() - start_time))
    logger.debug('Fitting')

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
        logger.debug('%d %s (%.2f secs) (delta: %.2f)' %
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
    logger.debug('Setup %.2f secs' % (time.time() - start_time))
    logger.debug('Performance evaluation')

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
        logger.debug('%d %s (%.2f secs) (delta: %.2f)' %
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
            subseq_seq = subarray['subarray_string']
            begin = subarray['begin']
            end = subarray['end']
            score = subarray['score']
            header = orig_header
            header += '<loc>%d:%d<loc>' % (begin, end)
            header += '<score>%.4f<score>' % (score)
            header += '<subseq>%s<subseq>' % (subseq_seq)
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
    logger.debug('Setup %.2f secs' % (time.time() - start_time))
    logger.debug('Annotating')

    start_time = time.time()
    subarrays_items = []
    for i, p in enumerate(results):
        loc_start_time = time.time()
        subarrays_item = p.get()
        subarrays_items += subarrays_item
        d_time = time.time() - start_time
        d_loc_time = time.time() - loc_start_time
        logger.debug('%d (%.2f secs) (delta: %.2f)' %
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
    logger.debug('Setup %.2f secs' % (time.time() - start_time))
    logger.debug('Predicting')

    start_time = time.time()
    scores_items = []
    for i, p in enumerate(results):
        loc_start_time = time.time()
        scores = p.get()
        scores_items += scores
        d_time = time.time() - start_time
        d_loc_time = time.time() - loc_start_time
        logger.debug('%d (%.2f secs) (delta: %.2f)' %
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
                 complexity=5,
                 n_clusters=10,
                 min_subarray_size=4,
                 max_subarray_size=10,
                 estimator=SGDClassifier(warm_start=True),
                 class_estimator=SGDClassifier(),
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
        self.class_estimator = class_estimator
        self.clusterer = clusterer
        self.clusterer_is_fit = False

    def save(self, model_name):
        """save."""
        joblib.dump(self, model_name, compress=1)

    def load(self, obj):
        """load."""
        self.__dict__.update(joblib.load(obj).__dict__)

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
            self.fit_decomposition(neg_seqs)
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

    def _decompose_header(self, header):
        score = header.split('<score>')[1]
        score = float(score)
        loc = header.split('<loc>')[1]
        begin, end = loc.split(':')
        begin = int(begin)
        end = int(end)
        subseq = header.split('<subseq>')[1]
        orig_header = header.split('<loc>')[0]
        return orig_header, score, begin, end, subseq

    def decompose(self, seqs=None, p_value=0.05):
        """decomposition_scores."""
        try:
            subarrays_items = multiprocess_subarray(
                seqs,
                vectorizer=self.vectorizer,
                estimator=self.estimator,
                min_subarray_size=self.min_subarray_size,
                max_subarray_size=self.max_subarray_size,
                block_size=self.pos_block_size,
                n_jobs=self.n_jobs)

            for header, seq in subarrays_items:
                components = self._decompose_header(header)
                orig_header, score, begin, end, subseq = components
                p = self.compute_p_value(score)
                if p <= p_value:
                    yield orig_header, begin, end, p, subseq

        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def decomposition_scores(self, seqs=None):
        """decomposition_scores."""
        try:
            subarrays_items = multiprocess_subarray(
                seqs,
                vectorizer=self.vectorizer,
                estimator=self.estimator,
                min_subarray_size=self.min_subarray_size,
                max_subarray_size=self.max_subarray_size,
                block_size=self.pos_block_size,
                n_jobs=self.n_jobs)

            for header, seq in subarrays_items:
                yield self._decompose_header(header)

        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def fit_decomposition(self, seqs=None):
        """fit_decomposition."""
        self.a, self.b = -4, 1
        scores = [score for header, score, begin, end, subseq in
                  self.decomposition_scores(seqs)]
        if scores:
            xs, ys = ecdf(scores)
            popt, pcov = curve_fit(sigmoid, xs, ys)
            self.a, self.b = popt
        else:
            logger.debug('Warning: reverting to default values')
        logger.debug('ECDF fit on %d values' % (len(scores)))
        logger.debug('Optimal params: a:%.2f  b:%.2f' % (self.a, self.b))

    def compute_p_value(self, value):
        """p_value."""
        y = sigmoid(value, self.a, self.b)
        p_val = 1 - y
        return p_val

    def compute_clusters(self, seqs=None, p_value=0.05):
        """compute_clusters."""
        try:
            subsequences = []
            iterable = self.decompose(seqs, p_value=p_value)
            for header, begin, end, p, subseq in iterable:
                new_header = header
                new_header += '<loc>' + str(begin) + ':'
                new_header += str(end) + '<loc>'
                subsequences.append((new_header, subseq))
            if not subsequences:
                raise Exception('No subarray was selected. Increase p_value.')
            logger.debug('Working on: %d fragments' % len(subsequences))
            n = multiprocessing.cpu_count()
            pos_block_size = len(subsequences) / n
            data_matrix = multiprocess_vectorize(
                subsequences,
                vectorizer=self.vectorizer,
                pos_block_size=pos_block_size,
                n_jobs=self.n_jobs)
            logger.debug('Clustering')
            logger.debug('working on %d instances' % data_matrix.shape[0])
            start_time = time.time()
            self.clusterer.set_params(n_clusters=self.n_clusters)
            if self.clusterer_is_fit:
                preds = self.class_estimator.predict(data_matrix)
            else:
                preds = self.clusterer.fit_predict(data_matrix)
                self.class_estimator.fit(data_matrix, preds)
                self.clusterer_is_fit = True
            dtime = time.time() - start_time
            logger.debug('...done  in %.2f secs' % (dtime))

            self.clusters = defaultdict(list)
            for pred, seq in zip(preds, subsequences):
                self.clusters[pred].append(seq)
            logger.debug('After clustering, %d motives' % len(self.clusters))
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

    def _compute_consensus_seq(self, align_seqs):
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

    def _is_high_quality(self,
                         seqs,
                         min_score=4,
                         min_freq=0.6,
                         min_cluster_size=10,
                         sample_size=200):
        ma = MuscleAlignWrapper(alphabet='rna')
        if len(seqs) > sample_size:
            sample_seqs = random.sample(seqs, sample_size)
        else:
            sample_seqs = seqs
        align_seqs = ma.transform(seqs=sample_seqs)
        score, trimmed_align_seqs = self._compute_score(align_seqs,
                                                        min_freq=min_freq)
        if score >= min_score and len(align_seqs) > min_cluster_size:
            return True
        else:
            return False

    def compute_motif(self,
                      seqs=None,
                      min_score=4,
                      min_freq=0.6,
                      min_cluster_size=10,
                      regex_th=.3,
                      sample_size=200):
        """compute_motif."""
        ma = MuscleAlignWrapper(alphabet='rna')
        if len(seqs) > sample_size:
            sample_seqs = random.sample(seqs, sample_size)
        else:
            sample_seqs = seqs
        align_seqs = ma.transform(seqs=sample_seqs)
        score, trimmed_align_seqs = self._compute_score(align_seqs,
                                                        min_freq=min_freq)
        if score >= min_score and len(align_seqs) > min_cluster_size:
            consensus_seq = self._compute_consensus_seq(trimmed_align_seqs)
            regex_seq = consensus_regex(trimmed_align_seqs, regex_th)
            motif = {'consensus_seq': consensus_seq,
                     'regex_seq': regex_seq,
                     'trimmed_align_seqs': trimmed_align_seqs,
                     'align_seqs': align_seqs,
                     'seqs': seqs}
            return True, motif
        else:
            return False, None

    def compute_motives(self,
                        clusters,
                        min_score=4,
                        min_freq=0.6,
                        min_cluster_size=10,
                        regex_th=.3,
                        sample_size=200):
        """compute_motives."""
        if not clusters:
            raise Exception('Error: No clusters.')
        mcs = min_cluster_size
        logger.debug('Alignment')
        motives = dict()
        for cluster_id in clusters:
            start_time = time.time()
            # align with muscle
            is_high_quality, motif = self.compute_motif(
                seqs=clusters[cluster_id],
                min_score=min_score,
                min_freq=min_freq,
                min_cluster_size=mcs,
                regex_th=regex_th,
                sample_size=sample_size)
            if is_high_quality:
                motives[cluster_id] = motif
                dtime = time.time() - start_time
                logger.debug(
                    'Cluster %d (#%d) (%.2f secs)' %
                    (cluster_id, len(clusters[cluster_id]), dtime))
        logger.debug('After motives computation, %d motives' % len(motives))
        return motives

    def _identify_mergeable_clusters(self, motives, similarity_th=0.8):
        for i in motives:
            for j in motives:
                if j > i:
                    seq_i = motives[i]['consensus_seq']
                    seq_j = motives[j]['consensus_seq']
                    nw_score = edit_distance(seq_i, seq_j, gap_penalty=-1)
                    rel_nw_score = 2 * nw_score / (len(seq_i) + len(seq_j))
                    if rel_nw_score > similarity_th:
                        yield rel_nw_score, i, j

    def merge(self,
              motives,
              similarity_th=0.5,
              min_score=4,
              min_freq=0.5,
              min_cluster_size=10,
              regex_th=.3,
              sample_size=200):
        """merge."""
        while True:
            ms = sorted([m for m in self._identify_mergeable_clusters(
                motives, similarity_th=similarity_th)], reverse=True)
            success = False
            for rel_nw_score, i, j in ms:
                if motives.get(i, None) and motives.get(j, None):
                    n_i = len(motives[i]['seqs'])
                    n_j = len(motives[j]['seqs'])
                    seqs = motives[i]['seqs'] + motives[j]['seqs']
                    is_high_quality, motif = self.compute_motif(
                        seqs=seqs,
                        min_score=min_score,
                        min_freq=min_freq,
                        min_cluster_size=min_cluster_size,
                        regex_th=regex_th,
                        sample_size=sample_size)
                    if is_high_quality:
                        info1 = 'Joining: %d (#%d), %d (#%d) score: %.2f' % \
                            (i, n_i, j, n_j, rel_nw_score)
                        info2 = ' deleting: %d  [%d is now #%d]' % \
                            (j, i, n_i + n_j)
                        logger.debug(info1 + info2)
                        # update motives
                        motives[i] = motif
                        del motives[j]
                        success = True
            if success is False:
                break
        # TODO: run the predictor to learn the new class definition
        logger.debug('After merge, %d motives' % len(motives))
        return motives

    def quality_filter(self,
                       seqs=None,
                       motives=None,
                       freq_th=None,
                       std_th=None):
        """quality_filter."""
        _motives = dict()
        for cluster_id in motives:
            regex_seq = motives[cluster_id]['regex_seq']
            counts, freq = occurrences(regex_seq, seqs)
            motives[cluster_id]['freq'] = freq
            motives[cluster_id]['counts'] = counts
            avg, std = extract_location(regex_seq, seqs)
            motives[cluster_id]['avg_pos'] = avg
            motives[cluster_id]['std_pos'] = std
            if freq_th is None or freq >= freq_th:
                if std_th is None or std <= std_th:
                    _motives[cluster_id] = motives[cluster_id]
        if len(_motives) == 0:
            logger.warning('Quality filter is too strict. Ignoring filter.')
            return motives
        else:
            logger.debug('After quality filter, %d motives' % len(_motives))
            return _motives

    def select_motives(self,
                       seqs=None,
                       p_value=0.05,
                       similarity_th=0.5,
                       min_score=4,
                       min_freq=0.5,
                       min_cluster_size=10,
                       regex_th=.3,
                       sample_size=200,
                       freq_th=None,
                       std_th=None):
        """select_motives."""
        orig_clusters = self.compute_clusters(seqs, p_value=p_value)
        motives = self.compute_motives(
            orig_clusters,
            min_score=min_score,
            min_freq=min_freq,
            min_cluster_size=min_cluster_size,
            regex_th=regex_th,
            sample_size=sample_size)
        motives = self.merge(
            motives,
            similarity_th=similarity_th,
            min_score=min_score,
            min_freq=min_freq,
            min_cluster_size=min_cluster_size,
            regex_th=regex_th,
            sample_size=sample_size)
        motives = self.quality_filter(
            seqs,
            motives,
            freq_th=freq_th,
            std_th=std_th)
        return motives

    def compute_logo(self,
                     cluster_id=None,
                     motif=None):
        """compute_logo."""
        alphabet = 'rna'
        color_scheme = 'classic'
        wb = Weblogo(output_format='png',
                     sequence_type=alphabet,
                     resolution=200,
                     stacks_per_line=60,
                     units='bits',
                     color_scheme=color_scheme)
        logo_image = wb.create_logo(seqs=motif['trimmed_align_seqs'])
        logo_txt = []
        info = '  - num subarrays: %d' % len(motif['seqs'])
        logo_txt.append(info)
        info = '  - consensus sequence: %s' % motif['consensus_seq']
        logo_txt.append(info)
        info = '  - consensus regex: %s' % motif['regex_seq']
        logo_txt.append(info)
        return logo_image, logo_txt

    def compute_logos(self,
                      motives,
                      ids=None):
        """compute_logos."""
        if motives:
            if ids is None:
                ids = [cluster_id for cluster_id in motives]
            logos = dict()
            for cluster_id in ids:
                logo_image, logo_txt = self.compute_logo(
                    cluster_id=cluster_id,
                    motif=motives[cluster_id])
                logos[cluster_id] = (logo_image, logo_txt)
            return logos
        else:
            logger.warning(
                'No logo to compute. Try more permissive parameters.')

    def _save_logo(self, logo, cluster_id, fname):
        imagename = '%s_logo_cl_%d.png' % (fname, cluster_id)
        with open(imagename, 'wb') as f:
            f.write(logo)
        return imagename

    def _wrap_image(self, fname, fill_width=True, output_type='screen'):
        pwd = os.getcwd()
        url = pwd + '/' + fname
        txt = []
        if fill_width:
            if output_type == 'pdf':
                txt.append('<p align="left"><img src="file://' + url +
                           '" style="width: 100%"></p>')
            else:
                txt.append('<p align="left"><img src="' + fname +
                           '" style="width: 100%"></p>')
        else:
            if output_type == 'pdf':
                txt.append('<p align="left"><img src="file://' + url +
                           '"></p>')
            else:
                txt.append('<p align="left"><img src="' + fname +
                           '"></p>')
        return '\n'.join(txt)

    def report(self,
               pos_seqs,
               all_seqs,
               motives,
               nbins=40,
               size=(17, 2),
               output_type='screen',
               fname=None):
        """Report in markdown format."""
        txt = []
        if motives:
            _, norm_cooccurence_mtx, distances = compute_cooccurence(motives)
            info = '### Summary: %d motives' % len(motives)
            txt.append(info)
            figname = plot_cumulative_score(
                self, pos_seqs, size=size, fname=fname)
            txt.append(self._wrap_image(figname, output_type=output_type))
            for freq, cluster_id in sorted([(motives[i]['freq'], i)
                                            for i in motives], reverse=True):
                info = '  - %.2s %s' % \
                    (cluster_id, motives[cluster_id]['consensus_seq'])
                txt.append(info)
            for freq, cluster_id in sorted([(motives[i]['freq'], i)
                                            for i in motives], reverse=True):
                info = '#### Motif id: %d' % cluster_id
                txt.append(info)
                logo_image, logo_txts = self.compute_logo(
                    cluster_id, motif=motives[cluster_id])
                figname = self._save_logo(logo_image, cluster_id, fname)
                for logo_txt in logo_txts:
                    txt.append(logo_txt)
                co = motives[cluster_id]['counts']
                fr = motives[cluster_id]['freq']
                info = '  - num occurrences of regex: %d' % (co)
                txt.append(info)
                info = '  - freq of occurrences of regex: %.2f' % (fr)
                txt.append(info)
                av = motives[cluster_id]['avg_pos']
                st = motives[cluster_id]['std_pos']
                info = '  - average location: %.1f +- %.1f' % (av, st)
                txt.append(info)
                txt.append(self._wrap_image(figname,
                                            fill_width=False,
                                            output_type=output_type))
                regex_i = motives[cluster_id]['regex_seq']
                figname = plot_location(
                    regex_i, all_seqs, cluster_id=cluster_id,
                    nbins=nbins, size=size, fname=fname)
                txt.append(self._wrap_image(figname, output_type=output_type))
                for j in motives:
                    regex_i = motives[i]['regex_seq']
                    if j != cluster_id:
                        regex_j = motives[j]['regex_seq']
                        ds = distances[(cluster_id, j)]
                        info = '  - num co-occurences %d %s vs %d %s: %d' % \
                            (cluster_id, regex_i, j, regex_j, len(ds))
                        txt.append(info)
                        if len(ds):
                            figname = plot_distance(
                                cluster_id, j,
                                regex_i, regex_j,
                                distances,
                                nbins=nbins, size=size, fname=fname)
                            txt.append(self._wrap_image(
                                figname,
                                output_type=output_type))
                txt.append('_' * 100)
        else:
            logger.warning(
                'No motives to report. Try more permissive parameters.')
        txt = '\n'.join(txt)
        return txt
