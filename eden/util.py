#!/usr/bin/env python
"""Provides utilities for file handling."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import io
from sklearn.externals import joblib
import requests
import os
import sys
from collections import deque
from itertools import tee
import random
import logging.handlers

import multiprocessing as mp
import time

from toolz.curried import concat

import logging
logger = logging.getLogger(__name__)


def timeit(method):
    """Time decorator."""
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logger.debug('%s  %2.2f sec' % (method.__name__, te - ts))
        return result
    return timed


def pmap(func, iterable, chunk_size=1):
    """Multi-core map."""
    pool = mp.Pool()
    result = pool.map(func, iterable, chunksize=chunk_size)
    pool.close()
    pool.join()
    return list(result)


def ppipe(iterable, func, chunk_size=1):
    """Multi-core pipe."""
    out = pmap(func, iterable, chunk_size)
    return list(concat(out))


def configure_logging(logger, verbosity=0, filename=None):
    """Utility to configure the logging aspects.

    If filename is None then no info is stored in files.
    If filename is not None then everything that is logged is dumped to file
    (including program traces).
    Verbosity is an int that can take values: 0 -> warning,
    1 -> info, >=2 -> debug.
    All levels are displayed on stdout, not on stderr.
    Please use exceptions and asserts to output on stderr.
    """
    logger.propagate = False
    logger.handlers = []
    log_level = logging.WARNING
    if verbosity == 1:
        log_level = logging.INFO
    elif verbosity == 2:
        log_level = logging.DEBUG
    else:
        log_level = 4
    logger.setLevel(log_level)
    # create console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level)
    # create formatter
    cformatter = logging.Formatter('%(message)s')
    # add formatter to ch
    ch.setFormatter(cformatter)
    # add handlers to logger
    logger.addHandler(ch)

    if filename is not None:
        # create a file handler
        fh = logging.handlers.RotatingFileHandler(filename=filename,
                                                  maxBytes=10000000,
                                                  backupCount=10)
        fh.setLevel(logging.DEBUG)
        # create formatter
        fformatter = logging.Formatter('%(asctime)s | %(levelname)-6s | %(name)10s | %(filename)10s |\
   %(lineno)4s | %(message)s')
        # add formatter to fh
        fh.setFormatter(fformatter)
        # add handlers to logger
        logger.addHandler(fh)


def _serialize_list(items, separator='_'):
    if isinstance(items, str):
        return items
    if is_iterable(items):
        if isinstance(items, list):
            return str(separator.join([str(item) for item in items]))
        if isinstance(items, dict):
            return str(separator.join([str(key) + ':' + str(items[key])
                                       for key in items]))
    else:
        return str(items)


def serialize_dict(the_dict, full=True, offset='small'):
    """serialize_dict."""
    if the_dict:
        text = []
        for key in sorted(the_dict):
            if offset == 'small':
                line = '%10s: %s' % (key, the_dict[key])
            elif offset == 'large':
                line = '%25s: %s' % (key, the_dict[key])
            elif offset == 'very_large':
                line = '%50s: %s' % (key, the_dict[key])
            else:
                raise Exception('unrecognized option: %s' % offset)
            line = line.replace('\n', ' ')
            if full is False:
                if len(line) > 100:
                    line = line[:100] + '  ...  ' + line[-20:]
            text.append(line)
        return '\n'.join(text)
    else:
        return ""


def read(uri):
    """Abstract read function.

    EDeN can accept a URL, a file path and a python list.
    In all cases an iterable object should be returned.
    """
    if isinstance(uri, list):
        # test if it is iterable: works for lists and generators, but not for
        # strings
        return uri
    else:
        try:
                    # try if it is a URL and if we can open it
            f = requests.get(uri).text.split('\n')
        except ValueError:
            # assume it is a file object
            f = open(uri)
        return f


def is_iterable(test):
    """is_iterable."""
    if hasattr(test, '__iter__'):
        return True
    else:
        return False


def describe(data_matrix):
    """Get the shape of a sparse matrix and its average nnz."""
    return 'Instances: %3d ; Features: %d with an avg of %d per instance' % \
        (data_matrix.shape[0], data_matrix.shape[1],
         data_matrix.getnnz() / data_matrix.shape[0])


def iterator_size(iterable):
    """Length of an iterator.

    Note: if the iterable is a generator it consumes it.
    """
    if hasattr(iterable, '__len__'):
        return len(iterable)

    d = deque(enumerate(iterable, 1), maxlen=1)
    if d:
        return d[0][0]
    else:
        return 0


def random_bipartition(int_range, relative_size=.7, random_state=None):
    """random_bipartition."""
    if not random_state:
        random_state = random.random()
    random.seed(random_state)
    ids = list(range(int_range))
    random.shuffle(ids)
    split_point = int(int_range * relative_size)
    return ids[:split_point], ids[split_point:]


def selection_iterator(iterable, ids):
    """selection_iterator.

    Given an iterable and a list of ids (zero based) yield only the
    items whose id matches.
    """
    ids = sorted(ids)
    counter = 0
    for id, item in enumerate(iterable):
        if id == ids[counter]:
            yield item
            counter += 1
            if counter == len(ids):
                break


def random_bipartition_iter(iterable, relative_size=.5, random_state=1):
    """random_bipartition_iter."""
    size_iterable, iterable1, iterable2 = tee(iterable, 3)
    size = iterator_size(size_iterable)
    part1_ids, part2_ids = random_bipartition(
        size, relative_size=relative_size, random_state=random_state)
    part1_iterable = selection_iterator(iterable1, part1_ids)
    part2_iterable = selection_iterator(iterable2, part2_ids)
    return part1_iterable, part2_iterable


def store_matrix(matrix='',
                 output_dir_path='',
                 out_file_name='',
                 output_format=''):
    """store_matrix."""
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    full_out_file_name = os.path.join(output_dir_path, out_file_name)
    if output_format == "MatrixMarket":
        if len(matrix.shape) == 1:
            raise Exception(
                "'MatrixMarket' format supports only 2D dimensional array\
                and not vectors")
        else:
            io.mmwrite(full_out_file_name, matrix, precision=None)
    elif output_format == "numpy":
        np.save(full_out_file_name, matrix)
    elif output_format == "joblib":
        joblib.dump(matrix, full_out_file_name)
    elif output_format == "text":
        with open(full_out_file_name, "w") as f:
            if len(matrix.shape) == 1:
                for x in matrix:
                    f.write("%s\n" % (x))
            else:
                raise Exception(
                    "'text' format supports only mono dimensional array\
                    and not matrices")
    logger.info("Written file: %s" % full_out_file_name)


def dump(obj, output_dir_path='', out_file_name=''):
    """dump."""
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    full_out_file_name = os.path.join(output_dir_path, out_file_name) + ".pkl"
    joblib.dump(obj, full_out_file_name)


def load(output_dir_path='', out_file_name=''):
    """load."""
    full_out_file_name = os.path.join(output_dir_path, out_file_name) + ".pkl"
    obj = joblib.load(full_out_file_name)
    return obj


def report_base_statistics(vec, separator='\n'):
    """report_base_statistics."""
    from collections import Counter
    c = Counter(vec)
    msg = ''
    for k in c:
        msg += "class: %s count:%d (%0.2f)%s" % (
            k, c[k], c[k] / float(len(vec)), separator)
    return msg


def save_output(text=None, output_dir_path=None, out_file_name=None):
    """save_output."""
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    full_out_file_name = os.path.join(output_dir_path, out_file_name)
    with open(full_out_file_name, 'w') as f:
        for line in text:
            f.write("%s\n" % str(line).strip())
    logger.info("Written file: %s (%d lines)" %
                (full_out_file_name, len(text)))
