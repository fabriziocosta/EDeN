import numpy as np
from scipy import io
from sklearn.externals import joblib
import multiprocessing as mp
import requests
import os
import sys
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn import cross_validation
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from scipy.stats import randint
from scipy.stats import uniform
from scipy.sparse import vstack
from itertools import tee
import random
from time import time
import logging.handlers
from eden import apply_async
import json

import logging
logger = logging.getLogger(__name__)


def print_json(text):
    print json.dumps(json.loads(text), sort_keys=True, indent=4, separators=(',', ': '))


def configure_logging(logger, verbosity=0, filename=None):
    """Utility to configure the logging aspects. If filename is None then no info is stored in files.
    If filename is not None then everything that is logged is dumped to file (including program traces).
    Verbosity is an int that can take values: 0 -> warning, 1 -> info, >=2 -> debug.
    All levels are displayed on stdout, not on stderr. Please use exceptions and asserts
    to output on stderr."""

    logger.propagate = False
    logger.handlers = []
    log_level = logging.WARNING
    if verbosity == 1:
        log_level = logging.INFO
    elif verbosity >= 2:
        log_level = logging.DEBUG
    logger.setLevel(logging.DEBUG)
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
        fh = logging.handlers.RotatingFileHandler(filename=filename, maxBytes=10000000, backupCount=10)
        fh.setLevel(logging.DEBUG)
        # create formatter
        fformatter = logging.Formatter('%(asctime)s | %(levelname)-6s | %(name)10s | %(filename)10s |\
   %(lineno)4s | %(message)s')
        # add formatter to fh
        fh.setFormatter(fformatter)
        # add handlers to logger
        logger.addHandler(fh)


def _serialize_list(items, separator='_'):
    if is_iterable(items):
        return str(separator.join([str(item) for item in items]))
    else:
        return str(items)


def serialize_dict(the_dict, full=True, offset='small'):
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
    """
    Abstract read function. EDeN can accept a URL, a file path and a python list.
    In all cases an iteratatable object should be returned.
    """
    if hasattr(uri, '__iter__'):
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
    if hasattr(test, '__iter__'):
        return True
    else:
        return False


def compute_intervals(size=None, n_blocks=None, block_size=None):
    if block_size is not None:
        n_blocks = int(size / block_size)
    # if n_blocks is the same or larger than size then decrease n_blocks so to have at least
    # 10 instances per block
    if n_blocks >= size:
        n_blocks = size / 10
    if n_blocks < 1:
        n_blocks = 1
    block_size = size / n_blocks
    intervals = [(s * block_size, (s + 1) * block_size)
                 for s in range(n_blocks)]
    # handle the remainder
    remainder = size % n_blocks
    # create an additional block for the remainder
    if remainder > 1:
        intervals += [(n_blocks * block_size,
                       n_blocks * block_size + remainder)]
    # if the remainder is one, just add it to the last block
    elif remainder == 1:
        s, e = intervals[-1]
        intervals[-1] = (s, e + 1)
    return intervals


def serial_pre_process(iterable, pre_processor=None, pre_processor_args=None):
    if pre_processor_args:
        return list(pre_processor(iterable, **pre_processor_args))
    else:
        return list(pre_processor(iterable))


def multiprocess_pre_process(iterable,
                             pre_processor=None,
                             pre_processor_args=None,
                             n_blocks=5,
                             block_size=None,
                             n_jobs=8):
    iterable = list(iterable)
    size = len(iterable)
    intervals = compute_intervals(
        size=size, n_blocks=n_blocks, block_size=block_size)
    if n_jobs == -1:
        pool = mp.Pool()
    else:
        pool = mp.Pool(n_jobs)
    results = [apply_async(pool, serial_pre_process,
                           args=(iterable[start:end], pre_processor, pre_processor_args))
               for start, end in intervals]
    output = [p.get() for p in results]
    pool.close()
    pool.join()
    return_list = []
    for items in output:
        for item in items:
            return_list.append(item)
    return return_list


def mp_pre_process(iterable,
                   pre_processor=None,
                   pre_processor_args=None,
                   n_blocks=5,
                   block_size=None,
                   n_jobs=8):
    if n_jobs == 1:
        return pre_processor(iterable, **pre_processor_args)
    else:
        return multiprocess_pre_process(iterable,
                                        pre_processor=pre_processor,
                                        pre_processor_args=pre_processor_args,
                                        n_blocks=n_blocks,
                                        block_size=block_size,
                                        n_jobs=n_jobs)


def serial_vectorize(iterators,
                     vectorizer=None,
                     pre_processor=None,
                     pre_processor_args=None,
                     fit_flag=False):
    if pre_processor is not None:
        if pre_processor_args is not None:
            graphs = pre_processor(iterators, **pre_processor_args)
        else:
            graphs = pre_processor(iterators)
    else:
        graphs = iterators
    if fit_flag:
        data_matrix = vectorizer.fit_transform(graphs)
    else:
        data_matrix = vectorizer.transform(graphs)
    return data_matrix


def multiprocess_vectorize(iterators,
                           vectorizer=None,
                           pre_processor=None,
                           pre_processor_args=None,
                           fit_flag=False,
                           n_blocks=5,
                           block_size=None,
                           n_jobs=8):
    iterators = list(iterators)
    # fitting happens in a serial fashion
    if fit_flag:
        if pre_processor is not None:
            if pre_processor_args is not None:
                graphs = pre_processor(iterators, **pre_processor_args)
            else:
                graphs = pre_processor(iterators)
        else:
            graphs = iterators
        vectorizer.fit(graphs)
    size = len(iterators)
    intervals = compute_intervals(size=size, n_blocks=n_blocks, block_size=block_size)
    if n_jobs == -1:
        pool = mp.Pool()
    else:
        pool = mp.Pool(n_jobs)
    results = [apply_async(pool, serial_vectorize,
                           args=(iterators[start:end],
                                 vectorizer,
                                 pre_processor,
                                 pre_processor_args,
                                 False))
               for start, end in intervals]
    output = [p.get() for p in results]
    pool.close()
    pool.join()
    data_matrix = vstack(output, format="csr")
    return data_matrix


def vectorize(iterators,
              vectorizer=None,
              pre_processor=None,
              pre_processor_args=None,
              fit_flag=False,
              n_blocks=5,
              block_size=None,
              n_jobs=8):
    if n_jobs == 1:
        return serial_vectorize(iterators,
                                vectorizer=vectorizer,
                                pre_processor=pre_processor,
                                pre_processor_args=pre_processor_args,
                                fit_flag=fit_flag)
    else:
        return multiprocess_vectorize(iterators,
                                      vectorizer=vectorizer,
                                      pre_processor=pre_processor,
                                      pre_processor_args=pre_processor_args,
                                      fit_flag=fit_flag,
                                      n_blocks=n_blocks,
                                      block_size=block_size,
                                      n_jobs=n_jobs)


def describe(data_matrix):
    return 'Instances: %d ; Features: %d with an avg of %d features per instance' % \
        (data_matrix.shape[0], data_matrix.shape[1],
         data_matrix.getnnz() / data_matrix.shape[0])


def iterator_size(iterable):
    iterable, iterable_ = tee(iterable)
    return sum(1 for x in iterable_)


def random_bipartition(int_range, relative_size=.7, random_state=1):
    random.seed(random_state)
    ids = range(int_range)
    random.shuffle(ids)
    split_point = int(int_range * relative_size)
    return ids[:split_point], ids[split_point:]


def selection_iterator(iterable, ids):
    '''Given an iterable and a list of ids (zero based) yield only the items whose id matches'''
    ids = sorted(ids)
    counter = 0
    for id, item in enumerate(iterable):
        if id == ids[counter]:
            yield item
            counter += 1
            if counter == len(ids):
                break


def random_bipartition_iter(iterable, relative_size=.5, random_state=1):
    size_iterable, iterable1, iterable2 = tee(iterable, 3)
    size = iterator_size(size_iterable)
    part1_ids, part2_ids = random_bipartition(
        size, relative_size=relative_size, random_state=random_state)
    part1_iterable = selection_iterator(iterable1, part1_ids)
    part2_iterable = selection_iterator(iterable2, part2_ids)
    return part1_iterable, part2_iterable


def join_pre_processes(iterable, pre_processes=None, weights=None):
    graphs_list = list()
    assert(len(weights) == len(pre_processes)), 'Different lengths'
    # NOTE: we have to duplicate the sequences iterator if we want to use
    # different modifiers in parallel
    iterables = tee(iterable, len(pre_processes))
    for pre_process_item, iterable_item in zip(pre_processes, iterables):
        graphs_list.append(pre_process_item(iterable_item))
    return (graphs_list, weights)


def make_data_matrix(positive_data_matrix=None, negative_data_matrix=None, target=None):
    assert(positive_data_matrix is not None), 'ERROR: expecting non null positive_data_matrix'
    if negative_data_matrix is None:
        negative_data_matrix = positive_data_matrix.multiply(-1)
    if target is None and negative_data_matrix is not None:
        yp = [1] * positive_data_matrix.shape[0]
        yn = [-1] * negative_data_matrix.shape[0]
        y = np.array(yp + yn)
        data_matrix = vstack(
            [positive_data_matrix, negative_data_matrix], format="csr")
    if target is not None:
        data_matrix = positive_data_matrix
        y = target
    return data_matrix, y


def fit_estimator(estimator,
                  positive_data_matrix=None,
                  negative_data_matrix=None,
                  target=None,
                  cv=10,
                  n_jobs=-1,
                  n_iter_search=40,
                  random_state=1):
    # hyperparameter optimization
    param_dist = {"n_iter": randint(5, 100),
                  "power_t": uniform(0.1),
                  "alpha": uniform(1e-08, 1e-03),
                  "eta0": uniform(1e-03, 1),
                  "penalty": ["l1", "l2", "elasticnet"],
                  "learning_rate": ["invscaling", "constant", "optimal"]}
    scoring = 'roc_auc'
    n_iter_search = n_iter_search
    random_search = RandomizedSearchCV(estimator,
                                       param_distributions=param_dist,
                                       n_iter=n_iter_search,
                                       cv=cv,
                                       scoring=scoring,
                                       n_jobs=n_jobs,
                                       random_state=random_state,
                                       refit=True)
    X, y = make_data_matrix(positive_data_matrix=positive_data_matrix,
                            negative_data_matrix=negative_data_matrix,
                            target=target)
    random_search.fit(X, y)

    logger.debug('\nClassifier:')
    logger.debug('%s' % random_search.best_estimator_)
    logger.debug('\nPredictive performance:')
    # assess the generalization capacity of the model via a 10-fold cross
    # validation
    for scoring in ['accuracy', 'precision', 'recall', 'f1', 'average_precision', 'roc_auc']:
        scores = cross_validation.cross_val_score(random_search.best_estimator_, X, y, cv=cv,
                                                  scoring=scoring, n_jobs=n_jobs)
        logger.debug('%20s: %.3f +- %.3f' %
                     (scoring, np.mean(scores), np.std(scores)))

    return random_search.best_estimator_


def fit(iterable_pos, iterable_neg=None,
        vectorizer=None,
        estimator=None,
        fit_flag=False,
        n_jobs=-1,
        cv=10,
        n_iter_search=1,
        random_state=1,
        n_blocks=5,
        block_size=None):
    estimator = SGDClassifier(average=True, class_weight='balanced', shuffle=True, random_state=random_state)
    start = time()
    positive_data_matrix = vectorize(iterable_pos,
                                     vectorizer=vectorizer,
                                     fit_flag=fit_flag,
                                     n_blocks=n_blocks,
                                     block_size=block_size,
                                     n_jobs=n_jobs)
    logger.debug('Positive data: %s' % describe(positive_data_matrix))
    if iterable_neg:
        negative_data_matrix = vectorize(iterable_neg,
                                         vectorizer=vectorizer,
                                         fit_flag=False,
                                         n_blocks=n_blocks,
                                         block_size=block_size,
                                         n_jobs=n_jobs)
        logger.debug('Negative data: %s' % describe(negative_data_matrix))
    else:
        negative_data_matrix = None
    if n_iter_search <= 1:
        X, y = make_data_matrix(positive_data_matrix=positive_data_matrix,
                                negative_data_matrix=negative_data_matrix)
        estimator.fit(X, y)
    else:
        # optimize hyper parameters classifier
        estimator = fit_estimator(estimator,
                                  positive_data_matrix=positive_data_matrix,
                                  negative_data_matrix=negative_data_matrix,
                                  cv=cv,
                                  n_jobs=n_jobs,
                                  n_iter_search=n_iter_search,
                                  random_state=random_state)
    logger.debug('Elapsed time: %.1f secs' % (time() - start))
    return estimator


def estimate_model(positive_data_matrix=None,
                   negative_data_matrix=None,
                   target=None,
                   estimator=None,
                   n_jobs=4):
    X, y = make_data_matrix(positive_data_matrix=positive_data_matrix,
                            negative_data_matrix=negative_data_matrix,
                            target=target)
    logger.info('Test set')
    logger.info(describe(X))
    logger.info('-' * 80)
    logger.info('Test Estimate')
    predictions = estimator.predict(X)
    margins = estimator.decision_function(X)
    logger.info(classification_report(y, predictions))
    apr = average_precision_score(y, margins)
    logger.info('APR: %.3f' % apr)
    roc = roc_auc_score(y, margins)
    logger.info('ROC: %.3f' % roc)

    logger.info('Cross-validated estimate')
    for scoring in ['accuracy', 'precision', 'recall', 'f1', 'average_precision', 'roc_auc']:
        scores = cross_validation.cross_val_score(estimator, X, y, cv=5,
                                                  scoring=scoring, n_jobs=n_jobs)
        logger.info('%20s: %.3f +- %.3f' % (scoring, np.mean(scores), np.std(scores)))

    return roc, apr


def estimate(iterable_pos=None,
             iterable_neg=None,
             estimator=None,
             vectorizer=None,
             n_blocks=5,
             block_size=None,
             n_jobs=4):
    positive_data_matrix = vectorize(iterable_pos,
                                     vectorizer=vectorizer,
                                     n_blocks=n_blocks,
                                     block_size=block_size,
                                     n_jobs=n_jobs)
    negative_data_matrix = vectorize(iterable_neg,
                                     vectorizer=vectorizer,
                                     n_blocks=n_blocks,
                                     block_size=block_size,
                                     n_jobs=n_jobs)
    return estimate_model(positive_data_matrix=positive_data_matrix,
                          negative_data_matrix=negative_data_matrix,
                          estimator=estimator,
                          n_jobs=n_jobs)


def predict(iterable=None,
            estimator=None,
            vectorizer=None,
            mode='decision_function',
            n_blocks=5,
            block_size=None,
            n_jobs=4):
    data_matrix = vectorize(iterable,
                            vectorizer=vectorizer,
                            n_blocks=n_blocks,
                            block_size=block_size,
                            n_jobs=n_jobs)
    if mode == 'decision_function':
        out = estimator.decision_function(data_matrix)
    elif mode == 'predict_proba':
        out = estimator.predict_proba(data_matrix)
    else:
        raise Exception('Unknown mode: %s' % mode)
    return out


def load_target(name):
    """
    Return a numpy array of integers to be used as target vector.

    Parameters
    ----------
    name : string
        A pointer to the data source.

    """

    target = [y.strip() for y in read(name) if y]
    return np.array(target).astype(int)


def store_matrix(matrix='', output_dir_path='', out_file_name='', output_format=''):
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    full_out_file_name = os.path.join(output_dir_path, out_file_name)
    if output_format == "MatrixMarket":
        if len(matrix.shape) == 1:
            raise Exception(
                "'MatrixMarket' format supports only 2D dimensional array and not vectors")
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
                    "Currently 'text' format supports only mono dimensional array and not matrices")
    logger.info("Written file: %s" % full_out_file_name)


def dump(obj, output_dir_path='', out_file_name=''):
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    full_out_file_name = os.path.join(output_dir_path, out_file_name) + ".pkl"
    joblib.dump(obj, full_out_file_name)


def load(output_dir_path='', out_file_name=''):
    full_out_file_name = os.path.join(output_dir_path, out_file_name) + ".pkl"
    obj = joblib.load(full_out_file_name)
    return obj


def report_base_statistics(vec, separator='\n'):
    from collections import Counter
    c = Counter(vec)
    msg = ''
    for k in c:
        msg += "class: %s count:%d (%0.2f)%s" % (k,
                                                 c[k], c[k] / float(len(vec)),
                                                 separator)
    return msg


def save_output(text=None, output_dir_path=None, out_file_name=None):
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    full_out_file_name = os.path.join(output_dir_path, out_file_name)
    with open(full_out_file_name, 'w') as f:
        for line in text:
            f.write("%s\n" % str(line).strip())
    logger.info("Written file: %s (%d lines)" %
                (full_out_file_name, len(text)))
