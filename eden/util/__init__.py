import numpy as np
from scipy import io
from sklearn.externals import joblib
import requests
import os
from sklearn.linear_model import SGDClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn import cross_validation
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from scipy.stats import randint
from scipy.stats import uniform
import numpy as np
from scipy import stats
from scipy.sparse import vstack
from itertools import tee
import random


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


def serial_pre_process(iterable, pre_processor=None, pre_processor_args=None):
    return list(pre_processor(iterable, **pre_processor_args ))


def multiprocess_pre_process(iterable, pre_processor=None, pre_processor_args=None, n_blocks=5, n_jobs=8):
    from eden import apply_async
    iterable = list(iterable)
    import multiprocessing as mp
    size = len(iterable)
    # if n_blocks is the same or larger than size then decrease n_blocks so to have at least 10 instances per block
    if n_blocks >= size:
        n_blocks = size / 10
    # if one block will end up containing a single instance reduce the number of blocks to avoid the case
    if size % n_blocks == 1:
        n_blocks = max(1, n_blocks - 1)
    block_size = size / n_blocks
    reminder = size % n_blocks
    if n_jobs == -1:
        pool = mp.Pool()
    else:
        pool = mp.Pool(n_jobs)
    intervals = [(s * block_size, (s + 1) * block_size) for s in range(n_blocks)]
    if reminder > 1:
        intervals += [(n_blocks * block_size, n_blocks * block_size + reminder)]
    results = [apply_async(pool, serial_pre_process, args=(iterable[start:end], pre_processor, pre_processor_args)) for start, end in intervals]
    output = [p.get() for p in results]
    pool.close()
    pool.join()
    return_list = []
    for items in output:
        for item in items:
            return_list.append(item)
    return return_list


def mp_pre_process(iterable, pre_processor=None, pre_processor_args=None, n_blocks=5, n_jobs=8):
    if n_jobs == 1:
        return serial_pre_process(iterable, pre_processor=pre_processor, pre_processor_args=pre_processor_args)
    else:
        return multiprocess_pre_process(iterable, pre_processor=pre_processor, pre_processor_args=pre_processor_args, n_blocks=n_blocks, n_jobs=n_jobs)


def describe(X):
    print 'Instances: %d ; Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1],  X.getnnz() / X.shape[0])


def size(iterable):
    return sum(1 for x in iterable)


def random_bipartition(int_range, relative_size=.7):
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


def random_bipartition_iter(iterable, relative_size=.5):
    size_iterable, iterable1, iterable2 = tee(iterable, 3)
    the_size = size(size_iterable)
    part1_ids, part2_ids = random_bipartition(the_size, relative_size=relative_size)
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
    if target is None and negative_data_matrix is not None:
        yp = [1] * positive_data_matrix.shape[0]
        yn = [-1] * negative_data_matrix.shape[0]
        y = np.array(yp + yn)
        X = vstack([positive_data_matrix, negative_data_matrix], format="csr")
    if target is not None:
        X = positive_data_matrix
        y = target
    return X, y


def fit_estimator(estimator, positive_data_matrix=None, negative_data_matrix=None, target=None, cv=10, n_jobs=-1, n_iter_search=40, random_state=1):
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

    print 'Classifier:'
    print random_search.best_estimator_
    print '-' * 80
    print 'Predictive performance:'
    # assess the generalization capacity of the model via a 10-fold cross
    # validation
    for scoring in ['accuracy', 'precision', 'recall', 'f1', 'average_precision', 'roc_auc']:
        scores = cross_validation.cross_val_score(random_search.best_estimator_, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
        print('%20s: %.3f +- %.3f' % (scoring, np.mean(scores), np.std(scores)))
    print '-' * 80
    return random_search.best_estimator_


def fit(iterable_pos, iterable_neg, vectorizer, n_jobs=-1, cv=10, n_iter_search=20, random_state=1, n_blocks=5):
    estimator = SGDClassifier(average=True, class_weight='auto', shuffle=True, n_jobs=n_jobs)
    from eden import vectorize
    X_pos = vectorize(iterable_pos, vectorizer=vectorizer, n_blocks=n_blocks, n_jobs=n_jobs)
    X_neg = vectorize(iterable_neg, vectorizer=vectorizer, n_blocks=n_blocks, n_jobs=n_jobs)
    if n_iter_search <= 1:
        X, y = make_data_matrix(positive_data_matrix=X_pos,
                                negative_data_matrix=X_neg)
        estimator.fit(X, y)
    else:
        # optimize hyperparameters classifier
        estimator = fit_estimator(estimator,
                                  positive_data_matrix=X_pos,
                                  negative_data_matrix=X_neg,
                                  cv=cv,
                                  n_jobs=n_jobs,
                                  n_iter_search=n_iter_search,
                                  random_state=random_state)
    return estimator


def estimate_estimator(positive_data_matrix=None, negative_data_matrix=None, target=None, estimator=None):
    X, y = make_data_matrix(positive_data_matrix=positive_data_matrix, negative_data_matrix=negative_data_matrix, target=target)
    print 'Test set'
    describe(X)
    print '-' * 80
    print 'Test Estimate'
    predictions = estimator.predict(X)
    margins = estimator.decision_function(X)
    print classification_report(y, predictions)
    apr = average_precision_score(y, margins)
    print 'APR: %.3f' % apr
    roc = roc_auc_score(y, margins)
    print 'ROC: %.3f' % roc
    return apr, roc


def estimate(iterable_pos, iterable_neg, estimator, vectorizer, n_blocks=5, n_jobs=4):
    from eden import vectorize
    X_pos = vectorize(iterable_pos, vectorizer=vectorizer, n_blocks=n_blocks, n_jobs=n_jobs)
    X_neg = vectorize(iterable_neg, vectorizer=vectorizer, n_blocks=n_blocks, n_jobs=n_jobs)
    return estimate_estimator(positive_data_matrix=X_pos, negative_data_matrix=X_neg, estimator=estimator)


def load_target(name):
    """
    Return a numpy array of integers to be used as target vector.

    Parameters
    ----------
    name : string
        A pointer to the data source.

    """

    Y = [y.strip() for y in read(name) if y]
    return np.array(Y).astype(int)


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
                #data_str = map(str, matrix)
                # f.write('\n'.join(data_str))
            else:
                raise Exception(
                    "Currently 'text' format supports only mono dimensional array and not matrices")


def dump(obj, output_dir_path='', out_file_name=''):
    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)
    full_out_file_name = os.path.join(output_dir_path, out_file_name) + ".pkl"
    joblib.dump(obj, full_out_file_name)


def load(output_dir_path='', out_file_name=''):
    full_out_file_name = os.path.join(output_dir_path, out_file_name) + ".pkl"
    obj = joblib.load(full_out_file_name)
    return obj
