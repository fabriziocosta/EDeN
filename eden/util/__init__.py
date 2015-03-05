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

def is_iterable(test):
    if hasattr(test, '__iter__'):
        return True
    else:
        return False

def describe(X):
    print 'Instances: %d ; Features: %d with an avg of %d features per instance' % (X.shape[0], X.shape[1],  X.getnnz()/X.shape[0])


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


def fit_estimator(positive_data_matrix=None, negative_data_matrix=None, target=None, cv=10, n_jobs=-1):
    assert(
        positive_data_matrix is not None), 'ERROR: expecting non null positive_data_matrix'
    if target is None and negative_data_matrix is not None:
        yp = [1] * positive_data_matrix.shape[0]
        yn = [-1] * negative_data_matrix.shape[0]
        y = np.array(yp + yn)
        X = vstack([positive_data_matrix, negative_data_matrix], format="csr")
    if target is not None:
        X = positive_data_matrix
        y = target

    predictor = SGDClassifier(class_weight='auto', shuffle=True, n_jobs=n_jobs)
    # hyperparameter optimization
    param_dist = {"n_iter": randint(5, 100),
                  "power_t": uniform(0.1),
                  "alpha": uniform(1e-08, 1e-03),
                  "eta0": uniform(1e-03, 10),
                  "penalty": ["l1", "l2", "elasticnet"],
                  "learning_rate": ["invscaling", "constant", "optimal"]}
    scoring = 'roc_auc'
    n_iter_search = 20
    random_search = RandomizedSearchCV(
        predictor, param_distributions=param_dist, n_iter=n_iter_search, cv=cv, scoring=scoring, n_jobs=n_jobs)
    random_search.fit(X, y)
    optpredictor = SGDClassifier(
        class_weight='auto', shuffle=True, n_jobs=n_jobs, **random_search.best_params_)
    # fit the predictor on all available data
    optpredictor.fit(X, y)

    print 'Classifier:'
    print optpredictor
    print '-' * 80

    print 'Predictive performance:'
    # assess the generalization capacity of the model via a 10-fold cross
    # validation
    for scoring in ['accuracy', 'precision', 'recall', 'f1', 'average_precision', 'roc_auc']:
        scores = cross_validation.cross_val_score(
            optpredictor, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
        print('%20s: %.3f +- %.3f' %
              (scoring, np.mean(scores), np.std(scores)))
    print '-' * 80

    return optpredictor


def fit(iterable_pos_train, iterable_neg_train, vectorizer, n_jobs=1, cv=10):
    X_pos_train = vectorizer.transform( iterable_pos_train, n_jobs=n_jobs )
    X_neg_train = vectorizer.transform( iterable_neg_train, n_jobs=n_jobs )
    #optimize hyperparameters classifier
    optpredictor = fit_estimator(positive_data_matrix=X_pos_train, negative_data_matrix=X_neg_train, cv=cv, n_jobs=n_jobs)
    return optpredictor


def estimate(iterable_pos_test, iterable_neg_test, estimator, vectorizer, n_jobs=1):
    X_pos_test = vectorizer.transform( iterable_pos_test, n_jobs=n_jobs )
    X_neg_test = vectorizer.transform( iterable_neg_test, n_jobs=n_jobs )
    yp =  [1] * X_pos_test.shape[0]
    yn = [-1] * X_neg_test.shape[0]
    y = np.array(yp + yn)
    X_test = vstack( [X_pos_test,X_neg_test] , format = "csr")
    print 'Test set'
    describe(X_test)
    print '-'*80
    print 'Test Estimate'
    predictions=estimator.predict(X_test)
    margins=estimator.decision_function(X_test)
    print classification_report(y, predictions)
    roc = roc_auc_score(y, margins)
    print 'ROC: %.3f' % roc
    apr = average_precision_score(y, margins)
    print 'APR: %.3f'% apr
    return roc, apr
    

def self_training(iterable_pos, iterable_neg, vectorizer=None, pos2neg_ratio=0.1, num_iterations=2,  threshold=0,  mode='less_than', n_jobs=-1):
    def select_ids(predictions, threshold, mode, desired_num_neg):
        ids = list()
        for i, prediction in enumerate(predictions):
            if mode == 'less_then':
                comparison = prediction < float(threshold)
            else:
                comparison = prediction > float(threshold)
            if comparison:
                ids.append(i)
        #keep a random sample of num_neg difficult cases
        random.shuffle(ids)
        ids = ids[:desired_num_neg]
        return ids

    Xpos = vectorizer.transform( iterable_pos, n_jobs=n_jobs )
    print 'Positives:'
    describe(Xpos)
    #select a fraction for the negatives
    num_pos = Xpos.shape[0]
    desired_num_neg = int(float(num_pos) * pos2neg_ratio)
    #select the initial ids for the negatives as the first num_neg
    ids = range(desired_num_neg)
    #iterate: select negatives and create a model using postives + selected negatives 
    for i in range(num_iterations):
        print 'Iteration: %d/%d'%(i+1,num_iterations)
        #select only a fraction of the negatives
        iterable_neg, iterable_neg_copy1, iterable_neg_copy2 = tee(iterable_neg,3)
        Xneg = vectorizer.transform( selection_iterator(iterable_neg_copy1,ids), n_jobs=n_jobs )
        print 'Negatives:'
        describe(Xneg)
        #fit the estimator on all positives and selected negatives
        from eden.util import fit_estimator
        estimator = fit_estimator( positive_data_matrix=Xpos, negative_data_matrix=Xneg, cv=10 )
        if i < num_iterations -1:
            #use the estimator to select the next batch of negatives
            predictions = vectorizer.predict(iterable_neg_copy2, estimator)
            ids = select_ids(predictions, threshold, mode, desired_num_neg)
    return estimator


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
