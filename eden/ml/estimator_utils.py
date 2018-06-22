#!/usr/bin/env python
"""Provides scikit interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from eden.util import timeit
import random
from toolz.sandbox.core import unzip
from collections import Counter
from toolz.curried import first, second, groupby
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
import pylab as plt
from eden.display import plot_confusion_matrices
from eden.display import plot_aucs
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger()


@timeit
def process_vec_info(g, n_clusters=8):
    """process_vec_info."""
    # extract node vec information and make np data matrix
    data_matrix = np.array([g.node[u]['vec'] for u in g.nodes()])
    # cluster with kmeans
    clu = MiniBatchKMeans(n_clusters=n_clusters, n_init=10)
    clu.fit(data_matrix)
    preds = clu.predict(data_matrix)
    vecs = clu.transform(data_matrix)
    vecs = 1 / (1 + vecs)
    # replace node information
    graph = g.copy()
    for u in graph.nodes():
        graph.node[u]['label'] = str(preds[u])
        graph.node[u]['vec'] = list(vecs[u])
    return graph


def paired_shuffle(iterable1, iterable2):
    """paired_shuffle."""
    i1i2 = list(zip(iterable1, iterable2))
    random.shuffle(i1i2)
    i1, i2 = unzip(i1i2)
    return list(i1), list(i2)


@timeit
def subsample(graphs, targets, subsample_size=100):
    """subsample."""
    tg = zip(targets, graphs)
    num_classes = len(set(targets))
    class_graphs = groupby(lambda x: first(x), tg)
    subgraphs = []
    subtargets = []
    for y in class_graphs:
        class_subgraphs = class_graphs[y][:subsample_size / num_classes]
        class_subgraphs = [second(x) for x in class_subgraphs]
        subgraphs += class_subgraphs
        subtargets += [y] * len(class_subgraphs)
    subgraphs, subtargets = paired_shuffle(subgraphs, subtargets)
    return list(subgraphs), list(subtargets)


@timeit
def balance(graphs, targets, estimator, ratio=2):
    """balance."""
    class_counts = Counter(targets)
    majority_class = None
    max_count = 0
    minority_class = None
    min_count = 1e6
    for class_key in class_counts:
        if max_count < class_counts[class_key]:
            majority_class = class_key
            max_count = class_counts[class_key]
        if min_count > class_counts[class_key]:
            minority_class = class_key
            min_count = class_counts[class_key]

    desired_size = int(min_count * ratio)

    tg = zip(targets, graphs)
    class_graphs = groupby(lambda x: first(x), tg)
    maj_graphs = [second(x) for x in class_graphs[majority_class]]
    min_graphs = [second(x) for x in class_graphs[minority_class]]

    if estimator:
        # select only the instances in the majority class that
        # have a small margin
        preds = estimator.decision_function(maj_graphs)
    else:
        # select at random
        preds = [random.random() for i in range(len(maj_graphs))]
    preds = [abs(pred) for pred in preds]
    pred_graphs = sorted(zip(preds, maj_graphs))[:desired_size]
    maj_graphs = [g for p, g in pred_graphs]

    bal_graphs = min_graphs + maj_graphs
    bal_pos = [minority_class] * len(min_graphs)
    bal_neg = [majority_class] * len(maj_graphs)
    bal_targets = bal_pos + bal_neg

    return paired_shuffle(bal_graphs, bal_targets)


def make_train_test_sets(pos_graphs, neg_graphs,
                         test_proportion=.3, random_state=2):
    """make_train_test_sets."""
    random.seed(random_state)
    random.shuffle(pos_graphs)
    random.shuffle(neg_graphs)
    pos_dim = len(pos_graphs)
    neg_dim = len(neg_graphs)
    tr_pos_graphs = pos_graphs[:-int(pos_dim * test_proportion)]
    te_pos_graphs = pos_graphs[-int(pos_dim * test_proportion):]
    tr_neg_graphs = neg_graphs[:-int(neg_dim * test_proportion)]
    te_neg_graphs = neg_graphs[-int(neg_dim * test_proportion):]
    tr_graphs = tr_pos_graphs + tr_neg_graphs
    te_graphs = te_pos_graphs + te_neg_graphs
    tr_targets = [1] * len(tr_pos_graphs) + [0] * len(tr_neg_graphs)
    te_targets = [1] * len(te_pos_graphs) + [0] * len(te_neg_graphs)
    tr_graphs, tr_targets = paired_shuffle(tr_graphs, tr_targets)
    te_graphs, te_targets = paired_shuffle(te_graphs, te_targets)
    return (tr_graphs, np.array(tr_targets)), (te_graphs, np.array(te_targets))


@timeit
def estimate_predictive_performance(x_y,
                                    estimator=None,
                                    n_splits=10,
                                    random_state=1):
    """estimate_predictive_performance."""
    x, y = x_y
    cv = ShuffleSplit(n_splits=n_splits,
                      test_size=0.3,
                      random_state=random_state)
    scoring = make_scorer(average_precision_score)
    scores = cross_val_score(estimator, x, y, cv=cv, scoring=scoring)
    return scores


def output_avg_and_std(iterable):
    """output_avg_and_std."""
    print(('score: %.2f +-%.2f' % (np.mean(iterable), np.std(iterable))))
    return iterable


@timeit
def perf(y_true, y_pred, y_score):
    """perf."""
    print('Accuracy: %.2f' % accuracy_score(y_true, y_pred))
    print(' AUC ROC: %.2f' % roc_auc_score(y_true, y_score))
    print('  AUC AP: %.2f' % average_precision_score(y_true, y_score))
    print()
    print('Classification Report:')
    print(classification_report(y_true, y_pred))
    print()
    plot_confusion_matrices(y_true, y_pred, size=int(len(set(y_true)) * 2.5))
    print()
    plot_aucs(y_true, y_score, size=10)


def compute_stats(scores):
    """compute_stats."""
    median = np.percentile(scores, 50, axis=1)
    low = np.percentile(scores, 25, axis=1)
    high = np.percentile(scores, 75, axis=1)
    low10 = np.percentile(scores, 10, axis=1)
    high90 = np.percentile(scores, 90, axis=1)
    return median, low, high, low10, high90


def plot_stats(x=None, y=None, label=None, color='navy'):
    """plot_stats."""
    y = np.array(y)
    y0 = y[0]
    y1 = y[1]
    y2 = y[2]
    y3 = y[3]
    y4 = y[4]
    plt.fill_between(x, y3, y4, color=color, alpha=0.08)
    plt.fill_between(x, y1, y2, color=color, alpha=0.08)
    plt.plot(x, y0, '-', lw=2, color=color, label=label)
    plt.plot(x, y0,
             linestyle='None',
             markerfacecolor='white',
             markeredgecolor=color,
             marker='o',
             markeredgewidth=2,
             markersize=8)


def plot_learning_curve(train_sizes, train_scores, test_scores):
    """plot_learning_curve."""
    plt.figure(figsize=(15, 5))
    plt.title('Learning Curve')
    plt.xlabel("Training examples")
    plt.ylabel("AUC ROC")
    tr_ys = compute_stats(train_scores)
    te_ys = compute_stats(test_scores)
    plot_stats(train_sizes, tr_ys,
               label='Training score',
               color='navy')
    plot_stats(train_sizes, te_ys,
               label='Cross-validation score',
               color='orange')
    plt.grid(linestyle=":")
    plt.legend(loc="best")
    plt.show()
