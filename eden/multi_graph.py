#!/usr/bin/env python

from sklearn.linear_model import SGDClassifier
import itertools
import copy
from eden.graph import Vectorizer

import logging
logger = logging.getLogger(__name__)


class ListVectorizer(Vectorizer):
    """Transform vector labeled, weighted, nested graphs in sparse vectors.

    A list of iterators over graphs and a list of weights are taken in input.
    The returned vector is the linear combination of sparse vectors obtained on each
    corresponding graph.
    """

    def __init__(self,
                 complexity=3,
                 r=None,
                 d=None,
                 min_r=0,
                 min_d=0,
                 nbits=20,
                 normalization=True,
                 inner_normalization=True,
                 n=1,
                 min_n=2):
        """
        Arguments:


        complexity : int
          The complexity of the features extracted.

        r : int
          The maximal radius size.

        d : int
          The maximal distance size.

        min_r : int
          The minimal radius size.

        min_d : int
          The minimal distance size.

        nbits : int
          The number of bits that defines the feature space size: |feature space|=2^nbits.

        normalization : bool
          If set the resulting feature vector will have unit euclidean norm.

        inner_normalization : bool
          If set the feature vector for a specific combination of the radius and
          distance size will have unit euclidean norm.
          When used together with the 'normalization' flag it will be applied first and
          then the resulting feature vector will be normalized.

        n : int
          The maximal number of clusters used to discretized label vectors.

        min:n : int
          The minimal number of clusters used to discretized label vectors.
        """
        self.vectorizer = Vectorizer(complexity=complexity,
                                     r=r,
                                     d=d,
                                     min_r=min_r,
                                     min_d=min_d,
                                     nbits=nbits,
                                     normalization=normalization,
                                     inner_normalization=inner_normalization,
                                     n=n,
                                     min_n=min_n)
        self.vectorizers = list()

    def fit(self, graphs_iterators_list):
        """
        Constructs an approximate explicit mapping of a kernel function on the data
        stored in the nodes of the graphs.

        Arguments:

        graphs_iterators_list : list of iterators over networkx graphs.
          The data.
        """
        for i, graphs in enumerate(graphs_iterators_list):
            self.vectorizers.append(copy.copy(self.vectorizer))
            self.vectorizers[i].fit(graphs)

    def fit_transform(self, graphs_iterators_list, weights=list()):
        """
        Arguments:

        graphs_iterators_list : list of iterators over networkx graphs.
          The data.

        weights : list of positive real values.
          Weights for the linear combination of sparse vectors obtained on each iterated tuple of graphs.
        """
        graphs_iterators_list_fit, graphs_iterators_list_transf = itertools.tee(graphs_iterators_list)
        self.fit(graphs_iterators_list_fit)
        return self.transform(graphs_iterators_list_transf)

    def transform(self, graphs_iterators_list, weights=list()):
        """
        Transforms a list of networkx graphs into a Numpy csr sparse matrix
        ( Compressed Sparse Row matrix ).

        Arguments:

        graphs_iterators_list : list of iterators over networkx graphs.
          The data.

        weights : list of positive real values.
          Weights for the linear combination of sparse vectors obtained on each iterated tuple of graphs.
        """
        # if no weights are provided then assume unitary weight
        if len(weights) == 0:
            weights = [1] * len(graphs_iterators_list)
        assert(len(graphs_iterators_list) == len(weights)), 'ERROR: weights size is different than iterators size.'
        assert(len(filter(lambda x: x < 0, weights)) == 0), 'ERROR: weight list contains negative values.'
        for i, graphs in enumerate(graphs_iterators_list):
            if len(self.vectorizers) == 0:
                data_matrix_curr = self.vectorizer.transform(graphs)
            else:
                data_matrix_curr = self.vectorizers[i].transform(graphs)
            if i == 0:
                data_matrix = data_matrix_curr * weights[i]
            else:
                data_matrix = data_matrix + data_matrix_curr * weights[i]
        return data_matrix

    def similarity(self, graphs_iterators_list, ref_instance=None, weights=list()):
        """
        This is a generator.
        """
        self._reference_vec = self._convert_dict_to_sparse_matrix(
            self._transform(0, ref_instance))

        # if no weights are provided then assume unitary weight
        if len(weights) == 0:
            weights = [1] * len(graphs_iterators_list)
        assert(len(graphs_iterators_list) == len(weights)
               ), 'ERROR: weights count is different than iterators count.'
        assert(len(filter(lambda x: x < 0, weights)) ==
               0), 'ERROR: weight list contains negative values.'
        try:
            while True:
                graphs = [G_iterator.next() for G_iterator in graphs_iterators_list]
                yield self._similarity(graphs, weights)
        except StopIteration:
            return

    def _similarity(self, graphs, weights=list()):
        # extract feature vector
        for i, graph in enumerate(graphs):
            x_curr = self.vectorizer._convert_dict_to_sparse_matrix(
                self.vectorizer._transform(0, graph))
            if i == 0:
                x = x_curr * weights[i]
            else:
                x = x + x_curr * weights[i]
        res = self._reference_vec.dot(x.T).todense()
        prediction = res[0, 0]
        return prediction

    def predict(self, graphs_iterators_list, estimator=SGDClassifier(), weights=list()):
        """
        Purpose:
        ----------
        It outputs the estimator prediction of the vectorized graph.

        Arguments:

        estimator : scikit-learn predictor trained on data sampled from the same distribution.
          If None the vertex weigths are by default 1.
        """
        self.estimator = estimator
        # if no weights are provided then assume unitary weight
        if len(weights) == 0:
            weights = [1] * len(graphs_iterators_list)
        assert(len(graphs_iterators_list) == len(weights)), 'ERROR: weights count is different than iterators count.'
        assert(len(filter(lambda x: x < 0, weights)) == 0), 'ERROR: weight list contains negative values.'
        try:
            while True:
                graphs = [G_iterator.next() for G_iterator in graphs_iterators_list]
                yield self._predict(graphs, weights)
        except StopIteration:
            return

    def _predict(self, graphs, weights=list()):
        # extract feature vector
        for i, graph in enumerate(graphs):
            x_curr = self.vectorizer._convert_dict_to_sparse_matrix(self.vectorizer._transform(0, graph))
            if i == 0:
                x = x_curr * weights[i]
            else:
                x = x + x_curr * weights[i]
        margins = self.estimator.decision_function(x)
        prediction = margins[0]
        return prediction
