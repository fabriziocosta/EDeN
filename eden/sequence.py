#!/usr/bin/env python
"""Provides vectorization of sequences."""

from collections import defaultdict
import numpy as np
import math
from scipy.sparse import csr_matrix
from eden import fast_hash_vec, fast_hash_2, fast_hash_4
from eden import AbstractVectorizer
import logging
logger = logging.getLogger(__name__)


class Vectorizer(AbstractVectorizer):
    """Transform real strings into sparse vectors.

    >>> # vectorize a sequence using default parameters
    >>> seqstrings = ['A']
    >>> str(Vectorizer().transform(seqstrings))
    '  (0, 930612)\\t1.0'

    >>> # vectorize a sequence using weights
    >>> weighttups = [('ID1', 'A', [0.5])]
    >>> str(Vectorizer().transform(weighttups))
    '  (0, 930612)\\t1.0'

    >>> # vectorize a sequence
    >>> weighttups_ones = [('ID2', 'HA', [1,1])]
    >>> str(Vectorizer(r=1, d=0).transform(weighttups_ones))
    '  (0, 8188)\\t0.57735026919\\n  (0, 304234)\\t0.408248290464\\n  (0, 431837)\\t0.57735026919\\n  (0, 930612)\\t0.408248290464'

    >>> # for comparison vectorize a sequence containing zero weight
    >>> weighttups_zero = [('ID2', 'HA', [1,0])]
    >>> str(Vectorizer(r=1, d=0).transform(weighttups_zero))
    '  (0, 8188)\\t0.57735026919\\n  (0, 304234)\\t0.57735026919\\n  (0, 431837)\\t0.57735026919\\n  (0, 930612)\\t0.0'

    >>> # annotate importance of positions
    >>> str(list(Vectorizer(r=0, d=0).annotate(['GATTACA'])))
    "[(['G', 'A', 'T', 'T', 'A', 'C', 'A'], array([1, 1, 1, 1, 1, 1, 1]))]"
    """

    def __init__(self,
                 complexity=None,
                 r=3,
                 d=3,
                 min_r=0,
                 min_d=0,
                 nbits=20,
                 normalization=True,
                 inner_normalization=True):
        """Constructor.

        Parameters
        ----------
        complexity : int (default 3)
            The complexity of the features extracted.
            This is equivalent to setting r = complexity, d = complexity.

        r : int
            The maximal radius size.

        d : int
            The maximal distance size.

        min_r : int
            The minimal radius size.

        min_d : int
            The minimal distance size.

        nbits : int (default 20)
            The number of bits that defines the feature space size:
            |feature space|=2^nbits.

        normalization : bool (default True)
            Flag to set the resulting feature vector to have unit euclidean
            norm.

        inner_normalization : bool (default True)
            Flag to set the feature vector for a specific combination of the
            radius and distance size to have unit euclidean norm.
            When used together with the 'normalization' flag it will be applied
            first and then the resulting feature vector will be normalized.
        """
        if complexity is not None:
            self.r = complexity
            self.d = complexity
        else:
            self.r = r
            self.d = d
        self.min_r = min_r
        self.min_d = min_d
        self.nbits = nbits
        self.normalization = normalization
        self.inner_normalization = inner_normalization
        self.bitmask = pow(2, nbits) - 1
        self.feature_size = self.bitmask + 2

    def set_params(self, **args):
        """Set the parameters of the vectorizer."""
        if args.get('complexity', None) is not None:
            self.r = args['complexity']
            self.d = args['complexity']
        if args.get('r', None) is not None:
            self.r = args['r']
        if args.get('d', None) is not None:
            self.d = args['d']
        if args.get('min_r', None) is not None:
            self.min_r = args['min_r']
        if args.get('min_d', None) is not None:
            self.min_d = args['min_d']
        if args.get('nbits', None) is not None:
            self.nbits = args['nbits']
            self.bitmask = pow(2, self.nbits) - 1
            self.feature_size = self.bitmask + 2
        if args.get('normalization', None) is not None:
            self.normalization = args['normalization']
        if args.get('inner_normalization', None) is not None:
            self.inner_normalization = args['inner_normalization']

    def __repr__(self):
        """Pretty print of vectorizer parameters."""
        representation = """path_graph.Vectorizer(r = %d, d = %d, min_r = %d, min_d = %d, nbits = %d, \
            normalization = %s, inner_normalization = %s)""" % (
            self.r,
            self.d,
            self.min_r,
            self.min_d,
            self.nbits,
            self.normalization,
            self. inner_normalization)
        return representation

    def transform(self, seq_list):
        """Transform.

        Parameters
        ----------
        seq_list: list of sequence strings or
                  list of id, seq tuples or
                  list of id, seq, list of weight tuples
        """
        feature_rows = []
        for seq in seq_list:
            feature_rows.append(self._transform(seq))
        return self._convert_dict_to_sparse_matrix(feature_rows)

    def _convert_dict_to_sparse_matrix(self, feature_rows):
        if len(feature_rows) == 0:
            raise Exception('ERROR: something went wrong, empty features.')
        data, row, col = [], [], []
        for i, feature_row in enumerate(feature_rows):
            for feature in feature_row:
                row.append(i)
                col.append(feature)
                data.append(feature_row[feature])
        shape = (max(row) + 1, self.feature_size)
        return csr_matrix((data, (row, col)), shape=shape)

    def _get_sequence_and_weights(self, seq):
        if seq is None or len(seq) == 0:
            raise Exception('ERROR: something went wrong, empty instance.')
        if isinstance(seq, basestring):
            return seq, None
        elif isinstance(seq, tuple) and len(seq) == 2 and len(seq[1]) > 0:
            # assume the instance is a pair (header,seq) and extract only seq
            return seq[1], None
        elif isinstance(seq, tuple) and len(seq) == 3 and len(seq[1]) > 0:
            # assume the instance is a triple (header,seq,weightlist) and extract only seq
            return seq[1], seq[2]
        else:
            raise Exception('ERROR: something went wrong,\
             unrecognized input type for: %s' % seq)

    def _transform(self, orig_seq):
        seq, weights = self._get_sequence_and_weights(orig_seq)
        # extract kmer hash codes for all kmers up to r in all positions in seq
        seq_len = len(seq)
        neigh_hash_cache = [self._compute_neighborhood_hash(seq, pos)
                            for pos in range(seq_len)]
        if weights:
            if len(weights) != seq_len:
                raise Exception('ERROR: sequence and weights \
                    must be same length.')
            neighborhood_weight_cache = \
                [self._compute_neighborhood_weight(weights, pos)
                 for pos in range(seq_len)]
        # construct features as pairs of kmers up to distance d
        # for all radii up to r
        feature_list = defaultdict(lambda: defaultdict(float))
        for pos in range(seq_len):
            for radius in range(self.min_r, self.r + 1 + 2):
                if radius < len(neigh_hash_cache[pos]):
                    for distance in range(self.min_d, self.d + 2):
                        end = pos + distance
                        if end + radius < seq_len:
                            feature_code = \
                                fast_hash_4(neigh_hash_cache[pos][radius],
                                            neigh_hash_cache[end][radius],
                                            radius,
                                            distance,
                                            self.bitmask)
                            key = fast_hash_2(radius, distance, self.bitmask)
                            if weights:
                                feature_list[key][feature_code] += \
                                    neighborhood_weight_cache[pos][radius]
                                feature_list[key][feature_code] += \
                                    neighborhood_weight_cache[end][radius]
                            else:
                                feature_list[key][feature_code] += 1
        return self._normalization(feature_list)

    def _normalization(self, feature_list):
        # inner normalization per radius-distance
        feature_vector = {}
        for features in feature_list.itervalues():
            norm = 0
            for count in features.itervalues():
                norm += count * count
            sqrt_norm = math.sqrt(norm)
            for feature, count in features.iteritems():
                feature_vector_key = feature
                if self.inner_normalization:
                    feature_vector_value = float(count) / sqrt_norm
                else:
                    feature_vector_value = count
                feature_vector[feature_vector_key] = feature_vector_value
        # global normalization
        if self.normalization:
            normalized_feature_vector = {}
            total_norm = 0.0
            for feature, value in feature_vector.iteritems():
                total_norm += value * value
            sqrt_total_norm = math.sqrt(float(total_norm))
            for feature, value in feature_vector.iteritems():
                normalized_feature_vector[feature] = value / sqrt_total_norm
            return normalized_feature_vector
        else:
            return feature_vector

    def _compute_neighborhood_hash(self, seq, pos):
        subseq = seq[pos:pos + self.r + 1]
        return fast_hash_vec(subseq, self.bitmask)

    def _compute_neighborhood_weight(self, weights, pos):
        """TODO."""
        weight_list = []
        curr_weight = 0
        for w in weights[pos:pos + self.r + 1]:
            curr_weight += w
            weight_list.append(curr_weight)
        return weight_list

    def predict(self, seqs, estimator):
        """Predict.

        Takes an iterator over graphs and a fit estimator, and returns
        an iterator over predictions.
        """
        for seq in seqs:
            if len(seq) == 0:
                raise Exception('ERROR: something went wrong, empty instance.')
            # extract feature vector
            x = self._convert_dict_to_sparse_matrix(self._transform(seq))
            margins = estimator.decision_function(x)
            yield margins[0]

    def similarity(self, seqs, ref_instance=None):
        """Similarity.

        Takes an iterator over graphs and a reference graph, and returns
        an iterator over similarity evaluations.
        """
        reference_vec = self._convert_dict_to_sparse_matrix(
            self._transform(ref_instance))
        for seq in seqs:
            if len(seq) == 0:
                raise Exception('ERROR: something went wrong, empty instance.')
            # extract feature vector
            x = self._convert_dict_to_sparse_matrix(self._transform(seq))
            res = reference_vec.dot(x.T).todense()
            yield res[0, 0]

    def annotate(self, seqs, estimator=None, relabel=False):
        """Annotate.

        Given a list of sequences, and a fitted estimator, it computes a vector
        of importance values for each char in the sequence. The importance
        corresponds to the part of the score that is imputable  to the features
        that involve the specific char.

        Args:
            sequences: iterable lists of strings

            estimator: scikit-learn predictor trained on data sampled from
            the same distribution. If None only relabeling is used.

            relabel: bool. If True replace the label attribute of each vertex
            with the sparse vector encoding of all features that have that
            vertex as root.

        Returns:
            If relabel is False: for each input sequence a pair: 1) the input
            string, 2) a list of real  numbers with size equal to the number of
            characters in each input sequence.


            If relabel is True: for each input sequence a triplet: 1) the input
            string, 2) a list of real  numbers with size equal to the number of
            characters in each input sequence, 3) a list with  size equal to
            the number of characters in each input sequence, of sparse vectors
            each corresponding to the vertex induced features.
        """
        self.estimator = estimator
        self.relabel = relabel

        for seq in seqs:
            if seq is None or len(seq) == 0:
                raise Exception('ERROR: something went wrong, empty instance')
            if len(seq) == 2 and len(seq[1]) > 0:
                # assume the instance is a pair (header,seq)
                # and extract only seq
                seq = seq[1]
            yield self._annotate(seq)

    def _annotate(self, seq):
        # extract per vertex feature representation
        data_matrix = self._compute_vertex_based_features(seq)
        # extract importance information
        score, vec = self._annotate_importance(seq, data_matrix)
        # extract list of chars
        out_sequence = [c for c in seq]
        # add or update label information
        if self.relabel:
            return out_sequence, score, vec
        else:
            return out_sequence, score

    def _annotate_importance(self, seq, data_matrix):
        # compute distance from hyperplane as proxy of vertex importance
        if self.estimator is None:
            # if we do not provide an estimator then consider default margin of
            # 1 for all vertices
            margins = np.array([1] * data_matrix.shape[0])
        else:
            margins = self.estimator.decision_function(data_matrix)
        # compute the list of sparse vectors representation
        vec = []
        for i in range(data_matrix.shape[0]):
            vec.append(data_matrix.getrow(i))
        return margins, vec

    def _compute_vertex_based_features(self, seq):
        if seq is None or len(seq) == 0:
            raise Exception('ERROR: something went wrong, empty instance.')
        # extract kmer hash codes for all kmers up to r in all positions in seq
        vertex_based_features = []
        seq_len = len(seq)
        neigh_hash_cache = [self._compute_neighborhood_hash(seq, pos)
                            for pos in range(seq_len)]
        for pos in range(seq_len):
            # construct features as pairs of kmers up to distance d
            # for all radii up to r
            local_features = defaultdict(lambda: defaultdict(float))
            for radius in range(self.min_r, self.r + 2):
                if radius < len(neigh_hash_cache[pos]):
                    for distance in range(self.min_d, self.d + 2):
                        if pos + distance + radius < seq_len:
                            feature_code = fast_hash_4(neigh_hash_cache[pos][radius],
                                                       neigh_hash_cache[pos + distance][radius],
                                                       radius,
                                                       distance,
                                                       self.bitmask)
                            key = fast_hash_2(radius, distance, self.bitmask)
                            local_features[key][feature_code] += 1
            vertex_based_features.append(self._normalization(local_features))
        data_matrix = self._convert_dict_to_sparse_matrix(vertex_based_features)
        return data_matrix
