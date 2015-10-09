from collections import defaultdict
import numpy as np
import math
from scipy.sparse import csr_matrix
from eden import fast_hash_vec_char, fast_hash_2, fast_hash_4
from eden import AbstractVectorizer
import logging
logger = logging.getLogger(__name__)


class Vectorizer(AbstractVectorizer):

    """Transform strings into sparse vectors."""

    def __init__(self,
                 complexity=None,
                 r=3,
                 d=3,
                 min_r=0,
                 min_d=0,
                 nbits=20,
                 normalization=True,
                 inner_normalization=True):
        if complexity is not None:
            self.r = complexity + 1
            self.d = complexity + 1
        else:
            self.r = r + 1
            self.d = d + 1
        self.min_r = min_r
        self.min_d = min_d
        self.nbits = nbits
        self.normalization = normalization
        self.inner_normalization = inner_normalization
        self.bitmask = pow(2, nbits) - 1
        self.feature_size = self.bitmask + 2

    def __repr__(self):
        representation = """path_graph.Vectorizer(r = %d, d = %d, min_r = %d, min_d = %d, nbits = %d, \
            normalization = %s, inner_normalization = %s)""" % (
            self.r - 1,
            self.d - 1,
            self.min_r,
            self.min_d,
            self.nbits,
            self.normalization,
            self. inner_normalization)
        return representation

    def transform(self, seq_list):
        """
        Args:
            seq_list: list of strings
        """

        feature_dict = {}
        for instance_id, seq in enumerate(seq_list):
            feature_dict.update(self._transform(instance_id, seq))
        return self._convert_dict_to_sparse_matrix(feature_dict)

    def transform_iter(self, seq_list):
        for instance_id, seq in enumerate(seq_list):
            yield self._convert_dict_to_sparse_matrix(self._transform(instance_id, seq))

    def _convert_dict_to_sparse_matrix(self, feature_dict):
        if len(feature_dict) == 0:
            raise Exception('ERROR: something went wrong, empty feature_dict.')
        data = feature_dict.values()
        row, col = [], []
        for i, j in feature_dict.iterkeys():
            row.append(i)
            col.append(j)
        data_matrix = csr_matrix((data, (row, col)), shape=(max(row) + 1, self.feature_size))
        return data_matrix

    def _transform(self, instance_id, seq):
        if seq is None or len(seq) == 0:
            raise Exception('ERROR: something went wrong, empty instance # %d.' % instance_id)
        if len(seq) == 2 and len(seq[1]) > 0:
            # assume the instance is a pair (header,seq) and extract only seq
            seq = seq[1]
        # extract kmer hash codes for all kmers up to r in all positions in seq
        seq_len = len(seq)
        neighborhood_hash_cache = [self._compute_neighborhood_hash(seq, pos) for pos in range(seq_len)]
        # construct features as pairs of kmers up to distance d for all radii up to r
        feature_list = defaultdict(lambda: defaultdict(float))
        for pos in range(seq_len):
            for radius in range(self.min_r, self.r + 1):
                if radius < len(neighborhood_hash_cache[pos]):
                    for distance in range(self.min_d, self.d + 1):
                        second_endpoint = pos + distance
                        if second_endpoint + radius < seq_len:
                            feature_code = fast_hash_4(neighborhood_hash_cache[pos][radius],
                                                       radius,
                                                       distance,
                                                       neighborhood_hash_cache[second_endpoint][radius],
                                                       self.bitmask)
                            key = fast_hash_2(radius, distance, self.bitmask)
                            feature_list[key][feature_code] += 1
        return self._normalization(feature_list, instance_id)

    def _normalization(self, feature_list, instance_id):
        # inner normalization per radius-distance
        feature_vector = {}
        total_norm = 0.0
        for features in feature_list.itervalues():
            norm = 0
            for count in features.itervalues():
                norm += count * count
            sqrt_norm = math.sqrt(norm)
            for feature, count in features.iteritems():
                feature_vector_key = (instance_id, feature)
                if self.inner_normalization:
                    feature_vector_value = float(count) / sqrt_norm
                else:
                    feature_vector_value = count
                feature_vector[feature_vector_key] = feature_vector_value
                total_norm += feature_vector_value * feature_vector_value
        # global normalization
        if self.normalization:
            normalization_feature_vector = {}
            sqrt_total_norm = math.sqrt(float(total_norm))
            for feature, value in feature_vector.iteritems():
                normalization_feature_vector[feature] = value / sqrt_total_norm
            return normalization_feature_vector
        else:
            return feature_vector

    def _compute_neighborhood_hash(self, seq, pos):
        """
        Given the seq and the pos, extract all kmers up to size r in a vector
        at position 0 in the vector there will be the hash of a single char, in position 1 of 2 chars, etc
        """

        subseq = seq[pos:pos + self.r]
        return fast_hash_vec_char(subseq, self.bitmask)

    def predict(self, seqs, estimator):
        """
        Takes an iterator over graphs and a fit estimator, and returns an iterator over predictions.
        """

        for seq in seqs:
            if len(seq) == 0:
                raise Exception('ERROR: something went wrong, empty instance.')
            # extract feature vector
            x = self._convert_dict_to_sparse_matrix(self._transform(0, seq))
            margins = estimator.decision_function(x)
            yield margins[0]

    def similarity(self, seqs, ref_instance=None):
        """Takes an iterator over graphs and a reference graph, and returns an iterator
        over similarity evaluations."""

        reference_vec = self._convert_dict_to_sparse_matrix(self._transform(0, ref_instance))
        for seq in seqs:
            if len(seq) == 0:
                raise Exception('ERROR: something went wrong, empty instance.')
            # extract feature vector
            x = self._convert_dict_to_sparse_matrix(self._transform(0, seq))
            res = reference_vec.dot(x.T).todense()
            yield res[0, 0]

    def annotate(self, seqs, estimator=None, relabel=False):
        """
        Given a list of sequences, and a fitted estimator, it computes a vector
        of importance values for each char in the sequence. The importance
        corresponds to the part of the score that is imputable  to the features
        that involve the specific char.

        Args:
            sequences: iterable lists of strings

            estimator: scikit-learn predictor trained on data sampled from the same distribution.
            If None only relabeling is used.

            relabel: bool. If True replace the label attribute of each vertex with the
            sparse vector encoding of all features that have that vertex as root.

        Returns:
            If relabel is False: for each input sequence a pair: 1) the input
            string, 2) a list of real  numbers with size equal to the number of
            characters in each input sequence.


            If relabel is True: for each input sequence a triplet: 1) the input
            string, 2) a list of real  numbers with size equal to the number of
            characters in each input sequence, 3) a list with  size equal to the
            number of characters in each input sequence, of sparse vectors each
            corresponding to the vertex induced features.
        """

        self.estimator = estimator
        self.relabel = relabel

        for seq in seqs:
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
        feature_dict = {}
        seq_len = len(seq)
        neighborhood_hash_cache = [self._compute_neighborhood_hash(seq, pos) for pos in range(seq_len)]
        for pos in range(seq_len):
            # construct features as pairs of kmers up to distance d for all radii up to r
            feature_list = defaultdict(lambda: defaultdict(float))
            for radius in range(self.min_r, self.r + 1):
                if radius < len(neighborhood_hash_cache[pos]):
                    for distance in range(self.min_d, self.d + 1):
                        if pos + distance + radius < seq_len:
                            feature_code = fast_hash_4(neighborhood_hash_cache[pos][radius],
                                                       radius,
                                                       distance,
                                                       neighborhood_hash_cache[pos + distance][radius],
                                                       self.bitmask)
                            key = fast_hash_2(radius, distance, self.bitmask)
                            feature_list[key][feature_code] += 1
            feature_dict.update(self._normalization(feature_list, pos))
        data_matrix = self._convert_dict_to_sparse_matrix(feature_dict)
        return data_matrix
