from collections import defaultdict
import numpy as np
import math
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier
import multiprocessing
from eden.util import util

class Vectorizer():
    """

    """

    def __init__(self, 
        r = 3,
        d = 3,
        min_r = 0,
        min_d = 0,
        nbits = 20,
        normalization = True,
        inner_normalization = True):

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
        representation = """path_graph.Vectorizer(r = %d, d = %d, nbits = %d, normalization = %s, inner_normalization = %s)""" % (
            self.r - 1,
            self.d - 1, 
            self.nbits, 
            self.normalization, 
            self. inner_normalization)
        return representation


    def transform(self, seq_list, n_jobs = -1):
        """
        Parameters
        ----------
        seq_list : list of strings 
            The data.

        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
            Use -1 to indicate the total number of CPUs available.
        """
        if n_jobs is 1:
            return self._transform_serial(seq_list)
        else:
            return self._transform_parallel(seq_list, n_jobs)


    def _transform_parallel(self, seq_list, n_jobs):
        feature_dict = {}

        def my_callback( result ):
            feature_dict.update( result )
        
        if n_jobs == -1:
            n_jobs = None
        pool = multiprocessing.Pool(n_jobs)
        for instance_id, seq in enumerate(seq_list):
            util.apply_async(pool, self._transform, args=(instance_id, seq), callback = my_callback)
        pool.close()
        pool.join()
        return self._convert_dict_to_sparse_matrix(feature_dict)


    def _transform_serial(self,seq_list):
        feature_dict={}
        for instance_id,seq in enumerate(seq_list):
            feature_dict.update(self._transform(instance_id,seq))
        return self._convert_dict_to_sparse_matrix(feature_dict)


    def transform_iter(self, seq_list):
        for instance_id, seq in enumerate(seq_list):
            yield self._convert_dict_to_sparse_matrix(self._transform(instance_id,seq))


    def _convert_dict_to_sparse_matrix(self, feature_dict):
        if len(feature_dict) == 0:
            raise Exception('ERROR: something went wrong, empty feature_dict.')
        data = feature_dict.values()
        row, col = [], []
        for i, j in feature_dict.iterkeys():
            row.append( i )
            col.append( j )
        X = csr_matrix( (data, (row,col)), shape=( max(row) + 1, self.feature_size) )
        return X


    def _transform(self, instance_id , seq):
        if seq is None or len(seq) == 0:
            raise Exception('ERROR: something went wrong, empty instance at position %d.' % instance_id)
        #extract kmer hash codes for all kmers up to r in all positions in seq
        seq_len = len(seq)
        neighborhood_hash_cache = [self._compute_neighborhood_hash(seq, pos) for pos in range( seq_len )]
        #construct features as pairs of kmers up to distance d for all radii up to r
        feature_list = defaultdict(lambda : defaultdict(float))
        for pos in range( seq_len ):
            for radius in range( self.min_r, self.r + 1 ):
                if radius < len( neighborhood_hash_cache[pos] ):
                    feature = [ neighborhood_hash_cache[pos][radius], radius ]
                    for distance in range(self.min_d, self.d +1 ):
                        if pos + distance + radius < seq_len:
                            dfeature = feature + [ distance, neighborhood_hash_cache[ pos + distance ][ radius ] ]
                            feature_code = util.fast_hash( feature, self.bitmask )
                            key = util.fast_hash( [radius, distance], self.bitmask )
                            feature_list[key][feature_code] += 1
        return self._normalization(feature_list, instance_id)


    def _normalization(self, feature_list, instance_id):
        #inner normalization per radius-distance
        feature_vector = {}
        total_norm = 0.0
        for features in feature_list.itervalues():
            norm = 0
            for count in features.itervalues():
                norm += count*count
            sqrt_norm = math.sqrt(norm)
            for feature, count in features.iteritems():
                feature_vector_key = ( instance_id, feature )
                if self.inner_normalization:
                    feature_vector_value = float(count)/sqrt_norm
                else :
                    feature_vector_value = count
                feature_vector[feature_vector_key] = feature_vector_value
                total_norm += feature_vector_value*feature_vector_value
        #global normalization
        if self.normalization:
            normalization_feature_vector = {}
            sqrt_total_norm = math.sqrt(float(total_norm))
            for feature, value in feature_vector.iteritems():
                normalization_feature_vector[feature] = value/sqrt_total_norm
            return normalization_feature_vector
        else :
            return feature_vector


    def _compute_neighborhood_hash(self,seq, pos):
        """Given the seq and the pos, extract all kmers up to size r in a vector
        at position 0 in the vector there will be the hash of a single char, in position 1 of 2 chars, etc 
        """
        subseq = seq[ pos:pos + self.r ]
        return util.fast_hash_vec_char( subseq, self.bitmask )



class OutOfCoreSimilarity(Vectorizer):
    def __init__(self,
        ref_instance = None,
        vectorizer = Vectorizer()):
        """
        Purpose:
        ----------
        It outputs the similarity score between 'graph' and a stream of graphs.  

        Parameters
        ----------
        graph : an EDeN compatible graph 

        vectorizer : EDeN graph vectorizer 
        """
        self.r = vectorizer.r 
        self.d = vectorizer.d
        self.min_r = vectorizer.min_r 
        self.min_d = vectorizer.min_d
        self.nbits = vectorizer.nbits
        self.normalization = vectorizer.normalization
        self.inner_normalization = vectorizer.inner_normalization
        self.bitmask = vectorizer.bitmask
        self.feature_size = vectorizer.feature_size
        if ref_instance is None:
            raise Exception('ERROR: null ref_instance.')
        self._reference_vec = self._convert_dict_to_sparse_matrix(self._transform(0 , ref_instance))


    def predict(self,sequences):
        """
        This is a generator.
        """
        for seq in sequences:
            if len(seq) == 0:
                raise Exception('ERROR: something went wrong, empty instance.')
            yield self._predict(seq)


    def _predict(self, seq_orig):
        #extract feature vector
        x = self._convert_dict_to_sparse_matrix(self._transform(0 , seq_orig))
        res = self._reference_vec.dot(x.T).todense()
        return res[0,0]



class OutOfCorePredictor(Vectorizer):
    def __init__(self,
        estimator = SGDClassifier(),
        vectorizer = Vectorizer()):
        """
        Purpose:
        ----------
        It outputs the estimator prediction of the vectorized graph.  

        Parameters
        ----------
        estimator : scikit-learn predictor trained on data sampled from the same distribution. 
            If None the vertex weigths are by default 1.

        vectorizer : EDeN graph vectorizer 
        """
        self._estimator = estimator
        self.r = vectorizer.r 
        self.d = vectorizer.d
        self.min_r = vectorizer.min_r 
        self.min_d = vectorizer.min_d
        self.nbits = vectorizer.nbits
        self.normalization = vectorizer.normalization
        self.inner_normalization = vectorizer.inner_normalization
        self.bitmask = vectorizer.bitmask
        self.feature_size = vectorizer.feature_size


    def predict(self,sequences):
        """
        
        This is a generator.
        """
        for seq in sequences:
            if len(seq) == 0:
                raise Exception('ERROR: something went wrong, empty instance.')
            yield self._predict(seq)


    def _predict(self, seq_orig):
        #extract feature vector
        x = self._convert_dict_to_sparse_matrix(self._transform(0 , seq_orig))
        margins = self._estimator.decision_function(x)
        return margins[0]