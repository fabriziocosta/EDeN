from collections import defaultdict
import numpy as np
import math
from scipy.sparse import csr_matrix

class feature_constructor():
    def __init__(self, 
        r=3, 
        d=3, 
        nbits=20,
        normalization=True,
        inner_normalization=True):
        
        self.r = r+1
        self.d = d+1
        self.nbits = nbits
        self.normalization=normalization
        self.inner_normalization = inner_normalization
        self.bitmask = pow(2,nbits)-1
        self.feature_size = self.bitmask+2
        
        
        
    def transform(self,seq_list):
        #return self.transform_parallel(seq_list)
        return self.transform_serial(seq_list)
    
    
    
    def transform_parallel(self,seq_list):
        feature_dict={}
        
        def my_callback( result ):
            feature_dict.update( result )
        
        pool = multiprocessing.Pool()
        for instance_id,seq in enumerate(seq_list):
            print instance_id
            pool.apply_async( self._transform, args=(instance_id, 4), callback = my_callback)
        pool.close()
        pool.join()
        print feature_dict
        return self._convert_to_sparse_vector(feature_dict)
            
        
        
    def transform_serial(self,seq_list):
        feature_dict={}
        for instance_id,seq in enumerate(seq_list):
            feature_dict.update(self._transform(instance_id,seq))
        return self._convert_to_sparse_vector(feature_dict)
        
        
        
    def transform_iter(self, seq_list):
        for instance_id , seq in enumerate(seq_list):
            yield self._convert_to_sparse_vector(self._transform(instance_id,seq))

    
    
    def _convert_to_sparse_vector(self,feature_dict):
        data=feature_dict.values()
        row_col=feature_dict.keys()
        row=[i for i,j in row_col]
        col=[j for i,j in row_col]
        X=csr_matrix( (data,(row,col)), shape=(max(row)+1, self.feature_size))
        return X
        
        
        
    def _transform(self, instance_id , seq):
        #extract kmer hash codes for all kmers up to r in all positions in seq
        neighborhood_hash_cache=[self._compute_neighborhood_hash(seq, pos) for pos in range(len(seq))]
        
        #construct features as pairs of kmers up to distance d for all radii up to r
        feature_list=defaultdict(lambda : defaultdict(float))
        for pos in range(len(seq)):
            for radius in range(self.r):
                if radius<len(neighborhood_hash_cache[pos]):
                    feature=[]
                    feature+=[neighborhood_hash_cache[pos][radius]]
                    feature+=[radius]
                    for distance in range(self.d):
                        if pos+distance+radius<len(seq):
                            feature+=[distance]
                            feature+=[neighborhood_hash_cache[pos+distance][radius]]
                            feature_code=self._fast_hash(feature)
                            key=self._fast_hash([radius,distance])
                            feature_list[key][feature_code]+=1
        return self._normalization(feature_list, instance_id)
    


    def _normalization(self, feature_list, instance_id):
        #inner normalization per radius-distance
        feature_vector = {}
        total_norm = 0.0
        for key, features in feature_list.iteritems():
            norm = 0
            for feature, count in features.iteritems():
                norm += count*count
            for feature, count in features.iteritems():
                feature_vector_key = (instance_id,feature)
                if self.inner_normalization:
                    feature_vector_value = float(count)/math.sqrt(norm)
                else :
                    feature_vector_value = count
                feature_vector[feature_vector_key] = feature_vector_value
                total_norm += feature_vector_value*feature_vector_value
        #global normalization
        if self.normalization:
            normalizationd_feature_vector = {}
            for feature, value in feature_vector.iteritems():
                normalizationd_feature_vector[feature]=value/math.sqrt(float(total_norm))
            return normalizationd_feature_vector    
        else :
            return feature_vector



    def _compute_neighborhood_hash(self,seq, pos):
        #given the seq and the pos, extract all kmers up to size r in a vector
        #at position 0 in the vector there will be the hash of a single char, in position 1 of 2 chars, etc 
        subseq=seq[pos:pos+self.r]
        return self._fast_hash_vec(subseq)



    def _fast_hash(self, vec):
        running_hash = 0xAAAAAAAA
        for i,list_item in enumerate(vec):
            running_hash  ^= ((~(((running_hash << 11) + list_item) ^ (running_hash >> 5))),((running_hash << 7) ^ list_item * (running_hash >> 3)))[bool((i & 1) == 0)]
        return int(running_hash & self.bitmask)+1



    def _fast_hash_vec(self, vec):
        hash_vec=[]
        running_hash = 0xAAAAAAAA
        for i,list_item_str in enumerate(vec):
            list_item=ord(list_item_str)
            running_hash  ^= ((~(((running_hash << 11) + list_item) ^ (running_hash >> 5))),((running_hash << 7) ^ list_item * (running_hash >> 3)))[bool((i & 1) == 0)]
            hash_vec+=[int(running_hash & self.bitmask)+1]
        return hash_vec