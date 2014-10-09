class WTA_hash:
    def __init__(self, num_functions = 1, dimensionality = 1, sparse = False):
        self.num_functions = num_functions
        self.dimensionality = dimensionality
        self.sparse=sparse


    def set_params(self, num_functions = 1, dimensionality = 1, sparse = False):
        self.num_functions = num_functions
        self.dimensionality = dimensionality
        self.sparse=sparse


    def fast_hash(self, item, seed = 0xAAAAAAAA):
        hashv = seed
        hashv  ^= ((~(((seed << 11) + item) ^ (seed >> 5))),((seed << 7) ^ item * (seed >> 3)))[True]
        return hashv+1
    
    
    def transform(self,X):
        #NOTE: we assume X is a numpy 2D array
        X_new = []
        for vec in X: 
            vec_new = self.signature(vec.tolist())
            X_new += vec_new
        return X_new


    def signature(self, vec):
        if self.sparse :
            return self.signature_sparse(vec)
        else :
            return self.signature_dense(vec)
        
        
    def signature_dense(self, vec):
        hash_signature = []
        for perm_index in range(1,self.num_functions+1):
            first_k_permuted_elements = self.extract_first_k_permuted_elements_dense(vec, perm_index, self.dimensionality)
            hash_signature.append(self.extract_code(first_k_permuted_elements))
        return hash_signature

    
    def signature_sparse(self, vec):
        hash_signature = []
        for perm_index in range(1,self.num_functions+1):
            first_k_permuted_elements = self.extract_first_k_permuted_elements_sparse(vec, perm_index, self.dimensionality)
            hash_signature.append(self.extract_code(first_k_permuted_elements))
        return hash_signature
    
    
    def extract_code(self, vec):
        max_id = 0
        max_val = vec[0][1]
        for i,(h,val) in enumerate(vec):
            if max_val < val :
                max_id = i
                max_val = val
        return max_id
    
    
    def extract_first_k_permuted_elements_sparse(self, vec, perm_index, k):
        hash_seed = self.fast_hash(perm_index)
        data = [(self.fast_hash(key,hash_seed),value) for key,value in vec.iteritems()]
        heapq.heapify(data)
        res = heapq.nsmallest(k,data)
        return res
    
    
    def extract_first_k_permuted_elements_dense(self, vec, perm_index, k):
        hash_seed = self.fast_hash(perm_index)
        data = [(self.fast_hash(key,hash_seed),value) for key,value in enumerate(vec)]
        heapq.heapify(data)
        res = heapq.nsmallest(k,data)
        return res

    
    def similarity(self, vec_a, vec_b):
        if self.sparse :
            return self.similarity_sparse(vec_a, vec_b)
        else :
            return self.similarity_dense(vec_a, vec_b)
        
        
    def similarity_sparse(self, vec_a, vec_b):
        sig_a = self.signature_sparse(vec_a)
        sig_b = self.signature_sparse(vec_b)
        return self.similarity_signature(sig_a, sig_b)
    
    
    def similarity_dense(self, vec_a, vec_b):
        sig_a = self.signature_dense(vec_a)
        sig_b = self.signature_dense(vec_b)
        return self.similarity_signature(sig_a, sig_b)

    
    def similarity_signature(self, sig_a, sig_b):
        sim = float(len(filter(lambda (x,y): x == y , zip(sig_a,sig_b))))/self.num_functions
        return sim



class discreteLSH:
    def __init__(self,r = 0.1, num_functions = 50, dimensionality = 128):
        self.r = r
        self.num_functions = num_functions
        self.dimensionality = dimensionality
        self.A = np.random.randn(dimensionality,num_functions)
        self.B = r*np.random.random_sample((1,num_functions))
    

    def set_params(self,r = 0.1, num_functions = 50, dimensionality = 128):
        self.r = r
        self.num_functions = num_functions
        self.dimensionality = dimensionality
        self.A = np.random.randn(dimensionality,num_functions)
        self.B = r*np.random.random_sample((1,num_functions))
    

    def transform(self,X):
        return np.array(np.floor((np.dot(X,self.A) + self.B) / self.r),numpy.int32)
    

    def transform_list(self, vec):
        return self.transform(np.array(vec))

    
    
class LSH:
    def __init__(self,r = 0.1, num_functions = 50, dimensionality = 128, gamma = 1):
        self.feature_map_LSH = discreteLSH(r, num_functions, dimensionality)
        self.feature_map_nystroem = Nystroem(kernel = 'rbf',gamma = gamma, n_components = dimensionality)
    

    def set_params(self,r = 0.1, num_functions = 50, dimensionality = 128, gamma = 1):
        self.feature_map_LSH = discreteLSH(r, num_functions, dimensionality)
        self.feature_map_nystroem = Nystroem(kernel = 'rbf',gamma = gamma, n_components = dimensionality)
        
    
    def transform(self,X):
        Xl = self.feature_map_nystroem.fit_transform(X)
        return self.feature_map_LSH.transform(Xl)