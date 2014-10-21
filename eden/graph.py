"""
Collection of classes and functions for the transformation of annotated graphs into sparse vectors.
"""
from collections import defaultdict
import numpy as np
import math
from scipy.sparse import csr_matrix
from scipy import stats
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from collections import deque
from operator import itemgetter, attrgetter
import itertools
import networkx as nx
import multiprocessing
from eden.util import util


class Vectorizer(object):
    """
    Transforms labeled, weighted, nested graphs in sparse vectors.
    """

    def __init__(self,
        r = 3,
        d = 3,
        nbits = 20,
        normalization = True,
        inner_normalization = True,
        pure_neighborhood_features = False,
        discretization_size = 0,
        discretization_dimension = 1):
        """
        Parameters
        ----------
        r : int 
            The radius size.

        d : int 
            The distance size.

        nbits : int 
            The number of bits that defines the feature space size: |feature space|=2^nbits.

        normalization : bool 
            If set the resulting feature vector will have unit euclidean norm.

        inner_normalization : bool 
            If set the feature vector for a specific combination of the radius and 
            distance size will have unit euclidean norm.
            When used together with the 'normalization' flag it will be applied first and 
            then the resulting feature vector will be normalized.

        pure_neighborhood_features : bool 
            If set additional features are going to be generated. 
            These features are generated in a similar fashion as the base features, 
            with the caveat that the first neighborhood is omitted.
            The purpose of these features is to allow vertices that have similar contexts to be 
            matched, even when they are completely different. 

        discretization_size : int
            Number of discretization levels for real vector labels.
            If 0 then treat all labels as strings. 

        discretization_dimension : int
            Size of the discretized label vector.
        """
        self.r = (r + 1) * 2
        self.d = (d + 1) * 2
        self.nbits = nbits
        self.normalization = normalization
        self.inner_normalization = inner_normalization
        self.pure_neighborhood_features = pure_neighborhood_features
        self.bitmask = pow(2, nbits) - 1
        self.feature_size = self.bitmask + 2
        self.discretization_size = discretization_size
        self.discretization_dimension = discretization_dimension
        self.discretization_model_dict = dict()


    def _extract_label_data_from_graph_list(self, G_list):
        label_data_dict = defaultdict( lambda : list( list() ) )
        #for all types in every node of every graph
        for instance_id, G in enumerate(G_list):
            label_data = self._extract_label_data(G)
            for kclass in label_data:
                label_data_dict[kclass] += label_data[kclass]
        #convert into dict of numpy matrices
        return self._convert_to_dict_of_dense_matrix(label_data_dict)


    def _extract_label_data(self, G_orig):
        label_data_dict = defaultdict( lambda : list( list() ) )
        #for all types in every node of every graph
        G=self._edge_to_vertex_transform(G_orig)
        for n, d in G.nodes_iter(data = True):
            kclass = d['class']
            label_data_dict[kclass] += [d['label']]
        return label_data_dict


    def _convert_to_dict_of_dense_matrix(self, label_data_dict):
        label_matrix_dict = dict()
        for kclass in label_data_dict:
            label_matrix_dict[kclass] = np.array(label_data_dict[kclass])
        return label_matrix_dict


    def fit(self, G_list, n_jobs = 1):
        """
        Constructs an approximate explicit mapping of a kernel function on the data 
        stored in the nodes of the graphs.

        Parameters
        ----------
        G_list : list of networkx graphs. 
            The data.

        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
            Use -1 to indicate the total number of CPUs available.
        """
        label_data_dict = self._extract_label_data_from_graph_list(G_list)
        for kclass in label_data_dict:
            #TODO: parameters for KMeans
            self.discretization_model_dict[kclass] = []
            for m in range(self.discretization_dimension):
                discretization_model = KMeans(init = 'random', n_clusters = self.discretization_size, n_jobs = n_jobs, n_init = 1)
                discretization_model.fit(label_data_dict[kclass])
                self.discretization_model_dict[kclass] += [discretization_model]


    def fit_transform(self, G_list, n_jobs = 1):
        """

        Parameters
        ----------
        G_list : list of networkx graphs. 
            The data.

        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1).
            Use -1 to indicate the total number of CPUs available.
        """
        self.fit(G_list, n_jobs = n_jobs)
        return self.transform(G_list, n_jobs = n_jobs)


    def transform(self, G_list, n_jobs = 1):
        """
        Transforms a list of networkx graphs into a Numpy csr sparse matrix 
        (Compressed Sparse Row matrix).

        Parameters
        ----------
        G_list : list of networkx graphs. 
            The data.

        n_jobs : integer, optional
            Number of jobs to run in parallel (default 1). 
            Use -1 to indicate the total number of CPUs available.
        """
        if n_jobs is 1:
            return self._transform_serial(G_list)
        else:
            return self._transform_parallel(G_list, n_jobs)


    def _convert_to_sparse_matrix(self, feature_dict):
        if len(feature_dict) == 0:
            raise Exception('ERROR: something went wrong, empty feature_dict. Perhaps wrong data format, i.e. do nodes have the "viewpoint" attribute?')
        data = feature_dict.values()
        row, col = [], []
        for i, j in feature_dict.iterkeys():
            row.append( i )
            col.append( j )
        X = csr_matrix( (data,(row,col)), shape = (max(row)+1, self.feature_size))
        return X


    def _transform_parallel(self,G_list, n_jobs):
        feature_dict = {}

        def my_callback( result ):
            feature_dict.update( result )

        if n_jobs == -1:
            n_jobs = None
        pool = multiprocessing.Pool(n_jobs)
        for instance_id , G in enumerate(G_list):
            util.apply_async(pool, self._transform, args=(instance_id, G), callback = my_callback)
        pool.close()
        pool.join()
        return self._convert_to_sparse_matrix( feature_dict )


    def _transform_serial(self, G_list):
        feature_dict={}
        for instance_id , G in enumerate( G_list ):
            feature_dict.update(self._transform( instance_id, G ))
        return self._convert_to_sparse_matrix( feature_dict )


    def transform_iter(self, G_list):
        """
        Transforms a list of networkx graphs into a Numpy csr sparse matrix 
        (Compressed Sparse Row matrix) and returns one sparse row at a time.
        This is a generator.
        """
        for instance_id , G in enumerate( G_list ):
            yield self._convert_to_sparse_matrix( self._transform(instance_id, G) )


    def _edge_to_vertex_transform(self, G_orig):
        """Converts edges to nodes so to process the graph ignoring the information on the 
        resulting edges."""
        G = nx.Graph()
        #build a graph that has as vertices the original vertex set
        for n,d in G_orig.nodes_iter(data=True):
            d['node']=True
            G.add_node(n,d)
        #and in addition a vertex for each edge
        for u, v, d in G_orig.edges_iter( data=True ):
            new_node_id = '%s|%s' % (u,v)
            d['edge'] = True
            G.add_node(new_node_id, d)
            #and the corresponding edges
            G.add_edge(new_node_id,u, label = 1)
            G.add_edge(new_node_id,v, label = 1)
        return G


    def _label_preprocessing(self, G):
        if self.discretization_size == 0:
            G.graph['label_size'] = 1
            for n,d in G.nodes_iter(data = True):
                G.node[n]['hlabel'] = [hash(str(d['label']))]
        else:
            G.graph['label_size'] = self.discretization_dimension
            for n,d in G.nodes_iter(data = True):
                kclass = d['class']
                data = np.array(d['label'])
                G.node[n]['hlabel'] = [self.discretization_model_dict[kclass][m].predict(data)[0]+1 for m in range(self.discretization_dimension)]


    def _weight_preprocessing(self, G):
        #we expect all vertices to have the attribute 'weight' in a weighted graph
        #sniff the attributes of the first node to determine if graph is weighted
        if 'weight' in G.nodes(data=True)[0][1]: 
            G.graph['weighted'] = True
            #check that edges are weighted, if not assign unitary weight
            for n,d in G.nodes_iter(data = True):
                if d.get('edge', False) == True :
                    if d.get('weight', False) == False :
                        G.node[n]['weight'] = 1


    def _revert_edge_to_vertex_transform(self, G_orig):
        """Converts nodes of type 'edge' to edges. Useful for display reasons."""
        #start from a copy of the original graph
        G = nx.Graph(G_orig)
        #re-wire the endpoints of edge-vertices
        for n,d in G_orig.nodes_iter(data = True):
            if d.get('edge', False) == True :
                #extract the endpoints
                endpoints = [u for u in G_orig.neighbors(n)]
                assert (len(endpoints) == 2), 'ERROR: more than 2 endpoints'
                u = endpoints[0]
                v = endpoints[1]
                #add the corresponding edge
                G.add_edge( u, v, d )
                #remove the edge-vertex
                G.remove_node(n)
            if d.get('node', False) == True:
                #remove stale information
                G.node[n].pop('remote_neighbours', None)
        return G


    def _graph_preprocessing(self, G_orig):
        assert(G_orig.number_of_nodes() > 0),'ERROR: Empty graph'
        G=self._edge_to_vertex_transform(G_orig)
        self._weight_preprocessing(G)
        self._label_preprocessing(G)
        self._compute_distant_neighbours(G, max(self.r,self.d))
        self._compute_neighborhood_graph_hash_cache(G)
        if G.graph.get('weighted',False):
            self._compute_neighborhood_graph_weight_cache(G)
        ################################################################################
        from eden.util import display
        display.draw_graph(G, size=6, 
                   node_size=2500, 
                   node_border=2, 
                   prog = 'neato', 
                   secondary_vertex_label='hlabel')
        ################################################################################
        return G


    def _transform(self, instance_id , G_orig):
        G = self._graph_preprocessing(G_orig)
        #collect all features for all vertices for each  label_index
        feature_list = defaultdict(lambda : defaultdict(float))
        for v,d in G.nodes_iter(data = True):
            if d.get('node', False): #only for vertices of type 'node', i.e. not for the 'edge' type
                if d.get('viewpoint', False): #only for vertices with attribute 'viewpoint'
                    self._transform_vertex(G, v, feature_list)
            if d.get('nesting', False): #only for vertices of type 'nesting'
                self._transform_nesting_vertex(G, v, feature_list)
        return self._normalization(feature_list, instance_id)


    def _transform_nesting_vertex(self, G, nesting_vertex, feature_list):
        #extract endpoints
        nesting_endpoints = [u for u in G.neighbors(nesting_vertex)]
        if len(nesting_endpoints) == 2:
            #new-style for nesting, in the original graph representation there are nesting edges 
            u = nesting_endpoints[0]
            v = nesting_endpoints[1]
            distance = 1
            self._transform_vertex_pair(G, v, u, distance, feature_list)
        else:
            #old-style for nesting, in the original graph representation there are nesting vertices 
            #separate ^ from @ nodes
            hat_nodes = [u for u in nesting_endpoints if G.node[u]['label'][0] == '^']
            at_nodes = [u for u in nesting_endpoints if G.node[u]['label'][0] == '@']
            #consider the neighbors of the neighbors
            for h_node in hat_nodes:
                hat_neighbors += [u for u in G.neighbors(h_node)]
            hat_neighbors = set(hat_neighbors)
            for a_node in at_nodes:
                at_neighbors += [u for u in G.neighbors(a_node)]
            at_neighbors = set(at_neighbors)
            pair_list = [(u,v) for u in hat_neighbors for v in at_neighbors if G.node[u]['label'][0] != '^' and G.node[v]['label'][0] != '^']
            for u,v in pair_list:
                distance = 1
                self._transform_vertex_pair(G, v, u, distance, feature_list)


    def _transform_vertex(self, G, v, feature_list):
        #for all distances 
        root_dist_dict = G.node[v]['remote_neighbours']
        for distance in range(0, self.d, 2):
            if root_dist_dict.has_key(distance):
                node_set = root_dist_dict[distance]
                for u in node_set:
                    self._transform_vertex_pair(G, v, u, distance, feature_list)


    def _transform_vertex_pair(self, G, v, u, distance, feature_list):
        self._transform_vertex_pair_base(G, v, u, distance, feature_list)
        if self.pure_neighborhood_features:
            self._transform_vertex_pair_pure_neighborhood(G, v, u, distance, feature_list)

#TODO: add features of type radius_1 + radius_2 = r_max
#TODO: add features of type radius_1 - radius_1 on each fronteer vertex

    def _transform_vertex_pair_base(self, G, v, u, distance, feature_list):
        #for all radii
        for radius in range(0,self.r,2):
            for label_index in range(G.graph['label_size']):
                if radius<len(G.node[v]['neighborhood_graph_hash'][label_index]) and radius<len(G.node[u]['neighborhood_graph_hash'][label_index]):
                    #feature as a pair of neighbourhoods at a radius,distance 
                    t = [G.node[v]['neighborhood_graph_hash'][label_index][radius],G.node[u]['neighborhood_graph_hash'][label_index][radius],radius,distance]
                    feature = util.fast_hash( t, self.bitmask )
                    key = util.fast_hash( [radius,distance], self.bitmask )
                    if G.graph.get('weighted',False) == False : #if self.weighted == False :
                        feature_list[key][feature]+=1
                    else :
                        feature_list[key][feature]+=G.node[v]['neighborhood_graph_weight'][radius]+G.node[u]['neighborhood_graph_weight'][radius]


    def _transform_vertex_pair_pure_neighborhood(self, G, v, u, distance, feature_list):
        #for all radii
        for radius in range(0, self.r, 2):
            for label_index in range(G.graph['label_size']):
                if radius < len(G.node[v]['neighborhood_graph_hash'][label_index]) and radius<len(G.node[u]['neighborhood_graph_hash'][label_index]):
                    #feature as a radius, distance and a neighbourhood 
                    t = [G.node[u]['neighborhood_graph_hash'][label_index][radius],radius,distance]
                    feature = util.fast_hash( t, self.bitmask )
                    key = util.fast_hash( [radius,distance], self.bitmask )
                    if G.graph.get('weighted',False) == False: #if self.weighted == False :
                        feature_list[key][feature]+=1
                    else:
                        feature_list[key][feature]+=G.node[u]['neighborhood_graph_weight'][radius]


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
                feature_vector_key = (instance_id,feature)
                if self.inner_normalization:
                    feature_vector_value = float(count)/sqrt_norm
                else:
                    feature_vector_value = count
                feature_vector[feature_vector_key] = feature_vector_value
                total_norm += feature_vector_value*feature_vector_value
        #global normalization
        if self.normalization:
            normalizationd_feature_vector = {}
            sqrt_total_norm = math.sqrt( float(total_norm) )
            for feature, value in feature_vector.iteritems():
                normalizationd_feature_vector[feature] = value/sqrt_total_norm
            return normalizationd_feature_vector
        else :
            return feature_vector


    def _compute_neighborhood_graph_hash_cache(self, G):
        assert (len(G)>0), 'ERROR: Empty graph'
        for u,d in G.nodes_iter(data=True):
            if d.get('node', False): 
                self._compute_neighborhood_graph_hash(u,G)


    def _compute_neighborhood_graph_hash(self, root, G):
        hash_neighborhood_list = []
        #for all labels
        for label_index in range(G.graph['label_size']):
            #list all hashed labels at increasing distances
            hash_list = []
            #for all distances 
            root_dist_dict = G.node[root]['remote_neighbours']
            for node_set in root_dist_dict.itervalues():
                #create a list of hashed labels
                hash_label_list = []
                for v in node_set:
                    vhlabel = G.node[v]['hlabel'][label_index]
                    hash_label_list.append(vhlabel)
                #sort it
                hash_label_list.sort()
                #hash it
                hashed_nodes_at_distance_d_in_neighborhood_set = util.fast_hash( hash_label_list, self.bitmask )
                hash_list.append(hashed_nodes_at_distance_d_in_neighborhood_set)
            #hash the sequence of hashes of the node set at increasing distances into a list of features
            hash_neighborhood = util.fast_hash_vec( hash_list, self.bitmask )
            hash_neighborhood_list.append( hash_neighborhood )
        G.node[root]['neighborhood_graph_hash']=hash_neighborhood_list


    def _compute_neighborhood_graph_weight_cache(self, G):
        assert (len(G)>0), 'ERROR: Empty graph'
        for u, d in G.nodes_iter( data=True ):
            if d.get('node', False): 
                self._compute_neighborhood_graph_weight(u,G)


    def _compute_neighborhood_graph_weight(self,root,G):
        #list all nodes at increasing distances
        #at each distance
        #compute the aritmetic mean weight on nodes
        #compute the geometric mean weight on edges
        #compute the pruduct of the two
        #make a list of the neighborhood_graph_weight at every distance
        neighborhood_graph_weight_list = []
        w = G.node[root]['weight']
        node_weight_list = np.array([w])
        node_average = node_weight_list[0]
        edge_weight_list = np.array([1])
        edge_average = edge_weight_list[0]
        #for all distances 
        root_dist_dict = G.node[root]['remote_neighbours']
        for distance, node_set in root_dist_dict.iteritems():
            #extract array of weights at given distance
            weight_array_at_d = np.array([G.node[v]['weight'] for v in node_set])
            if distance % 2 == 0: #nodes
                node_weight_list = np.concatenate((node_weight_list, weight_array_at_d))
                node_average = np.mean(node_weight_list)
            else : #edges
                edge_weight_list = np.concatenate((edge_weight_list, weight_array_at_d))
                edge_average = stats.gmean(edge_weight_list)
            weight = node_average * edge_average
            neighborhood_graph_weight_list.append(weight)
        G.node[root]['neighborhood_graph_weight'] = neighborhood_graph_weight_list


    def _single_vertex_breadth_first_visit(self, G, root, max_depth):
        #the map associates to each distance value (from 1:max_depth) 
        #the list of ids of the vertices at that distance from the root 
        dist_list = {}
        visited= set() #use a set as we can end up exploring few nodes
        q = deque() #q is the queue containing the frontieer to be expanded in the BFV
        q.append(root)
        dist = {} #the map associates to each vertex id the distance from the root
        dist[root] = 0
        visited.add(root)
        #add vertex at distance 0
        dist_list[0] = set()
        dist_list[0].add(root)
        while len(q) > 0:
            #extract the current vertex
            u = q.popleft()
            d = dist[u] + 1
            if d <= max_depth:
                #iterate over the neighbors of the current vertex
                for v in G.neighbors(u):
                    if v not in visited:
                        #skip nesting edge-nodes
                        if G.node[v].get('nesting', False) == False :
                            dist[v] = d
                            visited.add(v)
                            q.append(v)
                            if dist_list.has_key(d) == False :
                                dist_list[d] = set()
                                dist_list[d].add(v)
                            else:
                                dist_list[d].add(v)
        G.node[root]['remote_neighbours'] = dist_list


    def _compute_distant_neighbours(self, G, max_depth):
        for n, d in G.nodes_iter(data=True):
            if d.get('node', False): 
                self._single_vertex_breadth_first_visit(G, n, max_depth)


class Annotator(Vectorizer):
    def __init__(self,
        estimator = SGDClassifier(),
        vectorizer = Vectorizer(),
        reweight = 1.0):
        """
        Parameters
        ----------
        estimator : scikit-learn style predictor 

        vectorizer : EDeN graph vectorizer 

        reweight : float
            Update the 'weight' information as a linear combination of the previuous weight and 
            the absolute value of the margin predicted by the estimator. 
            If reweight = 0 then do not update.
            If reweight = 1 then discard previous weight information and use only abs(margin)
            If reweight = 0.5 then update with the aritmetic mean of the previous weight information 
            and the abs(margin)
        """
        self._estimator = estimator
        self.reweight = reweight
        self.r = vectorizer.r 
        self.d = vectorizer.d
        self.nbits = vectorizer.nbits
        self.normalization = vectorizer.normalization
        self.inner_normalization = vectorizer.inner_normalization
        self.pure_neighborhood_features = vectorizer.pure_neighborhood_features
        self.bitmask = vectorizer.bitmask
        self.feature_size = vectorizer.feature_size
        self.kernel_dict = vectorizer.kernel_dict
        self.discretizer_dict = vectorizer.discretizer_dict


    def transform(self,G_list):
        """
        Given a list of networkx graphs, and a fitted estimator, it returns a list of networkx 
        graphs where each vertex has an additional attribute with key 'importance'.
        The importance value of a vertex corresponds to the part of the score that is imputable 
        to the vertex and its neighborhood of radius r+d. 
        This is a generator.
        """
        for G in G_list:
            yield self._annotate_vertex_importance(G)


    def _annotate_vertex_importance(self, G_orig):
        #pre-processing phase: compute caches
        G = self._graph_preprocessing(G_orig)
        #extract per vertex feature representation
        X = self._compute_vertex_based_features(G)
        #compute distance from hyperplane as proxy of vertex importance
        margins = self._estimator.decision_function(X)
        #annotate graph structure with vertex importance
        vertex_id = 0
        for v,d in G.nodes_iter(data=True):
            if d.get('node', False): 
                #annotate the 'importance' attribute with the margin
                G.node[v]["importance"] = margins[vertex_id] 
                #update the 'weight' information as a linear combination of the previuous weight and the absolute margin 
                if G.node[v].has_key("weight") and self.reweight != 0:
                    G.node[v]["weight"] = self.reweight * abs(margins[vertex_id]) + (1-self.reweight) * G.node[v]["weight"] 
                else: #in case the original graph was not weighted then instantiate the 'weight' with the absolute margin 
                    G.node[v]["weight"] = abs(margins[vertex_id])
                vertex_id += 1
            if d.get('edge', False): #keep the weight of edges
                if G.node[v].has_key("weight") == False: #..unless they were unweighted, in this case add unit weight
                    G.node[v]["weight"] = 1
        return self._revert_edge_to_vertex_transform(G)


    def _compute_vertex_based_features(self, G):
        feature_dict = {}
        vertex_id = 0
        for v,d in G.nodes_iter(data=True):
            if d.get('node', False): #only for vertices of type 'node', i.e. not for the 'edge' type
                feature_list = defaultdict(lambda : defaultdict(float))
                self._transform_vertex(G, v, feature_list)
                feature_dict.update(self._normalization(feature_list,vertex_id))
                vertex_id += 1
        X = self._convert_to_sparse_matrix(feature_dict)
        return X
