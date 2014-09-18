"""
Collection of classes and functions for the transformation of annotated graphs into sparse vectors.
"""
from collections import defaultdict
import numpy as np
import math
from scipy.sparse import csr_matrix

from collections import deque
from operator import itemgetter, attrgetter
import itertools

import networkx as nx
from scipy import stats




class vectorizer(object):
    """
    Transforms graphs in sparse vectors.
    """

    def __init__(self,
        r=3,
        d=3,
        nbits=20,
        normalization=True,
        inner_normalization=True,
        additional_pure_neighborhood_features=False,
        weighted=False):
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

        additional_pure_neighborhood_features : bool 
            If set additional features are going to be generated. 
            These features are generated in a similar fashion as the original features, 
            with the caveat that the first neighborhood is omitted.
            The purpose of these features is to allow vertices that have similar contexts to be 
            matched, even when they are completely different. 

        weighted : bool 
            If set the occurrence of each neighborhood is weighted by 
            aritmetic_mean(w(V))*geometric_mean(w(E)), where w(V) is the weight function for the 
            vertices and w(E) is the weight function for the edges. 
        """
        self.r = (r+1)*2
        self.d = (d+1)*2
        self.nbits = nbits
        self.normalization = normalization
        self.inner_normalization = inner_normalization
        self.additional_pure_neighborhood_features = additional_pure_neighborhood_features
        self.weighted = weighted
        self.bitmask = pow(2,nbits)-1
        self.feature_size = self.bitmask+2
        
    
    
    
    def _edge_to_vertex_transform(self, G_orig):
        """Converts edges to nodes so to process the graph ignoring the information on the 
        resulting edges."""
        G=nx.Graph()
        #build a graph that has as vertices the original vertex set
        label_size=0
        for n,d in G_orig.nodes_iter(data=True):
            d['node']=True
            if label_size == 0 :
                label_size = len(d['hlabel'])
            else :
                assert(label_size == len(d['hlabel'])),'ERROR: not all label vectors have the same length'
            G.add_node(n,d)            
        #and in addition a vertex for each edge
        for u,v,d in G_orig.edges_iter(data=True):
            new_node_id='%s|%s'%(u,v)
            d['edge']=True
            assert(label_size == len(d['hlabel'])),'ERROR: not all label vectors have the same length: edge: %s-%s on graph:\n %s'%(u,v, self._serialize(G_orig))
            G.add_node(new_node_id, d)
            #and the corresponding edges
            G.add_edge(new_node_id,u, label=1)
            G.add_edge(new_node_id,v, label=1)    
        G.graph['label_size']=label_size
        return G
        
        
        
    def _revert_edge_to_vertex_transform(self, G_orig):
        """Converts nodes of type 'edge' to edges. Useful for display reasons."""
        #start from a copy of the original graph
        G=nx.Graph(G_orig)
        #re-wire the endpoints of edge-vertices  
        for n,d in G_orig.nodes_iter(data=True):
            if d.get('edge', False) == True :
                #extract the endpoints
                endpoints=[u for u in G_orig.neighbors(n)]
                assert (len(endpoints)==2), 'ERROR: more than 2 endpoints'
                u=endpoints[0]
                v=endpoints[1]
                #add the corresponding edge
                G.add_edge(u,v,d)
                #remove the edge-vertex
                G.remove_node(n)
            if d.get('node', False) == True :
                #remove stale information
                G.node[n].pop('distant_neighbours', None)
        return G
            
        
        
    def transform(self,G_list):
        """
        Transforms a list of networkx graphs into a Numpy csr sparse matrix 
        (Compressed Sparse Row matrix).
        """
        feature_dict={}
        for instance_id , G in enumerate(G_list):
            feature_dict.update(self._transform(instance_id,G))
        return self._convert_to_sparse_vector(feature_dict)
        
        
        
    def transform_iter(self,G_list):
        """
        Transforms a list of networkx graphs into a Numpy csr sparse matrix 
        (Compressed Sparse Row matrix) and returns one sparse row at a time.
        This is a generator.
        """
        for instance_id , G in enumerate(G_list):
            yield self._convert_to_sparse_vector(self._transform(instance_id,G))

    
    
    def _convert_to_sparse_vector(self,feature_dict):
        data=feature_dict.values()
        row_col=feature_dict.keys()
        row=[i for i,j in row_col]
        col=[j for i,j in row_col]
        X=csr_matrix( (data,(row,col)), shape=(max(row)+1, self.feature_size))
        return X
   

                
    def _transform(self, instance_id , G):
        G=self._edge_to_vertex_transform(G)
        self._compute_distant_neighbours(G, max(self.r,self.d))       
        self._compute_neighborhood_graph_hash_cache(G)
        if self.weighted :
            self._compute_neighborhood_graph_weight_cache(G)
        #collect all features for all vertices for each  label_index
        feature_list=defaultdict(lambda : defaultdict(float))
        for v,d in G.nodes_iter(data=True):
            if d.get('node', False): #only for vertices of type 'node', i.e. not for the 'edge' type
                self._transform_vertex(G, v, feature_list)
            if d.get('nesting', False): #only for vertices of type 'nesting'
                self._transform_nesting_vertex(G, v, feature_list)
        return self._normalization(feature_list, instance_id)
        
        
        
    def _transform_nesting_vertex(self, G, nesting_vertex, feature_list):
        #extract endpoints
        nesting_endpoints=[u for u in G.neighbors(nesting_vertex)]
        assert (len(nesting_endpoints)==2), 'ERROR: nesting error'
        u=nesting_endpoints[0]
        v=nesting_endpoints[1]
        distance=1
        self._transform_vertex_pair(G, v, u, distance, feature_list)

                                
                                
    def _transform_vertex(self, G, v, feature_list):
        #for all distances 
        root_dist_dict=G.node[v]['distant_neighbours']
        for distance in range(0,self.d,2):
            if root_dist_dict.has_key(distance):
                node_set=root_dist_dict[distance]
                for u in node_set:
                    self._transform_vertex_pair(G, v, u, distance, feature_list)

                    
            
    def _transform_vertex_pair(self, G, v, u, distance, feature_list):
        #for all radii
        for radius in range(0,self.r,2):
            for label_index in range(G.graph['label_size']):
                if radius<len(G.node[v]['neighborhood_graph_hash'][label_index]) and radius<len(G.node[u]['neighborhood_graph_hash'][label_index]):
                    #feature as a pair of neighbourhoods at a radius,distance 
                    t=[G.node[v]['neighborhood_graph_hash'][label_index][radius],G.node[u]['neighborhood_graph_hash'][label_index][radius],radius,distance]
                    feature=self._fast_hash(t)
                    key=self._fast_hash([radius,distance])
                    if self.weighted == False :
                        feature_list[key][feature]+=1
                    else :
                        feature_list[key][feature]+=G.node[v]['neighborhood_graph_weight'][radius]+G.node[u]['neighborhood_graph_weight'][radius]
                    if self.additional_pure_neighborhood_features:
                        #feature as a radius, distance and a neighbourhood 
                        t=[G.node[u]['neighborhood_graph_hash'][label_index][radius],radius,distance]
                        feature=self._fast_hash(t)
                        key=self._fast_hash([radius,distance])
                        if self.weighted == False :
                            feature_list[key][feature]+=1
                        else :
                            feature_list[key][feature]+=G.node[u]['neighborhood_graph_weight'][radius]


                                
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
    
    
    
    def _compute_neighborhood_graph_hash_cache(self, G):
        assert (len(G)>0), 'ERROR: Empty graph'
        for u,d in G.nodes_iter(data=True):
            if d.get('node', False): 
                self._compute_neighborhood_graph_hash(u,G)
    
    
    
    def _compute_neighborhood_graph_hash(self,root,G):
        hash_neighborhood_list=[]
        #for all labels
        for label_index in range(G.graph['label_size']):
            #list all hashed labels at increasing distances
            hash_list=[]
            #for all distances 
            root_dist_dict=G.node[root]['distant_neighbours']
            for distance, node_set in root_dist_dict.iteritems():
                #create a list of hashed labels
                hash_label_list=[]
                for v in node_set:
                    vhlabel=G.node[v]['hlabel'][label_index]
                    hash_label_list.append(vhlabel) 
                #sort it
                hash_label_list.sort()
                #hash it
                hashed_nodes_at_distance_d_in_neighborhood_set=self._fast_hash(hash_label_list)
                hash_list.append(hashed_nodes_at_distance_d_in_neighborhood_set)
            #hash the sequence of hashes of the node set at increasing distances into a list of features
            hash_neighborhood=self._fast_hash_vec(hash_list)
            hash_neighborhood_list.append(hash_neighborhood)
        G.node[root]['neighborhood_graph_hash']=hash_neighborhood_list
    
    
           
    def _compute_neighborhood_graph_weight_cache(self, G):
        assert (len(G)>0), 'ERROR: Empty graph'
        for u,d in G.nodes_iter(data=True):
            if d.get('node', False): 
                self._compute_neighborhood_graph_weight(u,G)
    
    
    
    def _compute_neighborhood_graph_weight(self,root,G):
        #list all nodes at increasing distances
        #at each distance
        #compute the aritmetic mean weight on nodes
        #compute the geometric mean weight on edges
        #compute the pruduct of the two
        #make a list of the neighborhood_graph_weight at every distance
        neighborhood_graph_weight_list=[]
        w=G.node[root]['weight']
        node_weight_list=np.array([w])
        node_average=node_weight_list[0]
        edge_weight_list=np.array([1])
        edge_average=edge_weight_list[0]
        #for all distances 
        root_dist_dict=G.node[root]['distant_neighbours']
        for distance, node_set in root_dist_dict.iteritems():
            #extract array of weights at given distance
            weight_array_at_d=np.array([G.node[v]['weight'] for v in node_set])
            if distance % 2 == 0: #nodes
                node_weight_list=np.concatenate((node_weight_list, weight_array_at_d))
                node_average=mean(node_weight_list)
            else :
                edge_weight_list=np.concatenate((edge_weight_list, weight_array_at_d))
                edge_average=stats.mstats.gmean(edge_weight_list)
            weight=node_average*edge_average
            neighborhood_graph_weight_list.append(weight)
        G.node[root]['neighborhood_graph_weight']=neighborhood_graph_weight_list
            
            
    
    def _fast_hash(self, vec):
        running_hash = 0xAAAAAAAA
        for i,list_item in enumerate(vec):
            running_hash  ^= ((~(((running_hash << 11) + list_item) ^ (running_hash >> 5))),((running_hash << 7) ^ list_item * (running_hash >> 3)))[bool((i & 1) == 0)]
        return int(running_hash & self.bitmask)+1
    
   

    def _fast_hash_vec(self, vec):
        hash_vec=[]
        running_hash = 0xAAAAAAAA
        for i,list_item in enumerate(vec):
            running_hash  ^= ((~(((running_hash << 11) + list_item) ^ (running_hash >> 5))),((running_hash << 7) ^ list_item * (running_hash >> 3)))[bool((i & 1) == 0)]
            hash_vec+=[int(running_hash & self.bitmask)+1]
        return hash_vec
    


    def _single_vertex_breadth_first_visit(self, G, root, max_depth):
        #the map associates to each distance value (from 1:max_depth) 
        #the list of ids of the vertices at that distance from the root 
        dist_list={} 
        visited= set() #use a set as we can end up exploring few nodes
        q=deque() #q is the queue containing the frontieer to be expanded in the BFV
        q.append(root)
        dist={} #the map associates to each vertex id the distance from the root
        dist[root]=0
        visited.add(root)
        #add vertex at distance 0
        dist_list[0]=set()
        dist_list[0].add(root)
        while len(q) > 0 :
            #extract the current vertex
            u=q.popleft()
            d=dist[u]+1
            if d <= max_depth :
                #iterate over the neighbors of the current vertex
                for v in G.neighbors(u) :
                    if v not in visited :
                        #skip nesting edge-nodes
                        if G.node[v].get('nesting',False) == False :
                            dist[v]=d
                            visited.add(v)
                            q.append(v)
                            if dist_list.has_key(d) == False :
                                dist_list[d]=set()
                                dist_list[d].add(v)
                            else :
                                dist_list[d].add(v)
        G.node[root]['distant_neighbours']=dist_list
    
    
    
    def _compute_distant_neighbours(self, G, max_depth):
        for n,d in G.nodes_iter(data=True):
            if d.get('node', False): 
                self._single_vertex_breadth_first_visit(G, n, max_depth)
        



















from sklearn.linear_model import SGDClassifier

class importance_annotator(vectorizer):
    def __init__(self,
        estimator=SGDClassifier(),
        r=3,
        d=3,
        nbits=20,
        normalization=True,
        inner_normalization=True,
        additional_pure_neighborhood_features=False,
        weighted=False
        ):
        super(importance_annotator, self).__init__(r=r, 
            d=d, 
            nbits=nbits, 
            normalization=normalization, 
            inner_normalization=inner_normalization, 
            additional_pure_neighborhood_features=additional_pure_neighborhood_features, 
            weighted=weighted)
        self._estimator=estimator
    
 
    
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
        G=self._edge_to_vertex_transform(G_orig)
        self._compute_distant_neighbours(G, max(self.r,self.d))       
        self._compute_neighborhood_graph_hash_cache(G)
        if self.weighted :
            self._compute_neighborhood_graph_weight_cache(G)
        #extract per vertex feature representation
        feature_dict={}
        vertex_id=0
        for v,d in G.nodes_iter(data=True):
            if d.get('node', False): #only for vertices of type 'node', i.e. not for the 'edge' type
                feature_list=defaultdict(lambda : defaultdict(float))
                self._transform_vertex(G, v, feature_list)
                feature_dict.update(self._normalization(feature_list,vertex_id))
                vertex_id+=1
        X=self._convert_to_sparse_vector(feature_dict)
        #compute distance from hyperplane as proxy of vertex importance
        margins=self._estimator.decision_function(X)
        #annotate graph structure with vertex importance
        vertex_id=0
        for v,d in G.nodes_iter(data=True):
            if d.get('node', False): 
                G.node[v]['importance']=margins[vertex_id]
                vertex_id+=1
        return self._revert_edge_to_vertex_transform(G)