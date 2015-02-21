#!/usr/bin/env python

import math
import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import SGDClassifier
from sklearn.cluster import KMeans
from collections import defaultdict, deque
from operator import attrgetter, itemgetter
import itertools
import networkx as nx
import multiprocessing
import copy
from eden.util import util


class Vectorizer(object):

    """
    Transforms vector labeled, weighted, nested graphs in sparse vectors.
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
                 pure_neighborhood_features=False,
                 discretization_size=0,
                 discretization_dimension=1):
        """
        Parameters
        ----------

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
        self.complexity = complexity
        if r is None:
            r = complexity
        if d is None:
            d = complexity
        self.r = r * 2
        self.d = d * 2
        self.min_r = min_r * 2
        self.min_d = min_d * 2
        self.nbits = nbits
        self.normalization = normalization
        self.inner_normalization = inner_normalization
        self.pure_neighborhood_features = pure_neighborhood_features
        self.bitmask = pow(2, nbits) - 1
        self.feature_size = self.bitmask + 2
        self.discretization_size = discretization_size
        self.discretization_dimension = discretization_dimension
        self.discretization_model_dict = dict()

    def __repr__(self):
        representation = """graph.Vectorizer( r = %d, d = %d, min_r = %d, min_d = %d, 
      nbits = %d, normalization = %s, inner_normalization = %s, pure_neighborhood_features = %s, 
      discretization_size = %d, discretization_dimension = %d )""" % (
            self.r / 2,
            self.d / 2,
            self.min_r / 2,
            self.min_d / 2,
            self.nbits,
            self.normalization,
            self. inner_normalization,
            self.pure_neighborhood_features,
            self.discretization_size,
            self.discretization_dimension)
        return representation

    def fit(self, graphs, n_jobs=-1):
        """
        Constructs an approximate explicit mapping of a kernel function on the data 
        stored in the nodes of the graphs.

        Parameters
        ----------
        graphs : list of networkx graphs. 
          The data.

        n_jobs : integer, optional
          Number of jobs to run in parallel ( default 1 ).
          Use -1 to indicate the total number of CPUs available.
        """
        label_data_matrix_dict = self._assemble_dense_data_matrices(graphs)
        label_data_matrix_dict.update(
            self._assemble_sparse_data_matrices(graphs))
        for node_class in label_data_matrix_dict:
            # TODO: parameters for KMeans
            self.discretization_model_dict[node_class] = []
            for m in range(self.discretization_dimension):
                discretization_model = KMeans(init='random',
                                              n_clusters=self.discretization_size,
                                              max_iter=100,
                                              n_jobs=n_jobs,
                                              n_init=1,
                                              random_state=m + 1)
                discretization_model.fit(label_data_matrix_dict[node_class])
                self.discretization_model_dict[
                    node_class] += [discretization_model]

    def fit_transform(self, graphs, n_jobs=-1):
        """

        Parameters
        ----------
        graphs : list of networkx graphs. 
          The data.

        n_jobs : integer, optional
          Number of jobs to run in parallel ( default 1 ).
          Use -1 to indicate the total number of CPUs available.
        """
        graphs_fit, graphs_transf = itertools.tee(graphs)
        self.fit(graphs_fit, n_jobs=n_jobs)
        return self.transform(graphs_transf, n_jobs=n_jobs)

    def transform(self, graphs, n_jobs=-1, block_size=-1):
        """
        Transforms a list of networkx graphs into a Numpy csr sparse matrix 
        ( Compressed Sparse Row matrix ).

        Parameters
        ----------
        graphs : list of networkx graphs. 
          The data.

        n_jobs : integer, optional
          Number of jobs to run in parallel ( default 1 ). 
          Use -1 to indicate the total number of CPUs available.

        block_size : integer, optional
          Number of instances per block ( default -1 meaning no blocks ). 
          After block_size instances a sparse matrix is 
          materialized and stacked to previous ones. 
        """

        if block_size == -1:
            return self._transform_block(graphs, n_jobs=n_jobs)
        else:
            G_block_list = util.grouper(graphs, block_size)
            for i, graphs in enumerate(G_block_list):
                X_curr = self._transform_block(graphs, n_jobs=n_jobs)
                if i == 0:
                    X = X_curr
                else:
                    X = vstack([X, X_curr], format="csr")
            return X

    def predict(self, graphs, estimator):
        """
        Takes an iterator over graphs and a fit estimator, and returns an iterator over predictions.
        """
        for G in graphs:
            self._test_goodness(G)
            # extract feature vector
            x = self._convert_dict_to_sparse_matrix(self._transform(0, G))
            margins = estimator.decision_function(x)
            prediction = margins[0]
            yield prediction

    def similarity(self, graphs, ref_instance=None):
        """
        Takes an iterator over graphs and a reference graph, and returns an iterator over similarity evaluations.
        """
        self._reference_vec = self._convert_dict_to_sparse_matrix(
            self._transform(0, ref_instance))
        for G in graphs:
            self._test_goodness(G)
            yield self._similarity(G)

    def _similarity(self, original_graph):
        # extract feature vector
        x = self._convert_dict_to_sparse_matrix(
            self._transform(0, original_graph))
        res = self._reference_vec.dot(x.T).todense()
        prediction = res[0, 0]
        return prediction

    def _test_goodness(self, G):
        if G.number_of_nodes() == 0:
            raise Exception('ERROR: something went wrong, empty graph.')

    def _transform_block(self, graphs, n_jobs=-1):
        if n_jobs == 1:
            return self._transform_serial(graphs)
        else:
            return self._transform_parallel(graphs, n_jobs)

    def _transform_parallel(self, graphs, n_jobs):
        feature_dict = {}

        def my_callback(result):
            feature_dict.update(result)

        if n_jobs == -1:
            n_jobs = None
        pool = multiprocessing.Pool(n_jobs)
        for instance_id, G in enumerate(graphs):
            self._test_goodness(G)
            util.apply_async(
                pool, self._transform, args=(instance_id, G), callback = my_callback)
        pool.close()
        pool.join()
        return self._convert_dict_to_sparse_matrix(feature_dict)

    def _transform_serial(self, graphs):
        instance_id = 0
        feature_dict = {}
        for instance_id, G in enumerate(graphs):
            self._test_goodness(G)
            feature_dict.update(self._transform(instance_id, G))
        if instance_id == 0:
            raise Exception(
                'ERROR: something went wrong, no graphs are present in current iterator.')
        return self._convert_dict_to_sparse_matrix(feature_dict)

    def transform_iter(self, graphs):
        """
        Transforms a list of networkx graphs into a Numpy csr sparse matrix 
        ( Compressed Sparse Row matrix ) and returns one sparse row at a time.
        This is a generator.
        """
        for instance_id, G in enumerate(graphs):
            self._test_goodness(G)
            yield self._convert_dict_to_sparse_matrix(self._transform(instance_id, G))

    def _extract_dense_vectors_from_labels(self, original_graph):
        # from each vertex extract the node_class and the label as a list and return a
        # dict with node_class as key and the vector associated to each vertex
        label_data_dict = defaultdict(lambda: list(list()))
        # transform edges to vertices to capture labels on edges too
        G = self._edge_to_vertex_transform(original_graph)
        # for all types in every node of every graph
        for n, d in G.nodes_iter(data=True):
            if isinstance(d['label'], list):
                node_class, data = self._extract_class_and_label(d)
                label_data_dict[node_class] += [data]
        return label_data_dict

    def _assemble_dense_data_matrix_dict(self, label_data_dict):
        # given a dict with node_class as keys and lists of vectors,
        # return a dict of numpy dense matrices
        label_matrix_dict = dict()
        for node_class in label_data_dict:
            label_matrix_dict[node_class] = np.array(
                label_data_dict[node_class])
        return label_matrix_dict

    def _assemble_dense_data_matrices(self, graphs):
        # take a list of graphs and extract the dictionaries of node_classes with
        # all the dense vectors associated to each vertex
        # then convert, for each node_class, the list of lists into a dense
        # numpy matrix
        label_data_dict = defaultdict(lambda: list(list()))
        # for every node of every graph
        for instance_id, G in enumerate(graphs):
            label_data = self._extract_dense_vectors_from_labels(G)
            # for all node_classes, add the list of dense vectors to the dict
            for node_class in label_data:
                label_data_dict[node_class] += label_data[node_class]
        # convert into dict of numpy matrices
        return self._assemble_dense_data_matrix_dict(label_data_dict)

    def _extract_sparse_vectors_from_labels(self, original_graph):
        # from each vertex extract the node_class and the label
        # if the label is of type dict
        # then
        label_data_dict = defaultdict(lambda: list(dict()))
        G = self._edge_to_vertex_transform(original_graph)
        for n, d in G.nodes_iter(data=True):
            if isinstance(d['label'], dict):
                node_class, data = self._extract_class_and_label(d)
                label_data_dict[node_class] += [data]
        return label_data_dict

    def _assemble_sparse_data_matrix_dict(self, label_data_dict):
        # given a dict with node_class as keys and lists of dicts,
        # return a dict of compressed sparse row matrices
        label_matrix_dict = dict()
        for node_class in label_data_dict:
            list_of_dicts = label_data_dict[node_class]
            feature_vector = {}
            for instance_id, vertex_dict in enumerate(list_of_dicts):
                for feature in vertex_dict:
                    feature_vector_key = (instance_id, int(feature))
                    feature_vector_value = vertex_dict[feature]
                    feature_vector[feature_vector_key] = feature_vector_value
            label_matrix_dict[node_class] = self._convert_dict_to_sparse_matrix(
                feature_vector)
        return label_matrix_dict

    def _assemble_sparse_data_matrices(self, graphs):
        # take a list of graphs and extract the dictionaries of node_classes with
        # all the sparse vectors associated to each vertex
        # then convert, for each node_class, the list of lists into a
        # compressed sparse row matrix
        label_data_dict = defaultdict(lambda: list(dict()))
        # for every node of every graph
        for instance_id, G in enumerate(graphs):
            label_data = self._extract_sparse_vectors_from_labels(G)
            # for all node_classes, update the list of dicts
            for node_class in label_data:
                label_data_dict[node_class] += label_data[node_class]
        # convert into dict of numpy matrices
        return self._assemble_sparse_data_matrix_dict(label_data_dict)

    def _convert_dict_to_sparse_vector(self, feature_dict):
        if len(feature_dict) == 0:
            raise Exception('ERROR: something went wrong, empty feature_dict.')
        data = feature_dict.values()
        row, col = [], []
        for j in feature_dict.iterkeys():
            row.append(0)
            col.append(int(hash(j) & self.bitmask) + 1)
        vec = csr_matrix((data, (row, col)), shape=(1, self.feature_size))
        return vec

    def _convert_dict_to_sparse_matrix(self, feature_dict):
        """Takes a dictionary with pairs as key and counts as values and returns a compressed sparse row matrix"""
        if len(feature_dict) == 0:
            raise Exception('ERROR: something went wrong, empty feature_dict.')
        data = feature_dict.values()
        row, col = [], []
        for i, j in feature_dict.iterkeys():
            row.append(i)
            col.append(j)
        return csr_matrix((data, (row, col)), shape=(max(row) + 1, self.feature_size))

    def _extract_class_and_label(self, d):
        # determine the class attribute
        # if the vertex does not have a 'class' attribute then provide a
        # default one
        if d.get('class', False):
            node_class = d['class']
        else:
            if isinstance(d['label'], list):
                node_class = 'vector'
            elif isinstance(d['label'], dict):
                node_class = 'sparse_vector'
            else:
                node_class = 'default'

        # determine the label type
        if isinstance(d['label'], list):
            # transform python list into numpy array
            data = np.array(d['label'])
        elif isinstance(d['label'], dict):
            # return the dict rerpesentation
            data = d['label']
        else:
            data = d['label']

        return node_class, data

    def _label_preprocessing(self, G):
        G.graph['label_size'] = self.discretization_dimension
        for n, d in G.nodes_iter(data=True):
            # for dense or sparse vectors
            if isinstance(d['label'], list) or isinstance(d['label'], dict):
                node_class, data = self._extract_class_and_label(d)
                if isinstance(d['label'], dict):
                    data = self._convert_dict_to_sparse_vector(data)
                # create a list of integer codes of size: discretization_dimension
                # each integer code is determined as follows:
                # for each class, use the correspondent discretization_model_dict[node_class] to extract the id of the
                # nearest cluster centroid, return the centroid id as the
                # integer code
                G.node[n]['hlabel'] = [util.fast_hash([hash(node_class), self.discretization_model_dict[
                                                      node_class][m].predict(data)[0] + 1], self.bitmask) for m in range(self.discretization_dimension)]
            if isinstance(d['label'], basestring):
                # copy a hashed version of the string for a number of times equal to self.discretization_dimension
                # in this way qualitative ( i.e. string ) labels can be
                # compared to the discretized labels
                hlabel = int(hash(d['label']) & self.bitmask) + 1
                G.node[n]['hlabel'] = [hlabel] * self.discretization_dimension

    def _weight_preprocessing(self, G):
        # it is expected that all vertices either have or do not have the attribute 'weight'
        # we sniff the attributes of the first node to determine if graph is
        # weighted
        if 'weight' in G.nodes(data=True)[0][1]:
            G.graph['weighted'] = True
            # check that edges are weighted, if not assign unitary weight
            for n, d in G.nodes_iter(data=True):
                if d.get('edge', False) == True:
                    if d.get('weight', False) == False:
                        G.node[n]['weight'] = 1

    def _edge_to_vertex_transform(self, original_graph):
        """Converts edges to nodes so to process the graph ignoring the information on the 
        resulting edges."""
        G = nx.Graph()
        # build a graph that has as vertices the original vertex set
        for n, d in original_graph.nodes_iter(data=True):
            d['node'] = True
            G.add_node(n, d)
        # and in addition a vertex for each edge
        new_node_id = max(original_graph.nodes()) + 1
        for u, v, d in original_graph.edges_iter(data=True):
            d['edge'] = True
            G.add_node(new_node_id, d)
            # and the corresponding edges
            G.add_edge(new_node_id, u, label=None)
            G.add_edge(new_node_id, v, label=None)
            new_node_id += 1
        return G

    def _revert_edge_to_vertex_transform(self, original_graph):
        """Converts nodes of type 'edge' to edges. Useful for display."""
        # start from a copy of the original graph
        G = nx.Graph(original_graph)
        # re-wire the endpoints of edge-vertices
        for n, d in original_graph.nodes_iter(data=True):
            if d.get('edge', False) == True:
                # extract the endpoints
                endpoints = [u for u in original_graph.neighbors(n)]
                assert (len(endpoints) == 2), 'ERROR: more than 2 endpoints'
                u = endpoints[0]
                v = endpoints[1]
                # add the corresponding edge
                G.add_edge(u, v, d)
                # remove the edge-vertex
                G.remove_node(n)
            if d.get('node', False) == True:
                # remove stale information
                G.node[n].pop('remote_neighbours', None)
        return G

    def _graph_preprocessing(self, original_graph):
        assert(original_graph.number_of_nodes() > 0), 'ERROR: Empty graph'
        G = self._edge_to_vertex_transform(original_graph)
        self._weight_preprocessing(G)
        self._label_preprocessing(G)
        self._compute_distant_neighbours(G, max(self.r, self.d))
        self._compute_neighborhood_graph_hash_cache(G)
        if G.graph.get('weighted', False):
            self._compute_neighborhood_graph_weight_cache(G)
        #################################################################
        # from eden.util import display
        # display.draw_graph( G, size=6,
        #      node_size=2500,
        #      node_border=2,
        #      prog = 'neato',
        #      secondary_vertex_label='hlabel' )
        #################################################################
        return G

    def _transform(self, instance_id, original_graph):
        if original_graph is None or original_graph.number_of_nodes() == 0:
            raise Exception(
                'ERROR: something went wrong, empty instance at position %d.' % instance_id)
        G = self._graph_preprocessing(original_graph)
        # collect all features for all vertices for each label_index
        feature_list = defaultdict(lambda: defaultdict(float))
        for v, d in G.nodes_iter(data=True):
            # only for vertices of type 'node', i.e. not for the 'edge' type
            if d.get('node', False):
                self._transform_vertex(G, v, feature_list)
            if d.get('nesting', False):  # only for vertices of type 'nesting'
                self._transform_nesting_vertex(G, v, feature_list)
        return self._normalization(feature_list, instance_id)

    def _transform_nesting_vertex(self, G, nesting_vertex, feature_list):
        # extract endpoints
        nesting_endpoints = [u for u in G.neighbors(nesting_vertex)]
        if len(nesting_endpoints) == 2:
            # new-style for nesting, in the original graph representation there
            # are nesting edges
            u = nesting_endpoints[0]
            v = nesting_endpoints[1]
            distance = 1
            self._transform_vertex_pair(G, v, u, distance, feature_list)
        else:
            # old-style for nesting, in the original graph representation there are nesting vertices
            # separate ^ from @ nodes
            hat_nodes = [
                u for u in nesting_endpoints if G.node[u]['label'][0] == '^']
            at_nodes = [
                u for u in nesting_endpoints if G.node[u]['label'][0] == '@']
            # consider the neighbors of the neighbors
            for h_node in hat_nodes:
                hat_neighbors += [u for u in G.neighbors(h_node)]
            hat_neighbors = set(hat_neighbors)
            for a_node in at_nodes:
                at_neighbors += [u for u in G.neighbors(a_node)]
            at_neighbors = set(at_neighbors)
            pair_list = [(u, v) for u in hat_neighbors for v in at_neighbors if G.node[
                u]['label'][0] != '^' and G.node[v]['label'][0] != '^']
            for u, v in pair_list:
                distance = 1
                self._transform_vertex_pair(G, v, u, distance, feature_list)

    def _transform_vertex(self, G, v, feature_list):
        # for all distances
        root_dist_dict = G.node[v]['remote_neighbours']
        for distance in range(self.min_d, self.d + 2, 2):
            if root_dist_dict.has_key(distance):
                node_set = root_dist_dict[distance]
                for u in node_set:
                    self._transform_vertex_pair(
                        G, v, u, distance, feature_list)

    def _transform_vertex_pair(self, G, v, u, distance, feature_list):
        self._transform_vertex_pair_base(G, v, u, distance, feature_list)
        if self.pure_neighborhood_features:
            self._transform_vertex_pair_pure_neighborhood(
                G, v, u, distance, feature_list)

    def _transform_vertex_pair_base(self, G, v, u, distance, feature_list):
        # for all radii
        for radius in range(self.min_r, self.r + 2, 2):
            for label_index in range(G.graph['label_size']):
                if radius < len(G.node[v]['neighborhood_graph_hash'][label_index]) and radius < len(G.node[u]['neighborhood_graph_hash'][label_index]):
                    # feature as a pair of neighbourhoods at a radius,distance
                    # canonicazation of pair of neighborhoods
                    v_hash = G.node[v]['neighborhood_graph_hash'][
                        label_index][radius]
                    u_hash = G.node[u]['neighborhood_graph_hash'][
                        label_index][radius]
                    if v_hash < u_hash:
                        first_hash = v_hash
                        second_hash = u_hash
                    else:
                        first_hash = u_hash
                        second_hash = v_hash
                    t = [first_hash, second_hash, radius, distance]
                    feature = util.fast_hash(t, self.bitmask)
                    key = util.fast_hash([radius, distance], self.bitmask)
                    # if self.weighted == False :
                    if G.graph.get('weighted', False) == False:
                        feature_list[key][feature] += 1
                    else:
                        feature_list[key][feature] += G.node[v]['neighborhood_graph_weight'][
                            radius] + G.node[u]['neighborhood_graph_weight'][radius]

    def _transform_vertex_pair_pure_neighborhood(self, G, v, u, distance, feature_list):
        # for all radii
        for radius in range(self.min_r, self.r + 2, 2):
            for label_index in range(G.graph['label_size']):
                if radius < len(G.node[v]['neighborhood_graph_hash'][label_index]) and radius < len(G.node[u]['neighborhood_graph_hash'][label_index]):
                    # feature as a radius, distance and a neighbourhood
                    t = [G.node[u]['neighborhood_graph_hash'][
                        label_index][radius], radius, distance]
                    feature = util.fast_hash(t, self.bitmask)
                    key = util.fast_hash([radius, distance], self.bitmask)
                    # if self.weighted == False :
                    if G.graph.get('weighted', False) == False:
                        feature_list[key][feature] += 1
                    else:
                        feature_list[key][
                            feature] += G.node[u]['neighborhood_graph_weight'][radius]

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
        #global normalization
        if self.normalization:
            normalized_feature_vector = {}
            sqrt_total_norm = math.sqrt(float(total_norm))
            for feature, value in feature_vector.iteritems():
                normalized_feature_vector[feature] = value / sqrt_total_norm
            return normalized_feature_vector
        else:
            return feature_vector

    def _compute_neighborhood_graph_hash_cache(self, G):
        assert (len(G) > 0), 'ERROR: Empty graph'
        for u, d in G.nodes_iter(data=True):
            if d.get('node', False):
                self._compute_neighborhood_graph_hash(u, G)

    def _compute_neighborhood_graph_hash(self, root, G):
        hash_neighborhood_list = []
        # for all labels
        for label_index in range(G.graph['label_size']):
            # list all hashed labels at increasing distances
            hash_list = []
            # for all distances
            root_dist_dict = G.node[root]['remote_neighbours']
            for node_set in root_dist_dict.itervalues():
                # create a list of hashed labels
                hash_label_list = []
                for v in node_set:
                    vhlabel = G.node[v]['hlabel'][label_index]
                    hash_label_list.append(vhlabel)
                # sort it
                hash_label_list.sort()
                # hash it
                hashed_nodes_at_distance_d_in_neighborhood_set = util.fast_hash(
                    hash_label_list, self.bitmask)
                hash_list.append(
                    hashed_nodes_at_distance_d_in_neighborhood_set)
            # hash the sequence of hashes of the node set at increasing
            # distances into a list of features
            hash_neighborhood = util.fast_hash_vec(hash_list, self.bitmask)
            hash_neighborhood_list.append(hash_neighborhood)
        G.node[root]['neighborhood_graph_hash'] = hash_neighborhood_list

    def _compute_neighborhood_graph_weight_cache(self, G):
        assert (len(G) > 0), 'ERROR: Empty graph'
        for u, d in G.nodes_iter(data=True):
            if d.get('node', False):
                self._compute_neighborhood_graph_weight(u, G)

    def _compute_neighborhood_graph_weight(self, root, G):
        # list all nodes at increasing distances
        # at each distance
        # compute the aritmetic mean weight on nodes
        # compute the geometric mean weight on edges
        # compute the pruduct of the two
        # make a list of the neighborhood_graph_weight at every distance
        neighborhood_graph_weight_list = []
        w = G.node[root]['weight']
        node_weight_list = np.array([w])
        node_average = node_weight_list[0]
        edge_weight_list = np.array([1])
        edge_average = edge_weight_list[0]
        # for all distances
        root_dist_dict = G.node[root]['remote_neighbours']
        for distance, node_set in root_dist_dict.iteritems():
            # extract array of weights at given distance
            weight_array_at_d = np.array(
                [G.node[v]['weight'] for v in node_set])
            if distance % 2 == 0:  # nodes
                node_weight_list = np.concatenate(
                    (node_weight_list, weight_array_at_d))
                node_average = np.mean(node_weight_list)
            else:  # edges
                edge_weight_list = np.concatenate(
                    (edge_weight_list, weight_array_at_d))
                edge_average = stats.gmean(edge_weight_list)
            weight = node_average * edge_average
            neighborhood_graph_weight_list.append(weight)
        G.node[root][
            'neighborhood_graph_weight'] = neighborhood_graph_weight_list

    def _single_vertex_breadth_first_visit(self, G, root, max_depth):
        # the map associates to each distance value ( from 1:max_depth )
        # the list of ids of the vertices at that distance from the root
        dist_list = {}
        visited = set()  # use a set as we can end up exploring few nodes
        # q is the queue containing the frontieer to be expanded in the BFV
        q = deque()
        q.append(root)
        # the map associates to each vertex id the distance from the root
        dist = {}
        dist[root] = 0
        visited.add(root)
        # add vertex at distance 0
        dist_list[0] = set()
        dist_list[0].add(root)
        while len(q) > 0:
            # extract the current vertex
            u = q.popleft()
            d = dist[u] + 1
            if d <= max_depth:
                # iterate over the neighbors of the current vertex
                for v in G.neighbors(u):
                    if v not in visited:
                        # skip nesting edge-nodes
                        if G.node[v].get('nesting', False) == False:
                            dist[v] = d
                            visited.add(v)
                            q.append(v)
                            if dist_list.has_key(d) == False:
                                dist_list[d] = set()
                                dist_list[d].add(v)
                            else:
                                dist_list[d].add(v)
        G.node[root]['remote_neighbours'] = dist_list

    def _compute_distant_neighbours(self, G, max_depth):
        for n, d in G.nodes_iter(data=True):
            if d.get('node', False):
                self._single_vertex_breadth_first_visit(G, n, max_depth)

    def annotate(self, graphs, estimator=None, reweight=1.0, relabel_vertex_with_vector=False):
        """
        Given a list of networkx graphs, and a fitted estimator, it returns a list of networkx 
        graphs where each vertex has an additional attribute with key 'importance'.
        The importance value of a vertex corresponds to the part of the score that is imputable 
        to the neighborhood of radius r+d of the vertex. 
        It can overwrite the label attribute with the sparse vector corresponding to the vertex induced features.

        Parameters
        ----------
        estimator : scikit-learn predictor trained on data sampled from the same distribution. 
          If None the vertex weigths are by default 1.

        reweight : float
          Update the 'weight' information of each vertex as a linear combination of the current weight and 
          the absolute value of the score computed by the estimator. 
          If reweight = 0 then do not update.
          If reweight = 1 then discard the current weight information and use only abs( score )
          If reweight = 0.5 then update with the aritmetic mean of the current weight information 
          and the abs( score )

        relabel_vertex_with_vector : bool
          If True replace the label attribute of each vertex with the 
          sparse vector encoding of all features that have that vertex as root. Create a new attribute 
          'original_label' to store the previous label. If the 'original_label' attribute is already present
          then it is left untouched: this allows an iterative application of the relabeling procedure while 
          preserving the original information. 
        """
        self.estimator = estimator
        self.reweight = reweight
        self.relabel_vertex_with_vector = relabel_vertex_with_vector

        for G in graphs:
            yield self._annotate(G)

    def _annotate(self, original_graph):
        # pre-processing phase: compute caches
        graph_dict = original_graph.graph
        G = self._graph_preprocessing(original_graph)
        # extract per vertex feature representation
        X = self._compute_vertex_based_features(G)
        # add or update weight and importance information
        G = self._annotate_importance(G, X)
        # add or update label information
        if self.relabel_vertex_with_vector:
            G = self._annotate_vector(G, X)
        G.graph = graph_dict
        return self._revert_edge_to_vertex_transform(G)

    def _annotate_vector(self, G, X):
        # annotate graph structure with vertex importance
        vertex_id = 0
        for v, d in G.nodes_iter(data=True):
            if d.get('node', False):
                # annotate 'vector' information
                row = X.getrow(vertex_id)
                vec_dict = {
                    str(index): value for index, value in zip(row.indices, row.data)}
                # if an original label does not exist then save it, else do
                # nothing and preserve the information in original label
                if G.node[v].get('original_label', False) == False:
                    G.node[v]["original_label"] = G.node[v]["label"]
                G.node[v]["label"] = vec_dict
                # if a node does not have a 'class' attribute then assign one
                # called 'vactor' by default
                if G.node[v].get("class", False) == False:
                    G.node[v]["class"] = 'vector'
                vertex_id += 1
        return G

    def _annotate_importance(self, G, X):
        # compute distance from hyperplane as proxy of vertex importance
        if self.estimator is None:
            # if we do not provide an estimator then consider default margin of
            # 1 for all vertices
            margins = np.array([1] * X.shape[0])
        else:
            margins = self.estimator.decision_function(X)
        # annotate graph structure with vertex importance
        vertex_id = 0
        for v, d in G.nodes_iter(data=True):
            if d.get('node', False):
                # annotate the 'importance' attribute with the margin
                G.node[v]["importance"] = margins[vertex_id]
                # update the 'weight' information as a linear combination of
                # the previuous weight and the absolute margin
                if G.node[v].has_key("weight") and self.reweight != 0:
                    G.node[v]["weight"] = self.reweight * \
                        abs(margins[vertex_id]) + \
                        (1 - self.reweight) * G.node[v]["weight"]
                # in case the original graph was not weighted then instantiate
                # the 'weight' with the absolute margin
                else:
                    G.node[v]["weight"] = abs(margins[vertex_id])
                vertex_id += 1
            if d.get('edge', False):  # keep the weight of edges
                # ..unless they were unweighted, in this case add unit weight
                if G.node[v].has_key("weight") == False:
                    G.node[v]["weight"] = 1
        return G

    def _compute_vertex_based_features(self, G):
        feature_dict = {}
        vertex_id = 0
        for v, d in G.nodes_iter(data=True):
            # only for vertices of type 'node', i.e. not for the 'edge' type
            if d.get('node', False):
                feature_list = defaultdict(lambda: defaultdict(float))
                self._transform_vertex(G, v, feature_list)
                feature_dict.update(
                    self._normalization(feature_list, vertex_id))
                vertex_id += 1
        X = self._convert_dict_to_sparse_matrix(feature_dict)
        return X


class ListVectorizer(Vectorizer):

    """
    Transforms vector labeled, weighted, nested graphs in sparse vectors. 

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
                 pure_neighborhood_features=False,
                 discretization_size=0,
                 discretization_dimension=1):
        """
        Parameters
        ----------

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
        self.vectorizer = Vectorizer(complexity=complexity,
                                     r=r,
                                     d=d,
                                     min_r=min_r,
                                     min_d=min_d,
                                     nbits=nbits,
                                     normalization=normalization,
                                     inner_normalization=inner_normalization,
                                     pure_neighborhood_features=pure_neighborhood_features,
                                     discretization_size=discretization_size,
                                     discretization_dimension=discretization_dimension)
        self.vectorizers = list()

    def fit(self, G_iterators_list, n_jobs=-1):
        """
        Constructs an approximate explicit mapping of a kernel function on the data 
        stored in the nodes of the graphs.

        Parameters
        ----------
        G_iterators_list : list of iterators over networkx graphs. 
          The data.
        """
        for i, graphs in enumerate(G_iterators_list):
            self.vectorizers.append(copy.copy(self.vectorizer))
            self.vectorizers[i].fit(graphs, n_jobs=n_jobs)

    def fit_transform(self, G_iterators_list, weights=list(), n_jobs=-1):
        """ 

        Parameters
        ----------
        G_iterators_list : list of iterators over networkx graphs. 
          The data.

        weights : list of positive real values.
          Weights for the linear combination of sparse vectors obtained on each iterated tuple of graphs.   
        """
        G_iterators_list_fit, G_iterators_list_transf = itertools.tee(
            G_iterators_list)
        self.fit(G_iterators_list_fit, n_jobs=n_jobs)
        return self.transform(G_iterators_list_transf, n_jobs=n_jobs)

    def transform(self, G_iterators_list, weights=list(), n_jobs=-1):
        """
        Transforms a list of networkx graphs into a Numpy csr sparse matrix 
        ( Compressed Sparse Row matrix ).

        Parameters
        ----------
        G_iterators_list : list of iterators over networkx graphs. 
          The data.

        weights : list of positive real values.
          Weights for the linear combination of sparse vectors obtained on each iterated tuple of graphs.   
        """
        # if no weights are provided then assume unitary weight
        if len(weights) == 0:
            weights = [1] * len(G_iterators_list)
        assert(len(G_iterators_list) == len(weights)
               ), 'ERROR: weights count is different than iterators count.'
        assert(len(filter(lambda x: x < 0, weights)) ==
               0), 'ERROR: weight list contains negative values.'
        for i, graphs in enumerate(G_iterators_list):
            if len(self.vectorizers) == 0:
                X_curr = self.vectorizer.transform(graphs, n_jobs=n_jobs)
            else:
                X_curr = self.vectorizers[i].transform(graphs, n_jobs=n_jobs)
            if i == 0:
                X = X_curr * weights[i]
            else:
                X = X + X_curr * weights[i]
        return X

    def similarity(self, G_iterators_list, ref_instance=None, weights=list()):
        """
        This is a generator.
        """
        self._reference_vec = self._convert_dict_to_sparse_matrix(
            self._transform(0, ref_instance))

        # if no weights are provided then assume unitary weight
        if len(weights) == 0:
            weights = [1] * len(G_iterators_list)
        assert(len(G_iterators_list) == len(weights)
               ), 'ERROR: weights count is different than iterators count.'
        assert(len(filter(lambda x: x < 0, weights)) ==
               0), 'ERROR: weight list contains negative values.'
        try:
            while True:
                graphs = [G_iterator.next() for G_iterator in G_iterators_list]
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

    def predict(self, G_iterators_list, estimator=SGDClassifier(), weights=list()):
        """
        Purpose:
        ----------
        It outputs the estimator prediction of the vectorized graph. 

        Parameters
        ----------
        estimator : scikit-learn predictor trained on data sampled from the same distribution. 
          If None the vertex weigths are by default 1.
        """
        self.estimator = estimator
        # if no weights are provided then assume unitary weight
        if len(weights) == 0:
            weights = [1] * len(G_iterators_list)
        assert(len(G_iterators_list) == len(weights)
               ), 'ERROR: weights count is different than iterators count.'
        assert(len(filter(lambda x: x < 0, weights)) ==
               0), 'ERROR: weight list contains negative values.'
        try:
            while True:
                graphs = [G_iterator.next() for G_iterator in G_iterators_list]
                yield self._predict(graphs, weights)
        except StopIteration:
            return

    def _predict(self, graphs, weights=list()):
        # extract feature vector
        for i, graph in enumerate(graphs):
            x_curr = self.vectorizer._convert_dict_to_sparse_matrix(
                self.vectorizer._transform(0, graph))
            if i == 0:
                x = x_curr * weights[i]
            else:
                x = x + x_curr * weights[i]
        margins = self.estimator.decision_function(x)
        prediction = margins[0]
        return prediction
