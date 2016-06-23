#!/usr/bin/env python
"""Provides vectorization of graphs."""

import math
import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict, deque
import itertools
import joblib
import networkx as nx
from eden import fast_hash, fast_hash_vec
from eden import fast_hash_2, fast_hash_3, fast_hash_4
from eden import AbstractVectorizer
from eden.util import serialize_dict

import logging
logger = logging.getLogger(__name__)


class Vectorizer(AbstractVectorizer):
    """Transform real vector labeled, weighted graphs in sparse vectors."""

    def __init__(self,
                 complexity=3,
                 r=None,
                 d=None,
                 n=1,
                 min_r=0,
                 min_d=0,
                 min_n=2,
                 weights_dict=None,
                 label_size=1,
                 nbits=20,
                 normalization=True,
                 inner_normalization=True,
                 positional=False,
                 key_label='label',
                 key_weight='weight',
                 key_nesting='nesting',
                 key_importance='importance',
                 key_class='class',
                 key_original_label='original_label',
                 key_entity='entity'):
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

        n : int (default 1)
            The maximal number of clusters used to discretize real label
            vectors.

        min_r : int
            The minimal radius size.

        min_d : int
            The minimal distance size.

        min_n : int (default 2)
            The minimal number of clusters used to discretize real label
            vectors.

        weights_dict : dict of floats
            Dictionary with keys pairs of radius distance and value weights.

        label_size : int (default 1)
            the number of discretization steps used in the conversion from
            real valued labels to discrete labels.

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

        positional : bool (default False)
            Flag to make the relative position be sorted by the node ID value.
            This is useful for ensuring isomorphism for sequences.

        key_label : string (default 'label')
            The key used to indicate the label information in nodes.

        key_weight : string (default 'weight')
            The key used to indicate the weight information in nodes.

        key_nesting : string (default 'nesting')
            The key used to indicate the nesting type in edges.

        key_importance : string (default 'importance')
            The key used to indicate the importance information in nodes.

        key_original_label : string (default 'original_label')
            The key used to indicate the original label information in nodes.

        key_entity : string (default 'entity')
            The key used to indicate the entity information in nodes.
        """
        self.name = self.__class__.__name__
        self.complexity = complexity
        if r is None:
            r = complexity
        if d is None:
            d = complexity
        self.r = r
        self.d = d
        self.min_r = min_r
        self.min_d = min_d
        self.n = n
        self.min_n = min_n
        self.weights_dict = weights_dict
        # Note: if the discretization is active then the default label_size
        # should be 5
        self.label_size = label_size
        if self.n > 1 and self.label_size == 1:
            self.label_size = min(self.n, 5)
        # Note: if the discretization is active then label_size cannot be
        # larger than n-min_n
        if self.n > 1:
            self.label_size = min(self.n - self.min_n, self.label_size)
        self.nbits = nbits
        self.normalization = normalization
        self.inner_normalization = inner_normalization
        self.positional = positional
        self.bitmask = pow(2, nbits) - 1
        self.feature_size = self.bitmask + 2
        self.discretizers = dict()
        self.fit_status = None
        self.key_label = key_label
        self.key_weight = key_weight
        self.key_nesting = key_nesting
        self.key_importance = key_importance
        self.key_class = key_class
        self.key_original_label = key_original_label
        self.key_entity = key_entity

    def set_params(self, **args):
        """Set the parameters of the vectorizer."""
        if args.get('complexity', None) is not None:
            self.complexity = args['complexity']
            self.r = self.complexity
            self.d = self.complexity
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
        if args.get('positional', None) is not None:
            self.positional = args['positional']
        if args.get('n', None) is not None:
            self.n = args['n']
        if args.get('min_n', None) is not None:
            self.min_n = args['min_n']
        if args.get('label_size', None) is not None:
            self.label_size = args['label_size']
        else:
            self.label_size = 1
        if self.n > 1 and self.label_size == 1:
            self.label_size = min(self.n, 5)
        # Note: if the discretization is active then label_size cannot be
        # larger than n-min_n
        if self.n > 1:
            self.label_size = min(self.n - self.min_n, self.label_size)

    def __repr__(self):
        """string."""
        return serialize_dict(self.__dict__, offset='large')

    def save(self, model_name):
        """save."""
        joblib.dump(self, model_name, compress=1)

    def load(self, obj):
        """load."""
        self.__dict__.update(joblib.load(obj).__dict__)

    def fit(self, graphs):
        """Fit to the real valued vector data stored in the nodes of the graphs.

        Parameters
        ----------
        graphs : list[graphs]
            The input list of networkx graphs.

        Returns
        -------
        self

        """
        if self.n == 1:
            # fit is meaningful only when n>1
            logger.debug('Warning: fit was asked with n=1')
        else:
            n_clusters_list = self._compute_n_clusters_list()
            label_data_matrixs = dict()
            graphs, graphs_ = itertools.tee(graphs)
            label_data_matrixs = self._assemble_dense_data_matrices(graphs_)
            label_data_matrixs.update(
                self._assemble_sparse_data_matrices(graphs))
            for node_entity in label_data_matrixs:
                self.discretizers[node_entity] = []
                for m in n_clusters_list:
                    discretization_model = MiniBatchKMeans(n_clusters=m,
                                                           init='k-means++',
                                                           max_iter=10,
                                                           n_init=10,
                                                           random_state=m)
                    discretization_model.fit(label_data_matrixs[node_entity])
                    self.discretizers[node_entity] += \
                        [discretization_model]
            self.fit_status = 'fit'
        return self

    def _compute_n_clusters_list(self):
        # compute a log spaced sequence (label_size elements) of number of
        # clusters in this way when asked for max 1000 clusters and min 4
        # clusters and 5 levels we produce the sequence of cluster sizes:
        # 4,16,64,256,1024
        c_start = math.log10(self.min_n)
        c_end = math.log10(self.n)
        n_clusters_list = [int(x)
                           for x in
                           np.ceil(np.logspace(c_start,
                                               c_end,
                                               num=self.label_size))]
        # remove repeated values; this manages artifacts resulting from very
        # large values of label_size
        n_clusters_list = sorted(list(set(n_clusters_list)))
        n_trailing = self.label_size - len(n_clusters_list)
        if n_trailing > 0:
            n_clusters_list += [n_clusters_list[-1]] * n_trailing
        return n_clusters_list

    def partial_fit(self, graphs):
        """Update the discretizer of the real valued vector data.

        Parameters
        ----------
        graphs : list[graphs]
            The input list of networkx graphs.

        Returns
        -------
        self
        """
        # if partial_fit is invoked prior to a fit invocation then run fit
        # instead
        if self.fit_status != 'fit':
            self.fit(graphs)
        else:
            label_data_matrixs = self._assemble_dense_data_matrices(graphs)
            label_data_matrixs.update(
                self._assemble_sparse_data_matrices(graphs))
            for node_entity in label_data_matrixs:
                self.discretizers[node_entity] = []
                for i in range(self.label_size):
                    d = label_data_matrixs[node_entity]
                    self.discretizers[node_entity][i].partial_fit(d)
        return self

    def fit_transform(self, graphs):
        """Fit the discretizer to the real valued vector data.

        Then transform a list of networkx graphs into a sparse matrix.

        Parameters
        ----------
        graphs : list[graphs]
            The input list of networkx graphs.

        Returns
        -------
        data_matrix : array-like, shape = [n_samples, n_features]
            Vector representation of input graphs.
        """
        graphs, graphs_ = itertools.tee(graphs)
        self.fit(graphs_)
        return self.transform(graphs)

    def transform(self, graphs):
        """Transform a list of networkx graphs into a sparse matrix.

        Parameters
        ----------
        graphs : list[graphs]
            The input list of networkx graphs.

        Returns
        -------
        data_matrix : array-like, shape = [n_samples, n_features]
            Vector representation of input graphs.
        """
        instance_id = None
        feature_rows = []
        for instance_id, graph in enumerate(graphs):
            self._test_goodness(graph)
            feature_rows.append(self._transform(graph))
        if instance_id is None:
            raise Exception('ERROR: something went wrong:\
                no graphs are present in current iterator.')
        return self._convert_dict_to_sparse_matrix(feature_rows)

    def transform_single(self, graph):
        """Transform a single networkx graph into one sparse row."""
        self._test_goodness(graph)
        return self._convert_dict_to_sparse_matrix([self._transform(graph)])

    def predict(self, graphs, estimator=None):
        """Return an iterator over the decision function output."""
        for graph in graphs:
            self._test_goodness(graph)
            # extract feature vector
            x = self._convert_dict_to_sparse_matrix([self._transform(graph)])
            if estimator.__class__.__name__ in ['SGDRegressor']:
                margins = estimator.predict(x)
            else:
                margins = estimator.decision_function(x)
            prediction = margins[0]
            yield prediction

    def similarity(self, graphs, ref_instance=None, estimator=None):
        """Iterator over the dot product between ref_instance and graphs."""
        reference_vec = \
            self._convert_dict_to_sparse_matrix(
                [self._transform(ref_instance)])
        for graph in graphs:
            self._test_goodness(graph)
            # extract feature vector
            x = self._convert_dict_to_sparse_matrix([self._transform(graph)])
            # if an estimator is given then consider the transformed
            # feature vectors instead
            if estimator is not None:
                reference_vec_ = reference_vec.multiply(estimator.coef_)
                x = x.multiply(estimator.coef_)
                res = reference_vec_.dot(x.T)
            else:
                res = reference_vec.dot(x.T).todense()
            prediction = res[0, 0]
            yield prediction

    def distance(self, graphs, ref_instance=None):
        """Iterator on euclidean distance between ref_instance and graphs."""
        reference_vec = \
            self._convert_dict_to_sparse_matrix(
                [self._transform(ref_instance)])
        for graph in graphs:
            self._test_goodness(graph)
            # extract feature vector
            x = self._convert_dict_to_sparse_matrix([self._transform(graph)])
            dist = reference_vec - x
            norm = dist.dot(dist.T).todense()
            norm = norm[0, 0]
            norm = math.sqrt(norm)
            yield norm

    def _test_goodness(self, graph):
        if graph.number_of_nodes() == 0:
            raise Exception('ERROR: something went wrong, empty graph.')

    def _extract_dense_vectors_from_labels(self, original_graph):
        # from each vertex extract the node_entity and the label as a list and
        # return a dict with node_entity as key and the vector associated to
        # each vertex
        label_data_dict = defaultdict(lambda: list(list()))
        # transform edges to vertices to capture labels on edges too
        graph = _edge_to_vertex_transform(original_graph)
        # for all types in every node of every graph
        for n, d in graph.nodes_iter(data=True):
            if isinstance(d[self.key_label], list):
                node_entity, data = _extract_entity_and_label(d,
                                                              self.key_entity,
                                                              self.key_label)
                label_data_dict[node_entity].append(data)
        return label_data_dict

    def _assemble_dense_data_matrix_dict(self, label_data_dict):
        # given a dict with node_entity as keys and lists of vectors,
        # return a dict of numpy dense matrices
        label_matrix_dict = dict()
        for node_entity in label_data_dict:
            label_matrix_dict[node_entity] = \
                np.array(label_data_dict[node_entity], dtype=np.float64)
        return label_matrix_dict

    def _assemble_dense_data_matrices(self, graphs):
        # take a list of graphs and extract the dictionaries of node_entityes
        # with all the dense vectors associated to each vertex
        # then convert, for each node_entity, the list of lists into a dense
        # numpy matrix
        label_data_dict = defaultdict(lambda: list(list()))
        # for every node of every graph
        for instance_id, G in enumerate(graphs):
            label_data = self._extract_dense_vectors_from_labels(G)
            # for all node_entityes, add the list of dense vectors to the dict
            for node_entity in label_data:
                label_data_dict[node_entity] += label_data[node_entity]
        # convert into dict of numpy matrices
        return self._assemble_dense_data_matrix_dict(label_data_dict)

    def _extract_sparse_vectors_from_labels(self, original_graph):
        # from each vertex extract the node_entity and the label
        # if the label is of type dict
        # then
        label_data_dict = defaultdict(lambda: list(dict()))
        graph = _edge_to_vertex_transform(original_graph)
        for n, d in graph.nodes_iter(data=True):
            if isinstance(d[self.key_label], dict):
                node_entity, data = self._extract_entity_and_label(d)
                label_data_dict[node_entity] += [data]
        return label_data_dict

    def _assemble_sparse_data_matrix_dict(self, label_data_dict):
        # given a dict with node_entity as keys and lists of dicts,
        # return a dict of compressed sparse row matrices
        label_matrix_dict = dict()
        for node_entity in label_data_dict:
            list_of_dicts = label_data_dict[node_entity]
            feature_vector = {}
            for instance_id, vertex_dict in enumerate(list_of_dicts):
                for feature in vertex_dict:
                    code = (int(hash(feature) & self.bitmask) + 1)
                    feature_vector_key = (instance_id, code)
                    feature_vector_value = vertex_dict[feature]
                    feature_vector[feature_vector_key] = feature_vector_value
            label_matrix_dict[node_entity] = \
                self._convert_dict_to_sparse_matrix(feature_vector)
        return label_matrix_dict

    def _assemble_sparse_data_matrices(self, graphs):
        # take a list of graphs and extract the dictionaries of node_entityes
        # with all the sparse vectors associated to each vertex
        # then convert, for each node_entity, the list of lists into a
        # compressed sparse row matrix
        label_data_dict = defaultdict(lambda: list(dict()))
        # for every node of every graph
        for instance_id, G in enumerate(graphs):
            label_data = self._extract_sparse_vectors_from_labels(G)
            # for all node_entityes, update the list of dicts
            for node_entity in label_data:
                label_data_dict[node_entity] += label_data[node_entity]
        # convert into dict of numpy matrices
        return self._assemble_sparse_data_matrix_dict(label_data_dict)

    def _convert_dict_to_sparse_matrix(self, feature_rows):
        if len(feature_rows) == 0:
            raise Exception('ERROR: something went wrong, empty features.')
        data, row, col = [], [], []
        for i, feature_row in enumerate(feature_rows):
            if len(feature_row) == 0:
                # case of empty feature set for a specific instance
                row.append(i)
                col.append(0)
                data.append(0)
            else:
                for feature in feature_row:
                    row.append(i)
                    col.append(feature)
                    data.append(feature_row[feature])
        shape = (max(row) + 1, self.feature_size)
        data_matrix = csr_matrix((data, (row, col)), shape=shape)
        return data_matrix

    def _weight_preprocessing(self, graph):
        # if at least one vertex or edge is weighted then ensure that all
        # vertices and edges are weighted in this case use a default weight
        # of 1 if the weight attribute is missing
        graph.graph['weighted'] = False
        for n, d in graph.nodes_iter(data=True):
            if self.key_weight in d:
                graph.graph['weighted'] = True
                break
        if graph.graph['weighted'] is True:
            for n, d in graph.nodes_iter(data=True):
                if self.key_weight not in d:
                    graph.node[n][self.key_weight] = 1

    def _graph_preprocessing(self, original_graph):
        graph = _edge_to_vertex_transform(original_graph)
        self._weight_preprocessing(graph)
        _label_preprocessing(graph, label_size=self.label_size,
                             key_label=self.key_label,
                             key_entity=self.key_entity,
                             discretizers=self.discretizers,
                             bitmask=self.bitmask)
        self._compute_distant_neighbours(graph, max(self.r, self.d) * 2)
        self._compute_neighborhood_graph_hash_cache(graph)
        if graph.graph.get('weighted', False):
            self._compute_neighborhood_graph_weight_cache(graph)
        return graph

    def _transform(self, original_graph):
        graph = self._graph_preprocessing(original_graph)
        # collect all features for all vertices for each label_index
        feature_list = defaultdict(lambda: defaultdict(float))
        for v, d in graph.nodes_iter(data=True):
            # only for vertices of type 'node', i.e. not for the 'edge' type
            if d.get('node', False):
                self._transform_vertex(graph, v, feature_list)
            # only for vertices of type self.key_nesting
            # if d.get(self.key_nesting, False):
            #    self._transform_nesting_vertex(graph, v, feature_list)
        _clean_graph(graph)
        return self._normalization(feature_list)

    def _transform_vertex(self, graph, vertex_v, feature_list):
        # for all distances
        root_dist_dict = graph.node[vertex_v]['remote_neighbours']
        for distance in range(self.min_d * 2, (self.d + 1) * 2, 2):
            if distance in root_dist_dict:
                node_set = root_dist_dict[distance]
                for vertex_u in node_set:
                    self._transform_vertex_pair(graph, vertex_v, vertex_u,
                                                distance, feature_list)
        # find all vertices, if any, that are second point of nesting edge
        endpoints = self._find_second_endpoint_of_nesting_edge(graph, vertex_v)
        for endpoint, connection_weight in endpoints:
            # for all vertices at distance d from each such second endpoint
            endpoint_dist_dict = graph.node[endpoint]['remote_neighbours']
            for distance in range(self.min_d * 2, (self.d + 1) * 2, 2):
                if distance in endpoint_dist_dict:
                    node_set = endpoint_dist_dict[distance]
                    # for all nodes u at distance distance from endpoint
                    for vertex_u in node_set:
                        self._transform_vertex_pair(
                            graph, vertex_v, vertex_u,
                            distance, feature_list,
                            connection_weight=connection_weight)

    def _find_second_endpoint_of_nesting_edge(self, graph, vertex_v):
        endpoints = []
        # find all neighbors
        for u in graph.neighbors(vertex_v):
            # test for type
            if graph.node[u].get(self.key_nesting, False):
                # if type is nesting
                # find endpoint that is not original vertex_v
                vertices = [j for j in graph.neighbors(u) if j != vertex_v]
                assert(len(vertices) == 1)
                connection_weight = graph.node[u].get(self.key_weight, 1)
                endpoints.append((vertices[0], connection_weight))
        return endpoints

    def _transform_vertex_pair(self, graph, vertex_v, vertex_u,
                               distance, feature_list, connection_weight=1):
        # for all radii
        for radius in range(self.min_r * 2, (self.r + 1) * 2, 2):
            for label_index in range(graph.graph['label_size']):
                if radius < len(
                    graph.node[vertex_v]['neigh_graph_hash'][label_index]) \
                    and radius < len(
                        graph.node[vertex_u]['neigh_graph_hash'][label_index]):
                    # feature as a pair of neighborhoods at a radius,distance
                    # canonicalization of pair of neighborhoods
                    vertex_v_labels = graph.node[vertex_v]['neigh_graph_hash']
                    vertex_v_hash = vertex_v_labels[label_index][radius]
                    vertex_u_labels = graph.node[vertex_u]['neigh_graph_hash']
                    vertex_u_hash = vertex_u_labels[label_index][radius]
                    if vertex_v_hash < vertex_u_hash:
                        first_hash, second_hash = (vertex_v_hash,
                                                   vertex_u_hash)
                    else:
                        first_hash, second_hash = (vertex_u_hash,
                                                   vertex_v_hash)
                    feature = fast_hash_4(first_hash, second_hash,
                                          radius, distance, self.bitmask)
                    # half features are those that ignore the central vertex v
                    # the reason to have those is to help model the context
                    # independently from the identity of the vertex itself
                    half_feature = fast_hash_3(vertex_u_hash,
                                               radius, distance, self.bitmask)
                    # Note: to be compatible with external radius, distance
                    # we need to revert to r/2 and d/2
                    key = (radius / 2, distance / 2)
                    if graph.graph.get('weighted', False) is False:
                        feature_list[key][feature] += connection_weight
                        feature_list[key][half_feature] += connection_weight
                    else:
                        weight_v = graph.node[vertex_v]['neigh_graph_weight']
                        weight_u = graph.node[vertex_u]['neigh_graph_weight']
                        weight_vu_radius = weight_v[radius] + weight_u[radius]
                        val = connection_weight * weight_vu_radius
                        feature_list[key][feature] += val
                        half_val = connection_weight * weight_u[radius]
                        feature_list[key][half_feature] += half_val

    def _normalization(self, feature_list):
        # inner normalization per radius-distance
        feature_vector = {}
        for r_d_key in feature_list:
            features = feature_list[r_d_key]
            norm = 0
            for count in features.itervalues():
                norm += count * count
            sqrt_norm = math.sqrt(norm)
            if self.weights_dict is not None:
                # reweight using external weight dictionary
                if self.weights_dict.get(r_d_key, None) is not None:
                    sqrtw = math.sqrt(self.weights_dict[r_d_key])
                    sqrt_norm = sqrt_norm / sqrtw
            for feature_id, count in features.iteritems():
                if self.inner_normalization:
                    feature_vector_value = float(count) / sqrt_norm
                else:
                    feature_vector_value = count
                feature_vector[feature_id] = feature_vector_value
        # global normalization
        if self.normalization:
            normalized_feature_vector = {}
            total_norm = 0.0
            for value in feature_vector.itervalues():
                total_norm += value * value
            sqrt_total_norm = math.sqrt(float(total_norm))
            for feature_id, value in feature_vector.iteritems():
                feature_vector_value = value / sqrt_total_norm
                normalized_feature_vector[feature_id] = feature_vector_value
            return normalized_feature_vector
        else:
            return feature_vector

    def _compute_neighborhood_graph_hash_cache(self, graph):
        assert (len(graph) > 0), 'ERROR: Empty graph'
        for u, d in graph.nodes_iter(data=True):
            if d.get('node', False):
                self._compute_neighborhood_graph_hash(u, graph)

    def _compute_neighborhood_graph_hash(self, root, graph):
        hash_neighborhood_list = []
        # for all labels
        for label_index in range(graph.graph['label_size']):
            # list all hashed labels at increasing distances
            hash_list = []
            # for all distances
            root_dist_dict = graph.node[root]['remote_neighbours']
            for node_set in root_dist_dict.itervalues():
                # create a list of hashed labels
                hash_label_list = []
                for v in node_set:
                    # compute the vertex hashed label by hashing the hlabel
                    # field of position label_index
                    # with the degree of the vertex (obtained as the size of
                    # the adjacency dictionary for the vertex v)
                    # or, in case positional is set, using the relative
                    # position of the vertex v w.r.t. the root vertex
                    if self.positional:
                        vhlabel = fast_hash_2(
                            graph.node[v]['hlabel'][label_index],
                            root - v)
                    else:
                        vhlabel = \
                            fast_hash_2(graph.node[v]['hlabel'][label_index],
                                        len(graph[v]))
                    hash_label_list.append(vhlabel)
                # sort it
                hash_label_list.sort()
                # hash it
                hashed_nodes_at_distance_d_in_neighborhood = fast_hash(
                    hash_label_list)
                hash_list.append(hashed_nodes_at_distance_d_in_neighborhood)
            # hash the sequence of hashes of the node set at increasing
            # distances into a list of features
            hash_neighborhood = fast_hash_vec(hash_list)
            hash_neighborhood_list.append(hash_neighborhood)
        graph.node[root]['neigh_graph_hash'] = hash_neighborhood_list

    def _compute_neighborhood_graph_weight_cache(self, graph):
        assert (len(graph) > 0), 'ERROR: Empty graph'
        for u, d in graph.nodes_iter(data=True):
            if d.get('node', False):
                self._compute_neighborhood_graph_weight(u, graph)

    def _compute_neighborhood_graph_weight(self, root, graph):
        # list all nodes at increasing distances
        # at each distance
        # compute the arithmetic mean weight on nodes
        # compute the geometric mean weight on edges
        # compute the product of the two
        # make a list of the neighborhood_graph_weight at every distance
        neighborhood_graph_weight_list = []
        w = graph.node[root][self.key_weight]
        node_weight_list = np.array([w], dtype=np.float64)
        node_average = node_weight_list[0]
        edge_weight_list = np.array([1], dtype=np.float64)
        edge_average = edge_weight_list[0]
        # for all distances
        root_dist_dict = graph.node[root]['remote_neighbours']
        for distance, node_set in root_dist_dict.iteritems():
            # extract array of weights at given distance
            weight_array_at_d = np.array([graph.node[v][self.key_weight]
                                          for v in node_set], dtype=np.float64)
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
        graph.node[root]['neigh_graph_weight'] = \
            neighborhood_graph_weight_list

    def _single_vertex_breadth_first_visit(self, graph, root, max_depth):
        # the map associates to each distance value ( from 1:max_depth )
        # the list of ids of the vertices at that distance from the root
        dist_list = {}
        visited = set()  # use a set as we can end up exploring few nodes
        # q is the queue containing the frontier to be expanded in the BFV
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
                for v in graph.neighbors(u):
                    if v not in visited:
                        # skip nesting edge-nodes
                        if graph.node[v].get(self.key_nesting, False) is False:
                            dist[v] = d
                            visited.add(v)
                            q.append(v)
                            if d in dist_list:
                                dist_list[d].add(v)
                            else:
                                dist_list[d] = set()
                                dist_list[d].add(v)
        graph.node[root]['remote_neighbours'] = dist_list

    def _compute_distant_neighbours(self, graph, max_depth):
        for n, d in graph.nodes_iter(data=True):
            if d.get('node', False):
                self._single_vertex_breadth_first_visit(graph, n, max_depth)

    def annotate(self, graphs, estimator=None, reweight=1.0, relabel=False):
        """Write importance and weight attribute according to estimator.

        Given a list of networkx graphs, and a fitted estimator, return a list
        of networkx graphs where each vertex has an additional attribute with
        key 'importance'. The importance value of a vertex corresponds to the
        part of the score that is imputable to the neighborhood of radius r+d
        of the vertex.
        It can overwrite the label attribute with the sparse vector
        corresponding to the vertex induced features.

        Parameters
        ----------
        estimator : scikit-learn estimator
            Scikit-learn predictor trained on data sampled from the same
            distribution. If None the vertex weights are set by default 1.

        reweight : float (default 1.0)
            The  coefficient used to weight the linear combination of the
            current weight and the absolute value of the score computed by the
            estimator.
            If reweight = 0 then do not update.
            If reweight = 1 then discard the current weight information and use
            only abs( score )
            If reweight = 0.5 then update with the arithmetic mean of the
            current weight information and the abs( score )

        relabel : bool (default false)
            Flag to replace the label attribute of each vertex with the sparse
            vector encoding of all features that have that vertex as root.
            Create a new attribute 'original_label' to store the previous
            label. If the 'original_label' attribute is already present then
            it is left untouched: this allows an iterative application of the
            relabeling procedure while preserving the original information.
        """
        self.estimator = estimator
        self.reweight = reweight
        self.relabel = relabel

        for graph in graphs:
            annotated_graph = self._annotate(graph)
            yield annotated_graph

    def _annotate(self, original_graph):
        # pre-processing phase: compute caches
        graph_dict = original_graph.graph
        graph = self._graph_preprocessing(original_graph)
        # extract per vertex feature representation
        data_matrix = self._compute_vertex_based_features(graph)
        if self.estimator is not None:
            # add or update weight and importance information
            graph = self._annotate_importance(graph, data_matrix)
        # add or update label information
        if self.relabel:
            graph = self._annotate_vector(graph, data_matrix)
        annotated_graph = _revert_edge_to_vertex_transform(graph)
        annotated_graph.graph = graph_dict
        return annotated_graph

    def _annotate_vector(self, graph, data_matrix):
        # annotate graph structure with vertex importance
        vertex_id = 0
        for v, d in graph.nodes_iter(data=True):
            if d.get('node', False):
                # annotate 'vector' information
                row = data_matrix.getrow(vertex_id)
                vec_dict = {str(index): value
                            for index, value in zip(row.indices, row.data)}
                # if an original label does not exist then save it, else do
                # nothing and preserve the information in original label
                if graph.node[v].get(self.key_original_label, False) is False:
                    graph.node[v][self.key_original_label] = \
                        graph.node[v][self.key_label]
                graph.node[v][self.key_label] = vec_dict
                # if a node does not have a 'entity' attribute then assign one
                # called 'vector' by default
                if graph.node[v].get(self.key_entity, False) is False:
                    graph.node[v][self.key_entity] = 'vector'
                vertex_id += 1
        return graph

    def _annotate_importance(self, graph, data_matrix):
        # compute distance from hyperplane as proxy of vertex importance
        if self.estimator is None:
            # if we do not provide an estimator then consider default margin of
            # 1/float(len(graph)) for all vertices
            margins = np.array([1 / float(len(graph))] * data_matrix.shape[0],
                               dtype=np.float64)
            predictions = np.array([1] * data_matrix.shape[0])
        else:
            if self.estimator.__class__.__name__ in ['SGDRegressor']:
                predicted_score = self.estimator.predict(data_matrix)
                predictions = np.array([1 if v >= 0 else -1 for v in predicted_score])
            else:
                predicted_score = self.estimator.decision_function(data_matrix)
                predictions = self.estimator.predict(data_matrix)
            margins = predicted_score - self.estimator.intercept_ + \
                self.estimator.intercept_ / float(len(graph))
        # annotate graph structure with vertex importance
        vertex_id = 0
        for v, d in graph.nodes_iter(data=True):
            if d.get('node', False):
                graph.node[v][self.key_class] = predictions[vertex_id]
                # annotate the 'importance' attribute with the margin
                graph.node[v][self.key_importance] = margins[vertex_id]
                # update the self.key_weight information as a linear
                # combination of the previous weight and the absolute margin
                if self.key_weight in graph.node[v] and self.reweight != 0:
                    graph.node[v][self.key_weight] = self.reweight * \
                        abs(margins[vertex_id]) +\
                        (1 - self.reweight) * \
                        graph.node[v][self.key_weight]
                # in case the original graph was not weighted then instantiate
                # the self.key_weight with the absolute margin
                else:
                    graph.node[v][self.key_weight] = abs(margins[vertex_id])
                vertex_id += 1
            if d.get('edge', False):  # keep the weight of edges
                # ..unless they were unweighted, in this case add unit weight
                if self.key_weight not in graph.node[v]:
                    graph.node[v][self.key_weight] = 1
        return graph

    def _compute_vertex_based_features(self, graph):
        feature_rows = []
        for v, d in graph.nodes_iter(data=True):
            # only for vertices of type 'node', i.e. not for the 'edge' type
            feature_list = defaultdict(lambda: defaultdict(float))
            if d.get('node', False):
                self._transform_vertex(graph, v, feature_list)
            feature_rows.append(self._normalization(feature_list))
        data_matrix = self._convert_dict_to_sparse_matrix(feature_rows)
        return data_matrix

    def components(self,
                   graphs,
                   estimator=None,
                   score_threshold=0,
                   min_size=2):
        """Iterator over the connected components.

        The nodes that have  a key_importance value below the score_threshold
        are deleted. The resulting components are returned.
        """
        annotated_graphs = self.annotate(graphs,
                                         estimator=estimator,
                                         reweight=1.0,
                                         relabel=False)
        for graph in annotated_graphs:
            connected_components = \
                self._extract_connected_components(
                    graph, score_threshold=score_threshold, min_size=min_size)
            for connected_component in connected_components:
                yield connected_component

    def _extract_connected_components(self,
                                      graph,
                                      score_threshold=0,
                                      min_size=2):
        # remove all vertices that have an importance score less then
        # score_threshold
        for v, d in graph.nodes_iter(data=True):
            if d.get(self.key_importance, False):
                if d[self.key_importance] < score_threshold:
                    graph.remove_node(v)
        for cc in nx.connected_component_subgraphs(graph):
            if len(cc) >= min_size:
                yield cc


def _label_preprocessing(graph, label_size=1,
                         key_label='label',
                         key_entity='entity',
                         discretizers={'entity': []},
                         bitmask=2 ** 20 - 1):
    try:
        graph.graph['label_size'] = label_size
        for n, d in graph.nodes_iter(data=True):
            # for dense or sparse vectors
            is_list = isinstance(d[key_label], list)
            is_dict = isinstance(d[key_label], dict)
            if is_list or is_dict:
                node_entity, data = _extract_entity_and_label(d, key_entity, key_label)
                if isinstance(d[key_label], list):
                    data = np.array(data, dtype=np.float64).reshape(1, -1)
                if isinstance(d[key_label], dict):
                    data = _convert_dict_to_sparse_vector(data)
                # create a list of integer codes of size: label_size
                # each integer code is determined as follows:
                # for each entity, use the correspondent
                # discretizers[node_entity] to extract
                # the id of the nearest cluster centroid, return the
                # centroid id as the integer code
                hlabel = []
                for i in range(label_size):
                    if len(discretizers[node_entity]) < i:
                        len_mod = \
                            len(discretizers[node_entity])
                        raise Exception('Error: discretizers for node entity: %s \
                            has length: %d but component %d was required'
                                        % (node_entity, len_mod, i))
                    predictions = \
                        discretizers[node_entity][i].predict(data)
                    if len(predictions) != 1:
                        raise Exception('Error: discretizer has not \
                            returned an individual prediction but\
                            %d predictions' % len(predictions))
                    discretization_code = predictions[0] + 1
                    code = fast_hash_2(hash(node_entity),
                                       discretization_code,
                                       bitmask)
                    hlabel.append(code)
                graph.node[n]['hlabel'] = hlabel
            elif isinstance(d[key_label], basestring):
                # copy a hashed version of the string for a number of times
                # equal to self.label_size in this way qualitative
                # ( i.e. string ) labels can be compared to the
                # discretized labels
                hlabel = int(hash(d[key_label]) & bitmask) + 1
                graph.node[n]['hlabel'] = [hlabel] * label_size
            else:
                raise Exception('ERROR: something went wrong, type of node label is unknown: \
                    %s' % d[key_label])
    except Exception as e:
        logger.debug('Failed iteration. Reason: %s' % e)
        logger.debug('Exception', exc_info=True)


def _extract_entity_and_label(d, key_entity, key_label):
    # determine the entity attribute
    # if the vertex does not have a 'entity' attribute then provide a
    # default one
    if d.get(key_entity, False):
        node_entity = d[key_entity]
    else:
        if isinstance(d[key_label], list):
            node_entity = 'vector'
        elif isinstance(d[key_label], dict):
            node_entity = 'sparse_vector'
        else:
            node_entity = 'default'
    data = d[key_label]
    return node_entity, data


def _convert_dict_to_sparse_vector(feature_row, bitmask=1048575):
    feature_size = bitmask + 2
    data, row, col = [], [], []
    if len(feature_row) == 0:
        # case of empty feature set for a specific instance
        row.append(0)
        col.append(0)
        data.append(0)
    else:
        for feature in feature_row:
            row.append(0)
            col.append(int(hash(feature) & bitmask) + 1)
            data.append(feature_row[feature])
    vec = csr_matrix((data, (row, col)), shape=(1, feature_size))
    return vec


def _edge_to_vertex_transform(original_graph):
    """Convert edges to nodes."""
    # if operating on graphs that have already been subject to the
    # edge_to_vertex transformation, then do not repeat the transformation
    # but simply return the graph
    if 'expanded' in original_graph.graph:
        return original_graph
    else:
        graph = nx.Graph()
        graph.graph.update(original_graph.graph)
        graph.graph['expanded'] = True
        # build a graph that has as vertices the original vertex set
        for n, d in original_graph.nodes_iter(data=True):
            d['node'] = True
            graph.add_node(n, d)
        # and in addition a vertex for each edge
        new_node_id = max(original_graph.nodes()) + 1
        for u, v, d in original_graph.edges_iter(data=True):
            if u != v:
                d['edge'] = True
                graph.add_node(new_node_id, d)
                # and the corresponding edges
                graph.add_edge(new_node_id, u, label=None)
                graph.add_edge(new_node_id, v, label=None)
                new_node_id += 1
        return graph


def _revert_edge_to_vertex_transform(original_graph):
    """Convert nodes of type 'edge' to edges."""
    if 'expanded' in original_graph.graph:
        # start from a copy of the original graph
        graph = nx.Graph(original_graph)
        _clean_graph(graph)
        # re-wire the endpoints of edge-vertices
        for n, d in original_graph.nodes_iter(data=True):
            if 'edge' in d:
                # extract the endpoints
                endpoints = [u for u in original_graph.neighbors(n)]
                if len(endpoints) != 2:
                    raise Exception('ERROR: more than 2 endpoints in \
                        a single edge: %s' % endpoints)
                u = endpoints[0]
                v = endpoints[1]
                # add the corresponding edge
                graph.add_edge(u, v, d)
                # remove the edge-vertex
                graph.remove_node(n)
        return graph
    else:
        return original_graph


def _clean_graph(graph):
    graph.graph.pop('expanded', None)
    for n, d in graph.nodes_iter(data=True):
        if 'node' in d:
            # remove stale information
            graph.node[n].pop('remote_neighbours', None)
            graph.node[n].pop('neigh_graph_hash', None)
            graph.node[n].pop('neigh_graph_weight', None)
            graph.node[n].pop('hlabel', None)
