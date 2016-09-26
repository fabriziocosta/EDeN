#!/usr/bin/env python
"""Provides vectorization of graphs."""

import math
import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix
import multiprocessing as mp
from scipy.sparse import vstack
from collections import defaultdict, deque
import joblib
import networkx as nx
from eden import apply_async
from eden import chunks
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
                 min_r=0,
                 min_d=0,
                 weights_dict=None,
                 auto_weights=False,
                 nbits=20,
                 normalization=True,
                 inner_normalization=True,
                 positional=False,
                 discrete=False,
                 block_size=100,
                 n_jobs=-1,
                 key_label='label',
                 key_weight='weight',
                 key_nesting='nesting',
                 key_importance='importance',
                 key_class='class',
                 key_vec='vec',
                 key_svec='svec'):
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

        weights_dict : dict of floats
            Dictionary with keys = pairs (radius, distance) and
            value = weights.

        auto_weights : bool (default False)
            Flag to set to 1 the weight of the kernels for r=i, d=i
            for i in range(complexity)

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

        discrete: bool (default False)
            Flag to activate more efficient computation of vectorization
            considering only discrete labels and ignoring vector attributes.

        key_label : string (default 'label')
            The key used to indicate the label information in nodes.

        key_weight : string (default 'weight')
            The key used to indicate the weight information in nodes.

        key_nesting : string (default 'nesting')
            The key used to indicate the nesting type in edges.

        key_importance : string (default 'importance')
            The key used to indicate the importance information in nodes.

        key_class : string (default 'class')
            The key used to indicate the predicted class associated to
            the node.

        key_vec : string (default 'vec')
            The key used to indicate the vector label information in nodes.

        key_svec : string (default 'svec')
            The key used to indicate the sparse vector label information
            in nodes.
        """
        self.name = self.__class__.__name__
        self.__version__ = '1.0.1'
        self.complexity = complexity
        if r is None:
            r = complexity
        if d is None:
            d = complexity
        self.r = r
        self.d = d
        self.min_r = min_r
        self.min_d = min_d
        self.weights_dict = weights_dict
        if auto_weights:
            self.weights_dict = {(i, i): 1 for i in range(complexity)}
        self.nbits = nbits
        self.normalization = normalization
        self.inner_normalization = inner_normalization
        self.positional = positional
        self.discrete = discrete
        self.block_size = block_size
        self.n_jobs = n_jobs
        self.bitmask = pow(2, nbits) - 1
        self.feature_size = self.bitmask + 2
        self.key_label = key_label
        self.key_weight = key_weight
        self.key_nesting = key_nesting
        self.key_importance = key_importance
        self.key_class = key_class
        self.key_vec = key_vec
        self.key_svec = key_svec

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

    def __repr__(self):
        """string."""
        return serialize_dict(self.__dict__, offset='large')

    def save(self, model_name):
        """save."""
        joblib.dump(self, model_name, compress=1)

    def load(self, obj):
        """load."""
        self.__dict__.update(joblib.load(obj).__dict__)

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

        >>> # transforming the same graph (with different node-ids).
        >>> import networkx as nx
        >>> def get_path_graph(length=4):
        ...     g = nx.path_graph(length)
        ...     for n,d in g.nodes(data=True):
        ...         d['label'] = 'C'
        ...     for a,b,d in g.edges(data=True):
        ...         d['label'] = '1'
        ...     return g
        >>> g = get_path_graph(4)
        >>> g2 = get_path_graph(5)
        >>> g2.remove_node(0)
        >>> g[1][2]['label']='2'
        >>> g2[2][3]['label']='2'
        >>> v = Vectorizer()
        >>> def vec_to_hash(vec):
        ...     return  hash(tuple(vec.data + vec.indices))
        >>> vec_to_hash(v.transform([g])) == vec_to_hash (v.transform([g2]))
        True
        """
        if self.n_jobs == 1:
            return self._transform_serial(graphs)

        if self.n_jobs == -1:
            pool = mp.Pool(mp.cpu_count())
        else:
            pool = mp.Pool(self.n_jobs)

        results = [apply_async(
            pool, self._transform_serial,
            args=([subset_graphs]))
            for subset_graphs in chunks(graphs, self.block_size)]
        for i, p in enumerate(results):
            pos_data_matrix = p.get()
            if i == 0:
                data_matrix = pos_data_matrix
            else:
                data_matrix = vstack([data_matrix, pos_data_matrix])
        pool.close()
        pool.join()
        return data_matrix

    def _transform_serial(self, graphs):
        instance_id = None
        feature_rows = []
        for instance_id, graph in enumerate(graphs):
            self._test_goodness(graph)
            feature_rows.append(self._transform(graph))
        if instance_id is None:
            raise Exception('ERROR: something went wrong:\
                no graphs are present in current iterator.')
        data_matrix = self._convert_dict_to_sparse_matrix(feature_rows)
        return data_matrix

    def vertex_transform(self, graphs):
        """Transform a list of networkx graphs into a list of sparse matrices.

        Each matrix has dimension n_nodes x n_features, i.e. each vertex is
        associated to a sparse vector that encodes the neighborhood of the
        vertex up to radius + distance.

        Parameters
        ----------
        graphs : list[graphs]
            The input list of networkx graphs.

        Returns
        -------
        matrix_list : array-like, shape = [n_samples, [n_nodes, n_features]]
            Vector representation of each vertex in the input graphs.

        """
        if self.n_jobs == 1:
            return self._vertex_transform_serial(graphs)

        if self.n_jobs == -1:
            pool = mp.Pool(mp.cpu_count())
        else:
            pool = mp.Pool(self.n_jobs)

        results = [apply_async(
            pool, self._vertex_transform_serial,
            args=([subset_graphs]))
            for subset_graphs in chunks(graphs, self.block_size)]
        matrix_list = []
        for i, p in enumerate(results):
            matrix_list += p.get()
        pool.close()
        pool.join()
        return matrix_list

    def _vertex_transform_serial(self, graphs):
        matrix_list = []
        for instance_id, graph in enumerate(graphs):
            self._test_goodness(graph)
            graph = self._graph_preprocessing(graph)
            # extract per vertex feature representation
            data_matrix = self._compute_vertex_based_features(graph)
            matrix_list.append(data_matrix)
        return matrix_list

    def _test_goodness(self, graph):
        if graph.number_of_nodes() == 0:
            raise Exception('ERROR: something went wrong, empty graph.')

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
        data_matrix = csr_matrix((data, (row, col)),
                                 shape=shape, dtype=np.float64)
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
        _label_preprocessing(graph,
                             key_label=self.key_label,
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
        _clean_graph(graph)
        return self._normalization(feature_list)

    def _update_feature_list(self, node_feature_list, feature_list):
        for radius_dist_key in node_feature_list:
            for feature in node_feature_list[radius_dist_key]:
                val = node_feature_list[radius_dist_key][feature]
                feature_list[radius_dist_key][feature] += val

    def _transform_vertex(self, graph, vertex_v, feature_list):
        if self.discrete:
            # for all distances
            root_dist_dict = graph.node[vertex_v]['remote_neighbours']
            for distance in range(self.min_d * 2, (self.d + 1) * 2, 2):
                if distance in root_dist_dict:
                    node_set = root_dist_dict[distance]
                    for vertex_u in node_set:
                        self._transform_vertex_pair(graph, vertex_v, vertex_u,
                                                    distance, feature_list)
            self._transform_vertex_nesting(graph, vertex_v, feature_list)
        else:
            node_feature_list = defaultdict(lambda: defaultdict(float))
            # for all distances
            root_dist_dict = graph.node[vertex_v]['remote_neighbours']
            for distance in range(self.min_d * 2, (self.d + 1) * 2, 2):
                if distance in root_dist_dict:
                    node_set = root_dist_dict[distance]
                    for vertex_u in node_set:
                        self._transform_vertex_pair(graph, vertex_v, vertex_u,
                                                    distance,
                                                    node_feature_list)
            self._transform_vertex_nesting(graph, vertex_v, node_feature_list)
            node_feature_list = self._add_vector_labes(
                graph, vertex_v, node_feature_list)
            self._update_feature_list(node_feature_list, feature_list)
            node_sparse_feature_list = self._add_sparse_vector_labes(
                graph, vertex_v, node_feature_list)
            self._update_feature_list(node_sparse_feature_list, feature_list)

    def _add_vector_labes(self, graph, vertex_v, node_feature_list):
        # add the vector with an offset given by the feature, multiplied by val
        vec = graph.node[vertex_v].get(self.key_vec, None)
        if vec:
            vec_feature_list = defaultdict(lambda: defaultdict(float))
            for radius_dist_key in node_feature_list:
                for feature in node_feature_list[radius_dist_key]:
                    val = node_feature_list[radius_dist_key][feature]
                    for i, vec_val in enumerate(vec):
                        key = (feature + i) % self.bitmask
                        vec_feature_list[radius_dist_key][key] += val * vec_val
            node_feature_list = vec_feature_list
        return node_feature_list

    def _add_sparse_vector_labes(self, graph, vertex_v, node_feature_list):
        # add the vector with a feature resulting from hashing
        # the discrete labeled graph sparse encoding with the sparse vector
        # feature, the val is then multiplied.
        svec = graph.node[vertex_v].get(self.key_svec, None)
        if svec:
            vec_feature_list = defaultdict(lambda: defaultdict(float))
            for radius_dist_key in node_feature_list:
                for feature in node_feature_list[radius_dist_key]:
                    val = node_feature_list[radius_dist_key][feature]
                    for i in svec:
                        vec_val = svec[i]
                        key = fast_hash_2(feature, i, self.bitmask)
                        vec_feature_list[radius_dist_key][key] += val * vec_val
            node_feature_list = vec_feature_list
        return node_feature_list

    def _transform_vertex_nesting(self, graph, vertex_v, feature_list):
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

    def _transform_vertex_pair(self,
                               graph,
                               vertex_v,
                               vertex_u,
                               distance,
                               feature_list,
                               connection_weight=1):
        cw = connection_weight
        # for all radii
        for radius in range(self.min_r * 2, (self.r + 1) * 2, 2):
            # Note: to be compatible with external radius, distance
            # we need to revert to r/2 and d/2
            radius_dist_key = (radius / 2, distance / 2)
            if self.weights_dict is None or \
                    self.weights_dict.get(radius_dist_key, 0) != 0:
                self._transform_vertex_pair_valid(graph,
                                                  vertex_v,
                                                  vertex_u,
                                                  radius,
                                                  distance,
                                                  feature_list,
                                                  connection_weight=cw)

    def _transform_vertex_pair_valid(self,
                                     graph,
                                     vertex_v,
                                     vertex_u,
                                     radius,
                                     distance,
                                     feature_list,
                                     connection_weight=1):
        cw = connection_weight
        # we need to revert to r/2 and d/2
        radius_dist_key = (radius / 2, distance / 2)
        # reweight using external weight dictionary
        len_v = len(graph.node[vertex_v]['neigh_graph_hash'])
        len_u = len(graph.node[vertex_u]['neigh_graph_hash'])
        if radius < len_v and radius < len_u:
            # feature as a pair of neighborhoods at a radius,distance
            # canonicalization of pair of neighborhoods
            vertex_v_labels = graph.node[vertex_v]['neigh_graph_hash']
            vertex_v_hash = vertex_v_labels[radius]
            vertex_u_labels = graph.node[vertex_u]['neigh_graph_hash']
            vertex_u_hash = vertex_u_labels[radius]
            if vertex_v_hash < vertex_u_hash:
                first_hash, second_hash = (vertex_v_hash, vertex_u_hash)
            else:
                first_hash, second_hash = (vertex_u_hash, vertex_v_hash)
            feature = fast_hash_4(
                first_hash, second_hash, radius, distance, self.bitmask)
            # half features are those that ignore the central vertex v
            # the reason to have those is to help model the context
            # independently from the identity of the vertex itself
            half_feature = fast_hash_3(vertex_u_hash,
                                       radius, distance, self.bitmask)
            if graph.graph.get('weighted', False) is False:
                feature_list[radius_dist_key][feature] += cw
                feature_list[radius_dist_key][half_feature] += cw
            else:
                weight_v = graph.node[vertex_v]['neigh_graph_weight']
                weight_u = graph.node[vertex_u]['neigh_graph_weight']
                weight_vu_radius = weight_v[radius] + weight_u[radius]
                val = cw * weight_vu_radius
                feature_list[radius_dist_key][feature] += val
                half_val = cw * weight_u[radius]
                feature_list[radius_dist_key][half_feature] += half_val

    def _normalization(self, feature_list):
        # inner normalization per radius-distance
        feature_vector = {}
        for r_d_key in feature_list:
            features = feature_list[r_d_key]
            norm = 0
            for count in features.values():
                norm += count * count
            sqrt_norm = math.sqrt(norm)
            if self.weights_dict is not None:
                # reweight using external weight dictionary
                if self.weights_dict.get(r_d_key, None) is not None:
                    sqrtw = math.sqrt(self.weights_dict[r_d_key])
                    sqrt_norm = sqrt_norm / sqrtw
            for feature_id, count in features.items():
                if self.inner_normalization:
                    feature_vector_value = float(count) / sqrt_norm
                else:
                    feature_vector_value = count
                feature_vector[feature_id] = feature_vector_value
        # global normalization
        if self.normalization:
            normalized_feature_vector = {}
            total_norm = 0.0
            for value in feature_vector.values():
                total_norm += value * value
            sqrt_total_norm = math.sqrt(float(total_norm))
            for feature_id, value in feature_vector.items():
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
        # list all hashed labels at increasing distances
        hash_list = []
        # for all distances
        root_dist_dict = graph.node[root]['remote_neighbours']
        for node_set in root_dist_dict.values():
            # create a list of hashed labels
            hash_label_list = []
            for v in node_set:
                # compute the vertex hashed label by hashing the hlabel
                # field
                # with the degree of the vertex (obtained as the size of
                # the adjacency dictionary for the vertex v)
                # or, in case positional is set, using the relative
                # position of the vertex v w.r.t. the root vertex
                if self.positional:
                    vhlabel = fast_hash_2(
                        graph.node[v]['hlabel'],
                        root - v)
                else:
                    vhlabel = \
                        fast_hash_2(graph.node[v]['hlabel'],
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
        graph.node[root]['neigh_graph_hash'] = hash_neighborhood

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
        for distance, node_set in root_dist_dict.items():
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

    def annotate(self,
                 graphs,
                 estimator=None,
                 reweight=1.0,
                 vertex_features=False):
        """Return graphs with extra attributes: importance and features.

        Given a list of networkx graphs, if the given estimator is not None and
        is fitted, return a list of networkx graphs where each vertex has
        additional attributes with key 'importance' and 'weight'.
        The importance value of a vertex corresponds to the
        part of the score that is imputable to the neighborhood of radius r+d
        of the vertex. The weight value is the absolute value of importance.
        If vertex_features is True then each vertex has additional attributes
        with key 'features' and 'vector'.

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

        vertex_features : bool (default false)
            Flag to compute the sparse vector encoding of all features that
            have that vertex as root. An attribute with key 'features' is
            created for each node that contains a CRS scipy sparse vector,
            and an attribute with key 'vector' is created that contains a
            python dictionary to store the key, values pairs.
        """
        self.estimator = estimator
        self.reweight = reweight
        self.vertex_features = vertex_features

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
        if self.vertex_features:
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
                graph.node[v]['features'] = row
                vec_dict = {str(index): value
                            for index, value in zip(row.indices, row.data)}
                graph.node[v]['vector'] = vec_dict
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
                predictions = np.array([1 if v >= 0 else -1
                                        for v in predicted_score])
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
            if d.get('node', False):
                feature_list = defaultdict(lambda: defaultdict(float))
                self._transform_vertex(graph, v, feature_list)
                feature_rows.append(self._normalization(feature_list))
        data_matrix = self._convert_dict_to_sparse_matrix(feature_rows)
        return data_matrix


def _label_preprocessing(graph,
                         key_label='label',
                         bitmask=2 ** 20 - 1):
    for n, d in graph.nodes_iter(data=True):
        graph.node[n]['hlabel'] = int(hash(d[key_label]) & bitmask) + 1


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
