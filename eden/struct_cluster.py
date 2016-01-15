from collections import defaultdict
import random

import numpy as np

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import DBSCAN

from eden.graph import Vectorizer
from eden.converter.rna.rnaplfold import rnaplfold_to_eden
from eden.modifier.graph.vertex_attributes import list_reweight, listof_list_reweight
from eden.modifier.graph import vertex_attributes
from eden.modifier.graph.structure import contraction, contraction_modifier
from eden.util import serialize_dict

import logging
logger = logging.getLogger(__name__)


class Transformer(object):

    def __init__(self, start_end_weight_list=[(-1, -1, 1)],
                 contraction_weight_scaling_factor=2.0,
                 contraction_level=2,
                 window_size=50,
                 max_bp_span=30,
                 avg_bp_prob_cutoff=0.6,
                 max_num_edges=1):
        """Transforms sequences to graphs that encode secondary structure information
            and weights nucleotides according to user defined list of intervals.

        Parameters
        ----------
        start_end_weight_list : list or triplets
            Each triplet specifies the start, end and (constant) weight. Positions are 0 based.

        contraction_weight_scaling_factor : float (default 2.0)
            Multiplicative factor that scales the weight attribute on contracted nodes.

        contraction_level : int (default 2)
            Level of contraction.

        window_size: int (default 50)
            Maximal window size for plfold.

        max_bp_span: int (default 30)
            Maximal base pair span interval for plfold.

        avg_bp_prob_cutoff: float (default 0.6)
            Edges probability threshold for plfold.

        max_num_edges: int (default 1)
            Maximal number of edges per nucleotide (not counting backbone edges)
        """
        self.name = self.__class__.__name__
        self.start_end_weight_list = start_end_weight_list
        self.contraction_weight_scaling_factor = contraction_weight_scaling_factor
        self.contraction_level = contraction_level
        self.window_size = window_size
        self.max_bp_span = max_bp_span
        self.avg_bp_prob_cutoff = avg_bp_prob_cutoff
        self.max_num_edges = max_num_edges

    def __repr__(self):
        return serialize_dict(self.__dict__, offset='large')

    def transform(self, seqs, listof_start_end_weight_list=None):
        """Transforms the sequences in input into graphs.

        Parameters
        ----------
        seqs : iterable strings
            Input sequences.

        listof_start_end_weight_list : list of start_end_weight_lists
            Each element in listof_start_end_weight_list specifies the start_end_weight_list
            for a single graph.

        Returns
        -------
        graphs : iterable graphs in Networkx format
            List of graphs resulting from the transformation of sequences into folded structures.
        """

        graphs = rnaplfold_to_eden(seqs,
                                   window_size=self.window_size,
                                   max_bp_span=self.max_bp_span,
                                   avg_bp_prob_cutoff=self.avg_bp_prob_cutoff,
                                   max_num_edges=self.max_num_edges)

        if listof_start_end_weight_list is None:
            graphs = list_reweight(graphs, start_end_weight_list=self.start_end_weight_list)
            graphs = list_reweight(graphs, start_end_weight_list=self.start_end_weight_list,
                                   attribute='level')
        else:
            graphs = listof_list_reweight(graphs,
                                          listof_start_end_weight_list=listof_start_end_weight_list)
            graphs = listof_list_reweight(graphs,
                                          listof_start_end_weight_list=listof_start_end_weight_list,
                                          attribute='level')
        # annotate in node attribute 'type' the incident edges' labels
        graphs = vertex_attributes.incident_edge_label(graphs,
                                                       level=self.contraction_level,
                                                       output_attribute='type',
                                                       separator='.')
        # reduce all 'label' attributes of contracted nodes to a histogram to be written in the
        # 'label' attribute of the resulting graph
        label_modifier = contraction_modifier(attribute_in='type',
                                              attribute_out='label',
                                              reduction='set_categorical')
        # reduce all 'weight' attributes of contracted nodes using a sum to be written in the
        # 'weight' attribute of the resulting graph
        weight_modifier = contraction_modifier(attribute_in='weight',
                                               attribute_out='weight',
                                               reduction='average')
        modifiers = [label_modifier, weight_modifier]
        # contract the graph on the 'type' attribute
        graphs = contraction(graphs,
                             contraction_attribute='type',
                             modifiers=modifiers,
                             contraction_weight_scaling_factor=self.contraction_weight_scaling_factor,
                             nesting=True)
        return graphs


class StructCluster(object):

    def __init__(self,
                 transformer=None,
                 vectorizer=Vectorizer(complexity=4, nbits=13),
                 clustering_algo=DBSCAN(),
                 distance_std_factor=2,
                 min_cluster_size=2,
                 random_state=1):
        """Cluster sequences according to regions of interest and structural folding.

        Parameters
        ----------
        transformer : initialized PreProcessor object
            Transforms sequences to graphs that encode secondary structure information
            and weights nucleotides according to user defined list of intervals.

        vectorizer : initialized Vectorizer object
            Transforms graphs to sparse vectors.

        clustering_algo : scikit-learn clustering algorithm
            Clusters sparse vectors in a finite number of classes.

        distance_std_factor : int (default 2)
            How many standard deviations less than the mean pairwise distance is the maximal
            distance required to join an instance in a cluster.

        min_cluster_size : int (default 2)
            Minimal size of any cluster.

        random_state: int (default 1)
            Random seed.

        Attributes
        ----------
        predictions : list(int)
            List of cluster ids, one per instance.

        clusters : defaultdict(list)
            Dictionary with cluster id as key and list of sequences as variable.

        data_matrix : Scipy sparse matrix (Compressed Sparse Row matrix)
            List of sparse vectors resulting from the transformation of sequences into structures.
        """
        self.name = self.__class__.__name__
        self.transformer = transformer
        self.vectorizer = vectorizer
        self.clustering_algo = clustering_algo
        self.distance_std_factor = distance_std_factor
        self.min_cluster_size = min_cluster_size
        self.clusters = defaultdict(list)
        self.predictions = list()
        self.data_matrix = None
        self.random_state = random_state
        random.seed(self.random_state)

    def __repr__(self):
        return serialize_dict(self.__dict__, offset='large')

    def set_params(self, **params):
        """Set the parameters.

        Returns
        -------
        self
        """

        if not params:
            return self
        valid_params = self.__dict__
        for key, value in params.iteritems():
            if key in valid_params:
                self.__dict__.update({key: value})
        return self

    def fit(self, seqs=None, listof_start_end_weight_list=None):
        """Transforms the sequences in input into sparse vectors using
        the transformer and the vectorizer.

        Parameters
        ----------
        seqs : iterable strings
            Input sequences.

        listof_start_end_weight_list : list of start_end_weight_lists
            Each element in listof_start_end_weight_list specifies the start_end_weight_list
            for a single graph.

        Attributes
        ----------
        data_matrix : Scipy sparse matrix (Compressed Sparse Row matrix)
            List of sparse vectors resulting from the transformation of sequences into structures.


        Returns
        -------
        self
        """
        graphs = self.transformer.transform(seqs, listof_start_end_weight_list=listof_start_end_weight_list)
        self.data_matrix = self.vectorizer.transform(graphs)
        logger.debug('#instances:%d  #features:%d' % (self.data_matrix.shape[0], self.data_matrix.shape[1]))
        return self

    def predict(self, seqs=None):
        """Transforms the sequences in input into sparse vectors using
        the transformer and the vectorizer.

        Parameters
        ----------
        seqs : iterable strings
            Input sequences.

        Attributes
        ----------
        predictions : list(int)
            List of cluster ids, one per instance.

        clusters : defaultdict(list)
                Dictionary with cluster id as key and list of sequences as variable.

        Returns
        -------
        predictions : list(int)
            List of cluster ids, one per instance.
        """
        if self.data_matrix is None:
            raise Exception('using predict on a non fit object; use fit first')

        distance_matrix = pairwise_distances(self.data_matrix)
        eps = np.mean(distance_matrix) - self.distance_std_factor * np.std(distance_matrix)
        logger.debug('eps: %.3f' % eps)

        self.clustering_algo.set_params(eps=eps)
        self.predictions = self.clustering_algo.fit_predict(self.data_matrix)

        clustered_seqs = defaultdict(list)
        for cluster_id, seq in zip(self.predictions, seqs):
            clustered_seqs[cluster_id].append(seq)

        counter = 0
        for cluster_id in clustered_seqs:
            cluster_seqs = [seq for seq in clustered_seqs[cluster_id]]
            logger.info('cluster id: %d  num seqs: %d' % (cluster_id, len(cluster_seqs)))
            if len(cluster_seqs) > self.min_cluster_size and cluster_id != -1:
                self.clusters[counter] = cluster_seqs
                counter += 1
        logger.debug('num clusters: %d' % len(self.clusters))
        return self.predictions
