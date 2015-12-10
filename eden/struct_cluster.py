from collections import defaultdict

import numpy as np

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import DBSCAN

from eden.graph import Vectorizer
from eden.converter.rna.rnaplfold import rnaplfold_to_eden
from eden.modifier.graph.vertex_attributes import list_reweight, listof_list_reweight
from eden.modifier.graph import vertex_attributes
from eden.modifier.graph.structure import contraction, contraction_modifier

import logging
logger = logging.getLogger(__name__)


class PreProcessor(object):

    def __init__(self, start_end_weight_list=[(-1, -1, 1)],
                 listof_start_end_weight_list=None,
                 scale=3,
                 level=2,
                 window_size=50,
                 max_bp_span=30,
                 avg_bp_prob_cutoff=0.6,
                 max_num_edges=1):
        self.start_end_weight_list = start_end_weight_list
        self.listof_start_end_weight_list = listof_start_end_weight_list
        self.scale = scale
        self.level = level
        self.window_size = window_size
        self.max_bp_span = max_bp_span
        self.avg_bp_prob_cutoff = avg_bp_prob_cutoff
        self.max_num_edges = max_num_edges

    def transform(self, seqs):
        graphs = rnaplfold_to_eden(seqs,
                                   window_size=self.window_size,
                                   max_bp_span=self.max_bp_span,
                                   avg_bp_prob_cutoff=self.avg_bp_prob_cutoff,
                                   max_num_edges=self.max_num_edges)

        if self.listof_start_end_weight_list is None:
            graphs = list_reweight(graphs, start_end_weight_list=self.start_end_weight_list)
            graphs = list_reweight(graphs, start_end_weight_list=self.start_end_weight_list,
                                   attribute='level')
        else:
            graphs = listof_list_reweight(graphs,
                                          listof_start_end_weight_list=self.listof_start_end_weight_list)
            graphs = listof_list_reweight(graphs,
                                          listof_start_end_weight_list=self.listof_start_end_weight_list,
                                          attribute='level')
        # annotate in node attribute 'type' the incident edges' labels
        graphs = vertex_attributes.incident_edge_label(graphs,
                                                       level=self.level,
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
                             scale=self.scale,
                             nesting=True)
        return graphs


class StructCluster(object):

    def __init__(self,
                 pre_processor=None,
                 vectorizer=Vectorizer(complexity=4, nbits=13),
                 clustering_algo=DBSCAN(),
                 factor=2,
                 threshold=2):
        self.pre_processor = pre_processor
        self.vectorizer = vectorizer
        self.clustering_algo = clustering_algo
        self.factor = factor
        self.threshold = threshold
        self.clusters = defaultdict(list)

    def fit(self, seqs):
        graphs = self.pre_processor.transform(seqs)
        self.data_matrix = self.vectorizer.transform(graphs)
        logger.debug('#instances:%d  #features:%d' % (self.data_matrix.shape[0], self.data_matrix.shape[1]))

    def predict(self, seqs=None):
        if seqs is not None:
            self.fit(seqs)
        distance_matrix = pairwise_distances(self.data_matrix)
        eps = np.mean(distance_matrix) - self.factor * np.std(distance_matrix)
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
            if len(cluster_seqs) > self.threshold and cluster_id != -1:
                self.clusters[counter] = cluster_seqs
                counter += 1
        logger.debug('num clusters: %d' % len(self.clusters))
        return self.predictions
