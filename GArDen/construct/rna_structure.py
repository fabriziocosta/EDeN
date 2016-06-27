#!/usr/bin/env python
"""Provides construction of sequences."""

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.neighbors import NearestNeighbors
from eden.graph import Vectorizer
from eden.util import vectorize
from GArDen.model import KNNWrapper
from sklearn import metrics
import time
import datetime
from itertools import izip
import numpy as np
from GArDen.transform.importance_annotation import AnnotateImportance
from GArDen.interfaces import transform, model, predict
from GArDen.transform.node import MarkWithIntervals
from GArDen.transform.node import MarkKTop
from GArDen.transform.node import ReplaceWithAllCombinations
from GArDen.transform.rna_structure import PathGraphToRNAFold


import logging
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
class KNNManager(object):
    """Efficiently compute knn allowing for dynamic insertion."""

    def __init__(self, n_neighbors=3, complexity=3):
        """Constructor."""
        self.n_neighbors = n_neighbors
        self.complexity = complexity
        self.distances_to_known_graphs = None
        self.distances_to_candidate_graphs = None

    def setup(self, known_graphs=None, candidate_graphs=None):
        """Setup."""
        # compute the nearest neighbors for the 'proposal_graphs' w.r.t. the
        # known graphs in the list 'known_graphs'
        parameters_priors = dict(n_neighbors=self.n_neighbors)
        parameters_priors.update(dict(vectorizer__complexity=self.complexity,
                                      vectorize__n_jobs=-1,
                                      vectorize__fit_flag=False,
                                      vectorize__n_blocks=5,
                                      vectorize__block_size=100))
        fit_wrapped_knn_predictor_known = \
            model(known_graphs,
                  program=KNNWrapper(program=NearestNeighbors()),
                  parameters_priors=parameters_priors)
        # compute distances of candidate_graphs to known_graphs
        knn_candidate_graphs = predict(candidate_graphs,
                                       program=fit_wrapped_knn_predictor_known)
        knn_candidate_graphs = list(knn_candidate_graphs)
        self.distances_to_known_graphs = []
        for knn_candidate_graph in knn_candidate_graphs:
            distances = knn_candidate_graph.graph['distances']
            self.distances_to_known_graphs.append(distances)
        # compute candidate_graphs encodings
        self.candidate_graphs_data_matrix = \
            vectorize(candidate_graphs,
                      vectorizer=Vectorizer(complexity=self.complexity),
                      block_size=400, n_jobs=-1)

    def average_distances(self):
        """Average distances."""
        return [np.mean(ds) for ds in self.distances_to_known_graphs]

    def add_element(self, candidate_graph_id):
        """Add element."""
        candidate_graph_vector = \
            self.candidate_graphs_data_matrix[candidate_graph_id]
        # compute the distance of candidate_graph to
        # all others candidate_graphs
        ds = metrics.pairwise.pairwise_distances(
            self.candidate_graphs_data_matrix,
            candidate_graph_vector,
            metric='euclidean')
        ds = ds.T.tolist()[0]
        # update self.distance_list by inserting the new element
        # and trimming the list
        new_distances_to_known_graphs = []
        for d, d_list in izip(ds, self.distances_to_known_graphs):
            augmented_list = list(d_list) + [d]
            sorted_augmented_list = sorted(augmented_list)
            trimmed_sorted_augmented_list = sorted_augmented_list[:-1]
            new_distances_to_known_graphs.append(trimmed_sorted_augmented_list)
        self.distances_to_known_graphs = new_distances_to_known_graphs


# ------------------------------------------------------------------------------


class RNAStructureGenerator(BaseEstimator, TransformerMixin):
    """RNAStructureGenerator."""

    def __init__(self,
                 known_graphs=None,
                 exclusion_quadruples=None,
                 fit_wrapped_predictor=None,
                 n_substitutions=3,
                 n_neighbors=3,
                 exploration_vs_exploitation_tradeoff=.1,
                 n_proposals=10,
                 optimization_mode=False,
                 seq_to_structure_prog=PathGraphToRNAFold(),
                 label_list=['A', 'C', 'G', 'U'],
                 random_state=1):
        """Generate sequences.

        Start from input sequences that are 'better' if enhance is set to True
        ('worse' otherwise) given the set of sequences used in the fit phase.

        Parameters
        ----------
        known_graphs : networkx graphs
            List of graphs that are already known.

        n_substitutions : int
           The number of most important nodes to change in the proposal.

        exclusion_quadruples : list of quadruples
            Each quadruple encodes the start, end position and an attribute
            and a value.
            Ex.
            [(-1,-1,'exclude',False),(0,19,'exclude',True),(55,75,'exclude',True)].
            This is used to endow the graphs' nodes with the
            attribute 'exclude' that can be used to define a forbidden area in
            the instance where mutations are not allowed.

        fit_wrapped_predictor: scikit wrapped regressor
            Already fit regressor.

        n_neighbors : int
            The number of nearest neighbors that are used to compute
            the acquisition score.

        exploration_vs_exploitation_tradeoff: float
            The lambda parameter in the acquisition function. >1 biases towards
            only very similar examples to be considered.

        n_proposals : int
            Number of instances returned

        optimization_mode : bool
            If True	use seed graphs as they are for selection.

        seq_to_structure_prog : callable
                Wrapped program to fold RNA sequences into structure.

        label_list : list of chars (default: ['A', 'C', 'G', 'U'])
            List of chars for the random replacement.

        random_state: int (default 1)
            The random seed.
        """
        self.known_graphs = known_graphs
        self.n_substitutions = n_substitutions
        self.exclusion_quadruples = exclusion_quadruples
        self.fit_wrapped_predictor = fit_wrapped_predictor
        self.n_neighbors = n_neighbors
        self.exploration_vs_exploitation_tradeoff = \
            exploration_vs_exploitation_tradeoff
        self.n_proposals = n_proposals
        self.optimization_mode = optimization_mode
        self.seq_to_structure_prog = seq_to_structure_prog
        self.label_list = label_list
        self.random_state = random_state

    def candidate_generator(self, seed_graphs):
        """Generate candidates.

        Parameters
        ----------
        seed_graphs : networkx graphs
            The iterator over the seed graphs, i.e. the graphs that are used as
            a starting point for the proposal.
        """
        start = time.time()
        graphs = transform(seed_graphs,
                           program=AnnotateImportance(
                               program=self.fit_wrapped_predictor.program))
        graphs = list(graphs)

        # mark the position of nodes with the attribute 'exclude' to remove
        # the influence of primers
        graphs = transform(graphs,
                           program=MarkWithIntervals(
                               quadruples=self.exclusion_quadruples))

        # find the ktop largest (reverse=True) values for the
        # attribute='importance' in the vertices of a graph
        # and add an attribute to each vertex that is 'selected'=True
        # if the node is among the ktop
        graphs = transform(graphs,
                           program=MarkKTop(attribute='importance',
                                            exclude_attribute='exclude',
                                            ktop=self.n_substitutions,
                                            reverse=True,
                                            mark_attribute='selected'))

        # generate graphs that have all possible combination of symbols in
        # the nodes marked by MarkTop
        graphs = transform(graphs, program=ReplaceWithAllCombinations(
            attribute='selected', label_list=self.label_list))

        # refold the sequences to account for structural changes
        graphs = transform(graphs, program=self.seq_to_structure_prog)

        # return the candidate graphs
        candidate_graphs = list(graphs)
        delta_time = datetime.timedelta(seconds=(time.time() - start))
        logger.info('Candidate generation took: %s' % (str(delta_time)))
        logger.info('Number of candidates: %d' % (len(candidate_graphs)))

        return candidate_graphs

    def _acquisition_func(self, score,
                          uncertainty, exploration_vs_exploitation_tradeoff):
        acquisition_val =\
            score + exploration_vs_exploitation_tradeoff * uncertainty
        logger.debug('score:%.4f  uncertainty:%.4f  = acquisition:%.4f' %
                     (score, uncertainty, acquisition_val))
        return acquisition_val

    def _acquisition(self, scores, uncertainties,
                     exploration_vs_exploitation_tradeoff=0.1):
        acquisition_vals = \
            [self._acquisition_func(score,
                                    uncertainty,
                                    exploration_vs_exploitation_tradeoff)
             for score, uncertainty in zip(scores, uncertainties)]
        maximal_id = np.argmax(acquisition_vals)
        return maximal_id

    def efficient_selection(self,
                            candidate_graphs,
                            known_graphs=None):
        """Propose a small number of alternative structures.

        Parameters
        ----------
        candidate_graphs : networkx graphs
            The iterator over the seed graphs, i.e. the graphs that are used
            as a starting point for the proposal.

        known_graphs : networkx graphs
            The iterator over the already known graphs. These are used to bias
            the exploration towards less similar proposals.
        """
        start = time.time()

        candidate_graphs = transform(
            candidate_graphs,
            program=AnnotateImportance(
                program=self.fit_wrapped_predictor.program))
        candidate_graphs = list(candidate_graphs)

        # transform graphs according to importance
        # this allows similarity notion to be task dependent
        known_graphs = transform(
            known_graphs,
            program=AnnotateImportance(
                program=self.fit_wrapped_predictor.program))
        known_graphs = list(known_graphs)
        # store the nearest neighbors in knn_manager
        # compute the k nearest neighbors distances of each proposal graph
        knn_manager = KNNManager(n_neighbors=self.n_neighbors, complexity=3)
        knn_manager.setup(known_graphs=known_graphs,
                          candidate_graphs=candidate_graphs)
        delta_time = datetime.timedelta(seconds=(time.time() - start))
        logger.info('Knn computation took: %s' % (str(delta_time)))

        # compute predictions
        predicted_graphs = predict(candidate_graphs,
                                   program=self.fit_wrapped_predictor)
        predicted_graphs = list(predicted_graphs)
        scores = np.array([graph.graph['score']
                           for graph in predicted_graphs]).reshape(-1, 1)

        # iterations
        tradeoff = self.exploration_vs_exploitation_tradeoff
        selection_ids = []
        for i in range(self.n_proposals):
            uncertainties = knn_manager.average_distances()
            # run the acquisition function (n_proposals times)
            # and return best_id
            maximal_id = self._acquisition(
                scores,
                uncertainties,
                exploration_vs_exploitation_tradeoff=tradeoff)
            # update distances with new selection
            knn_manager.add_element(maximal_id)
            # store id
            selection_ids.append(maximal_id)
            graph = candidate_graphs[maximal_id]
            logger.debug('>%s' % graph.graph['header'])
            logger.debug(graph.graph['sequence'])
        return selection_ids

    def transform(self, seed_graphs=None):
        """Transform."""
        if self.optimization_mode is True:
            candidate_graphs = seed_graphs
        else:
            candidate_graphs = self.candidate_generator(seed_graphs)
        selection_ids = self.efficient_selection(candidate_graphs,
                                                 self.known_graphs)
        for id in selection_ids:
            yield candidate_graphs[id]
