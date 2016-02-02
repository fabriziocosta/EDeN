from sklearn.base import BaseEstimator, TransformerMixin
from itertools import izip

import logging
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------


class ReweightDictionary(BaseEstimator, TransformerMixin):

    """
    Missing.
    """

    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, graphs,
                  weight_dict=None):
        """
        Assign weights according to ...
        """
        try:
            # TODO
            pass
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)


# ----------------------------------------------------------------------------------------------

class WeightWithIntervals(BaseEstimator, TransformerMixin):

    """
    Missing.
    """

    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, graphs,
                  attribute='weight',
                  start_end_weight_list=None,
                  listof_start_end_weight_list=None):
        """
        Assign weights according to a list of triplets: each triplet defines the start, end position
        and the (uniform) weight of the region; the order of the triplets matters: later triplets override
        the weight specification of previous triplets. The special triplet (-1,-1,w) assign a default weight
        to all nodes.

        If listof_start_end_weight_list is available then each element in listof_start_end_weight_list
        specifies the start_end_weight_list for a single graph.
        """
        try:
            if listof_start_end_weight_list is not None:
                for graph, start_end_weight_list in izip(graphs, listof_start_end_weight_list):
                    graph = self.start_end_weight_reweight(graph,
                                                           start_end_weight_list=start_end_weight_list,
                                                           attribute=attribute)
                    yield graph
            elif start_end_weight_list is not None:
                for graph in graphs:
                    graph = self.start_end_weight_reweight(graph,
                                                           start_end_weight_list=start_end_weight_list,
                                                           attribute=attribute)
                    yield graph
            else:
                raise Exception('Either start_end_weight_list or listof_start_end_weight_list \
                    must be non empty.')
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def start_end_weight_reweight(self, graph, start_end_weight_list=None, attribute='weight'):
        for start, end, weight in start_end_weight_list:
            # iterate over nodes
            for n, d in graph.nodes_iter(data=True):
                if 'position' not in d:
                    # assert nodes must have position attribute
                    raise Exception('Nodes must have "position" attribute')
                # given the 'position' attribute of node assign the weight accordingly
                pos = d['position']
                if pos >= start and pos < end:
                    graph.node[n][attribute] = weight
                if start == -1 and end == -1:
                    graph.node[n][attribute] = weight
        return graph

# ----------------------------------------------------------------------------------------------


class RelabelWithLabelOfIncidentEdges(BaseEstimator, TransformerMixin):

    """
    Missing.
    """

    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, graphs, output_attribute='type', separator='', distance=1):
        '''
        Delete an edge if its dictionary has a key equal to 'attribute' and the 'condition'
        is true between 'value' and the value associated to key=attribute.

        Parameters
        ----------
        graphs : iterator over path graphs of RNA sequences

        output_attribute : string
            The key of the node dictionary where to write the result.

        separator : string
            The string used to separate the sorted concatenation of labels.

        distance : integer (default 1)
            The neighborhood radius explored.

        Returns
        -------
        Iterator over networkx graphs.
        '''
        try:
            for graph in graphs:
                # iterate over nodes
                for n, d in graph.nodes_iter(data=True):
                    # for all neighbors
                    edge_labels = []
                    if distance == 1:
                        edge_labels += [ed.get('label', 'N/A') for u, v, ed in graph.edges_iter(n, data=True)]
                    elif distance == 2:
                        neighbors = graph.neighbors(n)
                        for nn in neighbors:
                            # extract list of edge labels
                            edge_labels += [ed.get('label', 'N/A')
                                            for u, v, ed in graph.edges_iter(nn, data=True)]
                    else:
                        raise Exception('Unknown distance: %s' % distance)
                    # consider the sorted serialization of all labels as a type
                    vertex_type = separator.join(sorted(edge_labels))
                    graph.node[n][output_attribute] = vertex_type
                yield graph
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)
