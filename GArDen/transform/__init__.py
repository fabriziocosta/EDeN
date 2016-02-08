from sklearn.base import BaseEstimator, TransformerMixin

import logging
logger = logging.getLogger(__name__)


class DeleteEdge(BaseEstimator, TransformerMixin):
    """
    Delete edges.

    Delete an edge if its dictionary has a key equal to 'attribute' and the 'condition'
    is true between 'value' and the value associated to key=attribute.
    """

    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, graphs, attribute=None, value=None, condition=lambda x, y: x == y):
        '''
        Delete an edge if its dictionary has a key equal to 'attribute' and the 'condition'
        is true between 'value' and the value associated to key=attribute.

        Parameters
        ----------
        graphs : iterator over path graphs of RNA sequences

        attribute : string
            The key of the edge dictionary.
        value : string
            The value associated to the attribute.
        Returns
        -------
        Iterator over networkx graphs.
        '''
        try:
            for graph in graphs:
                for edge_src, edge_dest, edge_dict in graph.edges_iter(data=True):
                    if attribute in edge_dict and condition(edge_dict[attribute], value) is True:
                        graph.remove_edge(edge_src, edge_dest)
                yield graph
        except Exception as e:
            print e.__doc__
            print e.message


class DeleteNode(BaseEstimator, TransformerMixin):

    """
    Delete a node if its dictionary has a key equal to 'attribute' and the 'condition'
    is true between 'value' and the value associated to key=attribute.

    Parameters
    ----------
    graphs : iterator over path graphs of RNA sequences

    attribute : string
        The key of the node dictionary.
    value : string
        The value associated to the attribute.
    """

    def __init__(self, attribute_value_dict=None, mode='AND'):
        self.attribute_value_dict = attribute_value_dict
        self.mode = mode

    def fit(self):
        return self

    def transform(self, graphs):
        '''
        TODO
        '''
        try:
            for graph in graphs:
                for n, node_dict in graph.nodes_iter(data=True):
                    if self.mode == 'AND':
                        trigger = True
                        for attribute in self.attribute_value_dict:
                            value = self.attribute_value_dict[attribute]
                            if attribute not in node_dict or node_dict[attribute] != value:
                                trigger = False
                    elif self.mode == 'OR':
                        trigger = False
                        for attribute in self.attribute_value_dict:
                            value = self.attribute_value_dict[attribute]
                            if attribute in node_dict and node_dict[attribute] == value:
                                trigger = True
                    if trigger is True:
                        graph.remove_node(n)
                yield graph
        except Exception as e:
            print e.__doc__
            print e.message