from eden import util
from sklearn.base import BaseEstimator, TransformerMixin
import json
from networkx.readwrite import json_graph

import logging
logger = logging.getLogger(__name__)


class GraphToNodeLinkData(BaseEstimator, TransformerMixin):

    """
    Transform graph into text.

    """

    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, graphs):
        '''Todo.'''
        try:
            for graph in graphs:
                json_data = json_graph.node_link_data(graph)
                serial_data = json.dumps(json_data)
                yield serial_data
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)


class NodeLinkDataToGraph(BaseEstimator, TransformerMixin):

    """
    Transform text into graphs.

    """

    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, data):
        '''Todo.'''
        try:
            for serial_data in util.read(data):
                py_obj = json.loads(serial_data)
                graph = json_graph.node_link_graph(py_obj)
                yield graph
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)
