#!/usr/bin/env python
"""Provides conversion from files to graphs."""

from eden import util
from sklearn.base import BaseEstimator, TransformerMixin
import json
import networkx as nx
from networkx.readwrite import json_graph

import logging
logger = logging.getLogger(__name__)


class GraphToNodeLinkData(BaseEstimator, TransformerMixin):
    """Transform graph into text."""

    def transform(self, graphs):
        """Transform."""
        try:
            for graph in graphs:
                json_data = json_graph.node_link_data(graph)
                serial_data = json.dumps(json_data)
                yield serial_data
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)


# ----------------------------------------------------------------------------

class NodeLinkDataToGraph(BaseEstimator, TransformerMixin):
    """Transform text into graphs."""

    def transform(self, data):
        """Transform."""
        try:
            for serial_data in util.read(data):
                py_obj = json.loads(serial_data)
                graph = json_graph.node_link_graph(py_obj)
                yield graph
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)


# ----------------------------------------------------------------------------

class GspanToGraph(BaseEstimator, TransformerMixin):
    """Transform gSpan text into graphs."""

    def transform(self, data):
        """Take a string list in the extended gSpan format and yields NetworkX graphs.

        Parameters
        ----------
        data : string or list
            data source, can be a list of strings, a file name or a url

        Returns
        -------
        iterator over networkx graphs

        Raises
        ------
        exception: when a graph is empty
        """
        try:
            header = ''
            string_list = []
            for line in util.read(data):
                if line.strip():
                    if line[0] in ['g', 't']:
                        if string_list:
                            yield self._gspan_to_networkx(header, string_list)
                        string_list = []
                        header = line
                    else:
                        string_list += [line]
            if string_list:
                yield self._gspan_to_networkx(header, string_list)
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _gspan_to_networkx(self, header, lines):
        # remove empty lines
        lines = [line for line in lines if line.strip()]

        graph = nx.Graph(id=header)
        for line in lines:
            tokens = line.split()
            fc = tokens[0]

            # process vertices
            if fc in ['v', 'V']:
                id = int(tokens[1])
                label = tokens[2]
                weight = 1.0

                # uppercase V indicates no-viewpoint, in the new EDeN this
                # is simulated via a smaller weight
                if fc == 'V':
                    weight = 0.1

                graph.add_node(id, ID=id, label=label, weight=weight)

                # extract the rest of the line  as a JSON string that
                # contains all attributes
                attribute_str = ' '.join(tokens[3:])
                if attribute_str.strip():
                    attribute_dict = json.loads(attribute_str)
                    graph.node[id].update(attribute_dict)

            # process edges
            elif fc == 'e':
                src = int(tokens[1])
                dst = int(tokens[2])
                label = tokens[3]
                graph.add_edge(src, dst, label=label, len=1)
                attribute_str = ' '.join(tokens[4:])
                if attribute_str.strip():
                    attribute_dict = json.loads(attribute_str)
                    graph.edge[src][dst].update(attribute_dict)
            else:
                logger.debug('line begins with unrecognized code: %s' % fc)
        if len(graph) == 0:
            raise Exception('ERROR: generated empty graph. \
                Perhaps wrong format?')
        return graph
