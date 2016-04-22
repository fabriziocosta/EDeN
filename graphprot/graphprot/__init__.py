"""Provides graphprot-specific functionality."""

from sklearn.base import BaseEstimator, TransformerMixin

import logging
logger = logging.getLogger(__name__)


class WeightViewpointCapitalization(BaseEstimator, TransformerMixin):
    """
    Set weights of graph nodes according to the capitalization of their "label"
    attribute, (e.g. the nucleotide). Normalizes the label attributes, i.e.
    all label attributes are converted to uppercase letters and T is converted
    to U.

    >>> # run a simple weighting operation
    >>> from GArDen.convert.sequence import SeqToPathGraph
    >>> graphs = SeqToPathGraph().transform([("ID0", "acgutACGUTacgut")])
    >>> wgraphs = WeightViewpointCapitalization().transform(graphs)
    >>> graph = wgraphs.next()
    >>> # evaluate sequence attribute
    >>> graph.graph['sequence']
    'ACGUUACGUUACGUU'
    >>> # evaluate label attributes
    >>> ''.join([ x['label'] for x in graph.node.values() ])
    'ACGUUACGUUACGUU'
    >>> # evaluate weight attributes
    >>> [ x["weight"] for x in graph.node.values() ]
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    """

    def __init__(self, weight_viewpoint=1.0, weight_context=0.0):
        """Construct.

        Parameters
        ----------
        weight_viewpoint : float (default: 1.0)
            Weight assigned to the nodes with capitalized label.

        weight_context : float (default: 0.0)
            Weight assigned to the nodes with lowercase label.
        """
        self.weight_viewpoint = weight_viewpoint
        self.weight_context = weight_context

    def transform(self, graphs):
        """Transform.

        Parameters
        ----------
        graphs : iterator over path graphs
        """
        try:
            for graph in graphs:
                # normalize sequence attribute
                graph.graph['sequence'] = graph.graph['sequence'].upper().replace('T', 'U')
                # iterate over nodes
                for n in graph.nodes():
                    # set weights
                    if graph.node[n]['label'][0].isupper():
                        graph.node[n]['weight'] = self.weight_viewpoint
                    else:
                        graph.node[n]['weight'] = self.weight_context
                    # normalize labels
                    graph.node[n]['label'] = graph.node[n]['label'].upper().replace('T', 'U')
                yield graph
            pass
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)
