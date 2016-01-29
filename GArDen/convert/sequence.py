from eden import util
from sklearn.base import BaseEstimator, TransformerMixin
import networkx as nx

import logging
logger = logging.getLogger(__name__)


class FastaToPathGraph(BaseEstimator, TransformerMixin):

    """
    Transform FASTA files into path graphs.

    normalize : bool
        If True all characters are uppercased and Ts are replaced by Us.
    """

    def __init__(self, normalize=True):
        self.normalize = normalize

    def fit(self):
        return self

    def transform(self, data):
        '''
        Parameters
        ----------
        data : filename or url or iterable
            Data source containing sequences information in FASTA format.


        Returns
        -------
        Iterator over networkx graphs.
        '''
        try:
            seqs = self.fasta_to_fasta(data)
            for header, seq in seqs:
                yield self.seq_to_networkx(header, seq)
        except Exception as e:
            print e.__doc__
            print e.message

    def seq_to_networkx(self, header, seq):
        """Convert sequence tuples to networkx graphs."""
        graph = nx.Graph()
        graph.graph['header'] = header
        for id, character in enumerate(seq):
            graph.add_node(id, label=character, position=id)
            if id > 0:
                graph.add_edge(id - 1, id, label='-')
        assert(len(graph) > 0), 'ERROR: generated empty graph. Perhaps wrong format?'
        graph.graph['sequence'] = seq
        return graph

    def fasta_to_fasta(self, data):
        iterable = self._fasta_to_fasta(data)
        for line in iterable:
            header = line
            seq = iterable.next()
            if self.normalize:
                seq = seq.upper()
                seq = seq.replace('T', 'U')
            yield header, seq

    def _fasta_to_fasta(self, data):
        seq = ""
        for line in util.read(data):
            if line:
                if line[0] == '>':
                    line = line[1:]
                    if seq:
                        yield seq
                        seq = ""
                    line_str = str(line)
                    yield line_str.strip()
                else:
                    line_str = line.split()
                    if line_str:
                        seq += str(line_str[0]).strip()
        if seq:
            yield seq
