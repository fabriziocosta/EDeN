from eden import util
from sklearn.base import BaseEstimator, TransformerMixin
import networkx as nx

import logging
logger = logging.getLogger(__name__)


def seq_to_networkx(header, seq, constr=None):
    """Convert sequence tuples to networkx graphs."""
    graph = nx.Graph()
    graph.graph['header'] = header
    for id, character in enumerate(seq):
        graph.add_node(id, label=character, position=id)
        if id > 0:
            graph.add_edge(id - 1, id, label='-')
    assert(len(graph) > 0), 'ERROR: generated empty graph. Perhaps wrong format?'
    graph.graph['sequence'] = seq
    if constr is not None:
        graph.graph['constraint'] = constr
    return graph


# ---------------------------------------------------------------------------------------

class SeqToPathGraph(BaseEstimator, TransformerMixin):

    """
    Transform seq lists into path graphs.

    """

    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, seqs):
        try:
            for header, seq in seqs:
                yield seq_to_networkx(header, seq)
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)


# ---------------------------------------------------------------------------------------

class SeqWithConstraintsToPathGraph(BaseEstimator, TransformerMixin):

    """
    Transform seq lists into path graphs.
    The seq format is a triplet of header, sequence and constraint.

    """

    def __init__(self):
        pass

    def fit(self):
        return self

    def transform(self, seqs):
        try:
            for header, seq, constr in seqs:
                yield seq_to_networkx(header, seq, constr=constr)
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)


# ---------------------------------------------------------------------------------------

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
                yield seq_to_networkx(header, seq)
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

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

# ---------------------------------------------------------------------------------------


class FastaWithConstraintsToPathGraph(BaseEstimator, TransformerMixin):

    """
    Transform FASTA files with constraints into path graphs.

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
            for header, seq, constr in seqs:
                yield seq_to_networkx(header, seq, constr=constr)
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def fasta_to_fasta(self, data):
        iterable = self._fasta_to_fasta(data)
        for line in iterable:
            header = line
            seq = iterable.next()
            constr = iterable.next()
            if self.normalize:
                seq = seq.upper()
                seq = seq.replace('T', 'U')
            yield header, seq, constr

    def _fasta_to_fasta(self, data):
        header = ""
        seq = ""
        constr = ""
        for line in util.read(data):
            line = str(line).strip()
            if line == "":
                # assume the empty line indicates that next line describes the constraints
                if seq:
                    yield seq
                seq = None
            elif line[0] == '>':
                if constr:
                    yield constr
                    header = ""
                    seq = ""
                    constr = ""
                header = line
                yield header
            else:
                # remove trailing chars, split and take only first part, removing comments
                line_str = line.split()[0]
                if line_str:
                    if seq is None:
                        constr += line_str
                    else:
                        seq += line_str
        if constr:
            yield constr
