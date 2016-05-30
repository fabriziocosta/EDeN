#!/usr/bin/env python
"""Provides conversion from FASTA."""

from eden import util
from sklearn.base import BaseEstimator, TransformerMixin
import networkx as nx

import logging
logger = logging.getLogger(__name__)


def sequence_dotbracket_to_graph(header=None, seq_info=None, seq_struct=None):
    """Build a graph given a sequence string and a string with dotbracket notation.

    Parameters
    ----------
    seq_info : string
        node labels eg a sequence string

    seq_struct : string
        dotbracket string

    Returns
    -------
    A nx.Graph secondary struct associated with seq_struct
    """
    graph = nx.Graph()
    graph.graph['sequence'] = seq_info
    graph.graph['structure'] = seq_struct
    graph.graph['header'] = header
    graph.graph['id'] = header.split()[0]
    assert(len(seq_info) == len(seq_struct)), 'Len seq:%d is different than\
     len struct:%d' % (len(seq_info), len(seq_struct))
    lifo = list()
    for i, (c, b) in enumerate(zip(seq_info, seq_struct)):
        graph.add_node(i, label=c, position=i)
        if i > 0:
            graph.add_edge(i, i - 1, label='-', type='backbone', len=1)
        if b == '(':
            lifo.append(i)
        if b == ')':
            j = lifo.pop()
            graph.add_edge(i, j, label='=', type='basepair', len=1)
    return graph


def seq_to_networkx(header, seq, constr=None):
    """Convert sequence tuples to networkx graphs."""
    graph = nx.Graph()
    graph.graph['id'] = header.split()[0]
    graph.graph['header'] = header
    for id, character in enumerate(seq):
        graph.add_node(id, label=character, position=id)
        if id > 0:
            graph.add_edge(id - 1, id, label='-')
    assert(len(graph) > 0), 'ERROR: generated empty graph.\
    Perhaps wrong format?'
    graph.graph['sequence'] = seq
    if constr is not None:
        graph.graph['constraint'] = constr
    return graph


# ------------------------------------------------------------------------------

class SeqToPathGraph(BaseEstimator, TransformerMixin):
    """Transform seq lists into path graphs.

    Transform a single id and sequence tuple to path graph:
    >>> id = 'ID0'
    >>> seq = 'IamAniceSEQUENCE'
    >>> graphs = SeqToPathGraph().transform([(id,seq)])
    >>> g = graphs.next()
    >>> ''.join([ x['label'] for x in g.node.values() ])
    'IamAniceSEQUENCE'
    >>> g.graph['id']
    'ID0'

    """

    def transform(self, seqs):
        """transform."""
        try:
            for header, seq in seqs:
                yield seq_to_networkx(header, seq)
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)


# ------------------------------------------------------------------------------

class SeqWithConstraintsToPathGraph(BaseEstimator, TransformerMixin):
    """Transform seq lists into path graphs.

    The seq format is a triplet of header, sequence and constraint.
    """

    def transform(self, seqs):
        """transform."""
        try:
            for header, seq, constr in seqs:
                yield seq_to_networkx(header, seq, constr=constr)
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)


# ------------------------------------------------------------------------------


class FastaToPathGraph(BaseEstimator, TransformerMixin):
    """Transform FASTA files into path graphs.

    Transform fasta using defaults:
    >>> fa_upper_lower = 'test/garden_convert_sequence_FastaToPathGraph_1.fa'
    >>> graphs = FastaToPathGraph().transform(fa_upper_lower)
    >>> g = graphs.next()
    >>> ''.join([ x['label'] for x in g.node.values() ])
    'GUGGCGUACUCACGGCCACCUUAGGACUCCGCGGACUUUAUGCCCACCAAAAAAACGAGCCGUUUCUACGCGUCCUCCGUCGCCUGUGUCGAUAAAGCAA'
    >>> g.graph['id']
    'ID0'

    Transform fasta with normalization enabled:
    >>> graphs = FastaToPathGraph(normalize=True).transform(fa_upper_lower)
    >>> ''.join([ x['label'] for x in graphs.next().node.values() ])
    'GUGGCGUACUCACGGCCACCUUAGGACUCCGCGGACUUUAUGCCCACCAAAAAAACGAGCCGUUUCUACGCGUCCUCCGUCGCCUGUGUCGAUAAAGCAA'

    Transform fasta without normalization:
    >>> graphs = FastaToPathGraph(normalize=False).transform(fa_upper_lower)
    >>> ''.join([ x['label'] for x in graphs.next().node.values() ])
    'gtggcgtactcacggccaCCTTAGGACTCCGCGGACTTTATGCCCACCAAAAAAACGAGCCGTTTCTACGCGTCCTCCGTCGCCTgtgtcgataaagcaa'

    Transform fasta containing 'N' nucleotides:
    >>> fa_n = 'test/garden_convert_sequence_FastaToPathGraph_2.fa'
    >>> graphs = FastaToPathGraph(normalize=False).transform(fa_n)
    >>> g = graphs.next()
    >>> ''.join([ x['label'] for x in g.node.values() ])
    'NtNNcNtactcacNNccaCCTTANNACTCCNCNNACTTTATNCCCACCAAAAAAACNANCCNTTTCTACNCNTCCTCCNTCNCCTNtNtcNataaaNcaa'
    """

    def __init__(self, normalize=True, dna=False):
        """constructor.

        Parameters
        ----------
        normalize : boolean (default: True)
            If set, transform all sequences to uppercase and convert T to U.

        dna : boolean (default: False)
            If set, transform U to T.
        """
        self.normalize = normalize
        self.dna = dna

    def transform(self, data):
        """Transform.

        Parameters
        ----------
        data : filename or url or iterable
            Data source containing sequences information in FASTA format.


        Returns
        -------
        Iterator over networkx graphs.
        """
        fastatoseq = FastaToSeq(normalize=self.normalize, dna=self.dna)
        try:
            seqs = fastatoseq.transform(data)
            for header, seq in seqs:
                yield seq_to_networkx(header, seq)
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)


# ------------------------------------------------------------------------------


class FastaToSeq(BaseEstimator, TransformerMixin):
    """Transform FASTA files into tuples of ids and sequences.

    Transform fasta using defaults:
    >>> fa_upper_lower = 'test/garden_convert_sequence_FastaToSeq_1.fa'
    >>> seqs = FastaToSeq().transform(fa_upper_lower)
    >>> id, seq = seqs.next()
    >>> id
    'ID0'
    >>> seq
    'GUGGCGUACUCACGGCCACCUUAGGACUCCGCGGACUUUAUGCCCACCAAAAAAACGAGCCGUUUCUACGCGUCCUCCGUCGCCUGUGUCGAUAAAGCAA'

    Transform fasta with normalization enabled:
    >>> seqs = FastaToSeq(normalize=True).transform(fa_upper_lower)
    >>> id, seq = seqs.next()
    >>> seq
    'GUGGCGUACUCACGGCCACCUUAGGACUCCGCGGACUUUAUGCCCACCAAAAAAACGAGCCGUUUCUACGCGUCCUCCGUCGCCUGUGUCGAUAAAGCAA'

    Transform fasta without normalization:
    >>> seqs = FastaToSeq(normalize=False).transform(fa_upper_lower)
    >>> id, seq = seqs.next()
    >>> seq
    'gtggcgtactcacggccaCCTTAGGACTCCGCGGACTTTATGCCCACCAAAAAAACGAGCCGTTTCTACGCGTCCTCCGTCGCCTgtgtcgataaagcaa'

    Transform fasta containing 'N' nucleotides:
    >>> fa_n = 'test/garden_convert_sequence_FastaToSeq_2.fa'
    >>> seqs = FastaToSeq(normalize=False).transform(fa_n)
    >>> id, seq = seqs.next()
    >>> seq
    'NtNNcNtactcacNNccaCCTTANNACTCCNCNNACTTTATNCCCACCAAAAAAACNANCCNTTTCTACNCNTCCTCCNTCNCCTNtNtcNataaaNcaa'
    """

    def __init__(self, normalize=True, dna=False):
        """constructor.

        Parameters
        ----------
        normalize : boolean (default: True)
            If set, transform all sequences to uppercase and convert T to U.

        dna : boolean (default: False)
            If set, transform U to T.
        """
        self.normalize = normalize
        self.dna = dna

    def transform(self, data):
        """Transform.

        Parameters
        ----------
        data : filename or url or iterable
            Data source containing sequences information in FASTA format.


        Returns
        -------
        Iterator over tuples of ids and sequences.
        """
        try:
            # seqs = self._fasta_to_seq(data)
            # yield seqs.next()
            # why not return iterator directly?
            return self._fasta_to_seq(data)
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _fasta_to_seq(self, data):
        iterable = self._fasta_to_fasta(data)
        for line in iterable:
            header = line
            seq = iterable.next()
            if self.normalize:
                seq = seq.upper()
                seq = seq.replace('T', 'U')
            if self.dna:
                seq = seq.replace('U', 'T')
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

# ------------------------------------------------------------------------------


class FastaWithConstraintsToPathGraph(BaseEstimator, TransformerMixin):
    """Transform FASTA files with constraints into path graphs."""

    def __init__(self, normalize=True):
        """Constructor."""
        self.normalize = normalize

    def transform(self, data):
        """Transform.

        Parameters
        ----------
        data : filename or url or iterable
            Data source containing sequences information in FASTA format.


        Returns
        -------
        Iterator over networkx graphs.
        """
        try:
            seqs = self._fasta_to_seq(data)
            for header, seq, constr in seqs:
                yield seq_to_networkx(header, seq, constr=constr)
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _fasta_to_seq(self, data):
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
                # assume the empty line indicates that next line describes
                # the constraints
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
                # remove trailing chars, split and take only first part,
                # removing comments
                line_str = line.split()[0]
                if line_str:
                    if seq is None:
                        constr += line_str
                    else:
                        seq += line_str
        if constr:
            yield constr


# ------------------------------------------------------------------------------


class FastaWithStructureToGraph(BaseEstimator, TransformerMixin):
    """Transform FASTA files with structure into graphs."""

    def __init__(self, normalize=True):
        """Constructor."""
        self.normalize = normalize

    def transform(self, data):
        """Transform.

        Parameters
        ----------
        data : filename or url or iterable
            Data source containing sequences information in FASTA format.


        Returns
        -------
        Iterator over networkx graphs.
        """
        try:
            seqs = self._fasta_to_seq(data)
            for header, seq, struct in seqs:
                yield sequence_dotbracket_to_graph(header, seq, struct)
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _fasta_to_seq(self, data):
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
                # assume the empty line indicates that next line describes
                # the constraints
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
                # remove trailing chars, split and take only first part,
                # removing comments
                line_str = line.split()[0]
                if line_str:
                    if seq is None:
                        constr += line_str
                    else:
                        seq += line_str
        if constr:
            yield constr
