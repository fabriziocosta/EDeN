#!/usr/bin/env python

import networkx as nx
import subprocess as sp
from eden.modifier.fasta import fasta_to_fasta
from eden.converter.fasta import seq_to_networkx
from eden.converter.rna import sequence_dotbracket_to_graph
from eden.util import is_iterable


def RNAfold_wrapper(sequence, **options):
    # defaults
    flags = options.get('flags', '--noPS')
    # command line
    cmd = 'echo "%s" | RNAfold %s' % (sequence, flags)
    out = sp.check_output(cmd, shell=True)
    text = out.strip().split('\n')
    seq_info = text[0]
    seq_struct = text[1].split()[0]
    return seq_info, seq_struct


def string_to_networkx(header, sequence, **options):
    seq_info, seq_struct = RNAfold_wrapper(sequence, **options)
    G = sequence_dotbracket_to_graph(seq_info=seq_info, seq_struct=seq_struct)
    G.graph['info'] = 'RNAfold'
    G.graph['sequence'] = sequence
    G.graph['structure'] = seq_struct
    G.graph['id'] = header
    return G


def rnafold_to_eden(iterable=None, **options):
    assert(is_iterable(iterable)), 'Not iterable'
    for header, seq in iterable:
        try:
            G = string_to_networkx(header, seq, **options)
        except Exception as e:
            print e.__doc__
            print e.message
            print 'Error in: %s' % seq
            G = seq_to_networkx(header, seq, **options)
        yield G
