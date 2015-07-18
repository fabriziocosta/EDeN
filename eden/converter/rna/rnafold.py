#!/usr/bin/env python

import subprocess as sp
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
    graph = sequence_dotbracket_to_graph(seq_info=seq_info, seq_struct=seq_struct)
    graph.graph['info'] = 'RNAfold'
    graph.graph['sequence'] = sequence
    graph.graph['structure'] = seq_struct
    graph.graph['id'] = header
    return graph


def rnafold_to_eden(iterable=None, **options):
    assert(is_iterable(iterable)), 'Not iterable'
    for header, seq in iterable:
        try:
            graph = string_to_networkx(header, seq, **options)
        except Exception as e:
            print e.__doc__
            print e.message
            print 'Error in: %s' % seq
            graph = seq_to_networkx(header, seq, **options)
        yield graph
