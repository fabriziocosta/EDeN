#!/usr/bin/env python

import networkx as nx
import subprocess as sp
from eden.converter.fasta import seq_to_networkx
from eden.converter.rna import sequence_dotbracket_to_graph
from eden.util import is_iterable

# TODO: use sampling if seq is too long
# use different parsing of output in this case
# check version of program


def rnashapes_wrapper(sequence, shape_type=None, energy_range=None, max_num=None):
    # command line
    cmd = 'echo "%s" | RNAshapes -t %d -c %d -# %d' % (sequence, shape_type, energy_range, max_num)
    out = sp.check_output(cmd, shell=True)
    # parse output
    text = out.strip().split('\n')
    seq_info = text[0]
    if 'configured to print' in text[-1]:
        struct_text = text[1:-1]
    else:
        struct_text = text[1:]
    seq_struct_list = [line.split()[1] for line in struct_text]
    struct_list = [line.split()[2] for line in struct_text]
    return seq_info, seq_struct_list, struct_list


def string_to_networkx(header, sequence, **options):
    # defaults
    shape_type = options.get('shape_type', 5)
    energy_range = options.get('energy_range', 10)
    max_num = options.get('max_num', 3)
    split_components = options.get('split_components', False)
    seq_info, seq_struct_list, struct_list = rnashapes_wrapper(sequence, shape_type=shape_type, energy_range=energy_range, max_num=max_num)
    if split_components:
        for seq_struct, struct in zip(seq_struct_list, struct_list):
            graph = sequence_dotbracket_to_graph(seq_info=seq_info, seq_struct=seq_struct)
            graph.graph['info'] = 'RNAshapes shape_type=%s energy_range=%s max_num=%s' % (shape_type, energy_range, max_num)
            graph.graph['id'] = header + '_' + struct
            if graph.number_of_nodes() < 2:
                graph = seq_to_networkx(header, sequence, **options)
                graph.graph['id'] = header
            graph.graph['sequence'] = sequence
            graph.graph['structure'] = seq_struct
            yield graph
    else:
        graph_global = nx.Graph()
        graph_global.graph['id'] = header
        graph_global.graph['info'] = 'RNAshapes shape_type=%s energy_range=%s max_num=%s' % (shape_type, energy_range, max_num)
        graph_global.graph['sequence'] = sequence
        for seq_struct in seq_struct_list:
            graph = sequence_dotbracket_to_graph(seq_info=seq_info, seq_struct=seq_struct)
            graph_global = nx.disjoint_union(graph_global, graph)
        if graph_global.number_of_nodes() < 2:
            graph_global = seq_to_networkx(header, sequence, **options)
        yield graph_global


def rnashapes_to_eden(iterable, **options):
    assert(is_iterable(iterable)), 'Not iterable'
    for header, seq in iterable:
        try:
            for graph in string_to_networkx(header, seq, **options):
                yield graph
        except Exception as e:
            print e.__doc__
            print e.message
            print 'Error in: %s' % seq
            graph = seq_to_networkx(header, seq, **options)
            yield graph
