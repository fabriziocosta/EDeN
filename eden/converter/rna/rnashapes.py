#!/usr/bin/env python

import networkx as nx
import subprocess as sp
from eden.modifier.fasta import fasta_to_fasta
from eden.converter.fasta import seq_to_networkx
from eden.converter.rna import sequence_dotbracket_to_graph
from eden.util import read
from eden.util import is_iterable


def rnashapes_wrapper(sequence, shape_type=None, energy_range=None, max_num=None):
    #command line
    cmd = 'echo "%s" | RNAshapes -t %d -c %d -# %d' % (sequence,shape_type,energy_range,max_num)
    out = sp.check_output(cmd, shell = True)
    #parse output
    text = out.strip().split('\n')
    seq_info = text[0]
    seq_struct_list = [line.split()[1] for line in text[1:-1]] 
    struct_list = [line.split()[2] for line in text[1:-1]] 
    return seq_info, seq_struct_list, struct_list


def string_to_networkx(header, sequence, **options):
    #defaults
    shape_type =  options.get('shape_type',5)
    energy_range =  options.get('energy_range',10)
    max_num =  options.get('max_num',3)
    split_components = options.get('split_components', False)
    seq_info, seq_struct_list, struct_list = rnashapes_wrapper(sequence, shape_type=shape_type, energy_range=energy_range, max_num=max_num)
    if split_components:
        for seq_struct, struct in zip(seq_struct_list, struct_list):
            G = sequence_dotbracket_to_graph(seq_info=seq_info, seq_struct=seq_struct)
            G.graph['info'] = 'RNAshapes shape_type=%s energy_range=%s max_num=%s' % (shape_type, energy_range, max_num)
            G.graph['id'] = header + '_' + struct
            if G.number_of_nodes() < 2 :
                G = seq_to_networkx(header,sequence, **options)
                G.graph['id'] = header
            G.graph['sequence'] = sequence
            yield G
    else:
        G_global = nx.Graph()   
        G_global.graph['id'] = header
        G_global.graph['info'] = 'RNAshapes shape_type=%s energy_range=%s max_num=%s' % (shape_type, energy_range, max_num)
        for seq_struct in seq_struct_list:
            G = sequence_dotbracket_to_graph(seq_info=seq_info, seq_struct=seq_struct)
            G_global = nx.disjoint_union(G_global, G)
        if G_global.number_of_nodes() < 2 :
            G_global = seq_to_networkx(header,sequence, **options)
        yield G_global


def rnashapes_to_eden(iterable, **options):
    assert(is_iterable(iterable)), 'Not iterable'
    for header, seq in iterable:
        try:
            for G in string_to_networkx(header, seq, **options):
                yield G
        except Exception as e:
            print e.__doc__
            print e.message
            print 'Error in: %s' % seq
            G = seq_to_networkx(header, seq, **options)
        yield G
