#!/usr/bin/env python

import networkx as nx
import subprocess as sp
from eden.modifier.fasta import fasta_to_fasta
from eden.converter.fasta import seq_to_networkx
from eden.converter.rna import sequence_dotbracket_to_graph


def RNAfold_wrapper(sequence, **options):
    #defaults
    flags =  options.get('flags','--noPS')
    #command line
    cmd = 'echo "%s" | RNAfold %s' % (sequence,flags)
    out = sp.check_output(cmd, shell = True)
    text = out.strip().split('\n')
    seq_info = text[0]
    seq_struct = text[1].split()[0]
    return seq_info, seq_struct


def string_to_networkx(sequence, **options):
    try:
        seq_info, seq_struct = RNAfold_wrapper(sequence, **options)
    except:
        print 'Error in: %s' % sequence
    else:
        G = sequence_dotbracket_to_graph(seq_info=seq_info, seq_struct=seq_struct)
        G.graph['info'] = 'RNAfold'
        G.graph['sequence'] = sequence
        return G


def rnafold_to_eden(input = None, **options):
    data_type =  options.get('data_type','fasta')
    if data_type == 'fasta':
        lines =  fasta_to_fasta(input)
        for line in lines:
            header = line
            seq = lines.next()
            G = string_to_networkx(seq, **options)
            #in case something goes wrong fall back to simple sequence
            if G.number_of_nodes() < 2 :
                G = seq_to_networkx(seq, **options)
            G.graph['id'] = header
            yield G
    elif data_type == 'seq':
        lines = read(input)
        for line in lines:
            items = line.split('\t')
            header = items[:-1]
            header = ''.join(header)
            seq = items[-1]
            seq = ''.join(seq)
            G = string_to_networkx(seq, **options)
            #in case something goes wrong fall back to simple sequence
            if G.number_of_nodes() < 2 :
                G = seq_to_networkx(seq, **options)
            G.graph['id'] = header
            yield G
    else:
        raise Exception('Unknown data type: %s' % data_type)