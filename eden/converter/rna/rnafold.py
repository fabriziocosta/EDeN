#!/usr/bin/env python

import networkx as nx
import subprocess as sp
from eden.modifier.fasta import fasta_to_fasta
from eden.converter.fasta import seq_to_networkx


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
    seq_info, seq_struct = RNAfold_wrapper(sequence, **options)
    G = nx.Graph()
    lifo = list()

    for i,c,b in enumerate( zip(seq_info, seq_struct) ):
        G.add_node(i)
        G.node[i]['label'] = c
        G.node[i]['position'] = i
        if i > 0:
            G.add_edge(i,i-1)
            G.edge[i][i-1]['label'] = '-'
            G.edge[i][i-1]['type'] = 'backbone'
        if b == '(':
            lifo += [i]
        if b == ')':
            j = lifo.pop()
            G.add_edge(i,j)
            G.edge[i][j]['label'] = '='
            G.edge[i][j]['type'] = 'basepair'

    return G


def rnafold_to_eden(input = None, **options):
    lines =  fasta_to_fasta(input = input)
    for line in lines:
        header = line
        seq = lines.next()
        G = string_to_networkx(seq, **options)
        #in case something goes wrong fall back to simple sequence
        if G.number_of_nodes() < 2 :
            G = seq_to_networkx(seq, **options)
        G.graph['ID'] = header
        yield G
