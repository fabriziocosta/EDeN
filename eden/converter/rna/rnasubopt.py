#!/usr/bin/env python

import networkx as nx
import subprocess as sp
import numpy as np
from eden.modifier.fasta import fasta_to_fasta
from eden.converter.fasta import seq_to_networkx
from eden.converter.rna import sequence_dotbracket_to_graph
from eden.util import read
from eden.util import is_iterable


def difference(seq_a, seq_b):
    ''' Computes the number of characters that are different between the two sequences
    '''
    return sum(1 if a != b else 0 for a, b in zip(seq_a, seq_b))


def difference_matrix(seqs):
    ''' Computes the matrix of differences between all pairs of sequences in input
    '''
    size = len(seqs)
    D = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            D[i, j] = difference(seqs[i], seqs[j])
    return D + D.T


def max_difference_subselection(seqs, scores=None, max_num=None):
    # extract difference matrix
    D = difference_matrix(seqs)
    size = len(seqs)
    m = np.max(D) + 1
    # iterate size - k times, i.e. until only k instances are left
    for t in range(size - max_num):
        # find pairs with smallest difference
        (min_i, min_j) = np.unravel_index(np.argmin(D), D.shape)
        # choose instance with highest score
        if scores[min_i] > scores[min_j]:
            id = min_i
        else:
            id = min_j
        # remove instance with highest score by setting all its pairwise differences to max value
        D[id, :] = m
        D[:, id] = m
    # extract surviving elements, i.e. element that have 0 on the diagonal
    return np.array([i for i, x in enumerate(np.diag(D)) if x == 0])


def rnasubopt_wrapper(sequence, energy_range=None, max_num=None, max_num_subopts=None):
    # command line
    cmd = 'echo "%s" | RNAsubopt -e %d' % (sequence, energy_range)
    out = sp.check_output(cmd, shell=True)
    # parse output
    text = out.strip().split('\n')
    seq_struct_list = [line.split()[0] for line in text[1:max_num_subopts]]
    energy_list = [line.split()[1] for line in text[1:max_num_subopts]]
    selected_ids = max_difference_subselection(seq_struct_list, scores=energy_list, max_num=max_num)
    np_seq_struct_list = np.array(seq_struct_list)
    selected_seq_struct_list = list(np_seq_struct_list[selected_ids])
    selected_energy_list = list(np.array(energy_list)[selected_ids])
    return selected_seq_struct_list, selected_energy_list


def string_to_networkx(header, sequence, **options):
    # defaults
    energy_range = options.get('energy_range', 10)
    max_num = options.get('max_num', 3)
    max_num_subopts = options.get('max_num_subopts', 100)
    split_components = options.get('split_components', False)
    seq_struct_list, energy_list = rnasubopt_wrapper(sequence, energy_range=energy_range, max_num=max_num, max_num_subopts=max_num_subopts)
    if split_components:
        for seq_struct, energy in zip(seq_struct_list, energy_list):
            G = sequence_dotbracket_to_graph(seq_info=sequence, seq_struct=seq_struct)
            G.graph['info'] = 'RNAsubopt energy=%s max_num=%s' % (energy, max_num)
            if G.number_of_nodes() < 2:
                G = seq_to_networkx(header, sequence, **options)
            G.graph['id'] = header 
            G.graph['sequence'] = sequence
            yield G
    else:
        G_global = nx.Graph()
        G_global.graph['id'] = header
        G_global.graph['info'] = 'RNAsubopt energy_range=%s max_num=%s' % (energy_range, max_num)
        G_global.graph['sequence'] = sequence
        for seq_struct in seq_struct_list:
            G = sequence_dotbracket_to_graph(seq_info=sequence, seq_struct=seq_struct)
            G_global = nx.disjoint_union(G_global, G)
        if G_global.number_of_nodes() < 2:
            G_global = seq_to_networkx(header, sequence, **options)
        yield G_global


def rnasubopt_to_eden(iterable, **options):
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
