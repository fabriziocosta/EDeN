#!/usr/bin/env python

import os
import networkx as nx
import subprocess as sp
from eden.modifier.fasta import fasta_to_fasta
from eden.converter.fasta import seq_to_networkx


def RNAplfold_wrapper(sequence, **options):
    # default settings.
    window_size = options.get('window_size', 150)
    max_bp_span = options.get('max_bp_span', 100)
    avg_bp_prob_cutoff = options.get('avg_bp_prob_cutoff', 0.2)
    no_lonely_bps = options.get('no_lonely_bps', True)
    no_lonely_bps_str = ""
    if no_lonely_bps:
       no_lonely_bps_str = "--noLP"
    # Call RNAplfold on command line.
    cmd = 'echo "%s" | RNAplfold -W %d -L %d -c %.2f %s' % (sequence,window_size,max_bp_span,avg_bp_prob_cutoff, no_lonely_bps_str)
    sp.check_output(cmd, shell = True)
    # Extract base pair information.
    start_flag = False
    plfold_bp_list = []
    with open('plfold_dp.ps') as f:
        for line in f:
            if start_flag:
               values = line.split()
               if len(values) == 4:
                   plfold_bp_list += [(values[2],values[0],values[1])]
            if '%start of base pair probability data' in line:
                start_flag = True
    f.closed
    # Delete RNAplfold output file.
    os.remove("plfold_dp.ps")
    # Return list with base pair information.
    return plfold_bp_list


def string_to_networkx(sequence, **options):
    # Sort edges by average base pair probability in order to stop after
    # max_num_edges edges have been added to a specific vertex.
    plfold_bp_list = sorted(RNAplfold_wrapper(sequence, **options), reverse=True)
    max_num_edges =  options.get('max_num_edges', 1)
    # Create graph.
    G = nx.Graph()
    # Add nucleotide vertices.
    for i,c in enumerate(sequence):
        G.add_node(i)
        G.node[i]['label'] = c
        G.node[i]['position'] = i
    # Add plfold base pairs and average probabilites.
    for avg_prob_str,source_str,dest_str in plfold_bp_list:
        source = int(source_str)-1
        dest = int(dest_str)-1
        avg_prob = float(avg_prob_str)
        # Check if either source or dest already have more than max_num_edges edges.
        if len(G.edges(source)) >= max_num_edges or len(G.edges(dest)) >= max_num_edges:
            pass
        else:
            G.add_edge(source,dest,label='=',type='basepair',value=avg_prob)
    # Add backbone edges.
    for i,c in enumerate(sequence):
        if i > 0:
            G.add_edge(i,i-1)
            G.edge[i][i-1]['label'] = '-'
            G.edge[i][i-1]['type'] = 'backbone'
    return G


def rnaplfold_to_eden(input, **options):
    lines =  fasta_to_fasta(input)
    for line in lines:
        header = line
        seq = lines.next()
        G = string_to_networkx(seq, **options)
        #in case something goes wrong fall back to simple sequence
        if G.number_of_nodes() < 2 :
            G = seq_to_networkx(seq, **options)
        G.graph['ID'] = header
        yield G