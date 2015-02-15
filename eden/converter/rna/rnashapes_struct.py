#!/usr/bin/env python

import networkx as nx
import subprocess as sp
from eden.modifier.fasta import fasta_to_fasta
from eden.converter.fasta import seq_to_networkx
import math


def rnashapes_wrapper(sequence, shape_type=None, energy_range=None, max_num=None, shape=None, energy=None, dotbracket=None):
    #command line
    cmd = 'echo "%s" | RNAshapes -t %d -c %d -# %d' % (sequence,shape_type,energy_range,max_num)
    out = sp.check_output(cmd, shell = True)
    text = out.strip().split('\n')
    seq_info = text[0]
    seq_struct_list = []
    if shape:
        #extract the shape bracket notation
        seq_struct_list += [line.split()[2] for line in text[1:-1]] 
    if energy:
        #convert negative energy into positive integer
        #create a string of length equal to the energy
        for line in text[1:-1]:
            energy = int(math.floor(-float(line.split()[0])))
            energy_dec = int( energy / 10 )
            reminder = energy % 10
            energy_string = ""
            for i in range(energy_dec):
                energy_string += str(i)
            if reminder > 5:
                energy_string += 'x'
            seq_struct_list += [energy_string] 
    if dotbracket:
        #extract the dot bracket notation
        seq_struct_list += [line.split()[1] for line in text[1:-1]] 
    return seq_info, seq_struct_list


def string_to_networkx(sequence, **options):
    #defaults
    shape_type =  options.get('shape_type',5)
    energy_range =  options.get('energy_range',10)
    max_num =  options.get('max_num',3)
    shape = options.get('shape',False)
    energy = options.get('energy',False)
    dotbracket = options.get('dotbracket',True)
    seq_info, seq_struct_list = rnashapes_wrapper(sequence, 
        shape_type=shape_type, 
        energy_range=energy_range, 
        max_num=max_num,
        shape=shape,
        energy=energy,
        dotbracket=dotbracket)
    G_global = nx.Graph()
    G_global.graph['info'] = 'RNAshapes shape_type=%s energy_range=%s max_num=%s shape=%s energy=%s dotbracket=%s' % (shape_type, energy_range, max_num, shape, energy, dotbracket)
    for seq_struct in seq_struct_list:
        G = nx.Graph()
        lifo = list()
        for i,b in enumerate( seq_struct ):
            G.add_node(i)
            G.node[i]['label'] = b
            G.node[i]['position'] = i
            if i > 0:
                G.add_edge(i,i-1)
                G.edge[i][i-1]['label'] = '-'
        G_global = nx.disjoint_union(G_global, G)
    return G_global


def rnashapes_struct_to_eden(input, **options):
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
