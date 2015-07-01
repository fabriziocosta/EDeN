#!/usr/bin/env python

import networkx as nx
import subprocess as sp
from eden.converter.fasta import seq_to_networkx
import math
from eden.util import is_iterable


def rnashapes_wrapper(sequence, shape_type=None, energy_range=None, max_num=None):
    # command line
    cmd = 'echo "%s" | RNAshapes -t %d -c %d -# %d' % (sequence, shape_type, energy_range, max_num)
    out = sp.check_output(cmd, shell=True)
    text = out.strip().split('\n')
    seq_info = text[0]
    if 'configured to print' in text[-1]:
        struct_text = text[1:-1]
    else:
        struct_text = text[1:]
    # shape:
    shape_list = []
    # extract the shape bracket notation
    shape_list += [line.split()[2] for line in struct_text]
    # energy:
    energy_list = []
    # convert negative energy into positive integer
    # create a string of length equal to the energy
    for line in struct_text:
        energy = int(math.floor(-float(line.split()[0])))
        energy_dec = int(energy / 10)
        reminder = energy % 10
        energy_string = ""
        for i in range(energy_dec + 1):
            energy_string += str(i)
        if reminder > 5:
            energy_string += 'x'
        energy_list += [energy_string]
    # dotbracket:
    dotbracket_list = []
    # extract the dot bracket notation
    dotbracket_list += [line.split()[1] for line in struct_text]
    # make a list of triplets
    assert(len(shape_list) == len(energy_list) == len(dotbracket_list)), 'ERROR: unequal length in lists'
    seq_struct_list = []
    for shape, energy, dotbracket in zip(shape_list, energy_list, dotbracket_list):
        assert(len(shape) > 0), 'ERROR: null shape'
        assert(len(energy) > 0), 'ERROR: null energy'
        assert(len(dotbracket) > 0), 'ERROR: null dotbracket'
        seq_struct_list.append((shape, energy, dotbracket))
    return seq_info, seq_struct_list


def string_to_networkx(header, sequence, **options):
    # defaults
    shape_type = options.get('shape_type', 5)
    energy_range = options.get('energy_range', 10)
    max_num = options.get('max_num', 3)
    shape = options.get('shape', False)
    energy = options.get('energy', False)
    dotbracket = options.get('dotbracket', True)
    split_components = options.get('split_components', False)
    seq_info, seq_struct_list = rnashapes_wrapper(sequence, shape_type=shape_type, energy_range=energy_range, max_num=max_num)
    if split_components:
        for shape_str, energy_str, dotbracket_str in seq_struct_list:
            graph = nx.Graph()
            if shape:
                graph_shape = seq_to_networkx('', shape_str)
                graph = nx.disjoint_union(graph, graph_shape)
            if energy:
                graph_energy = seq_to_networkx('', energy_str)
                graph = nx.disjoint_union(graph, graph_energy)
            if dotbracket:
                graph_dotbracket = seq_to_networkx('', dotbracket_str)
                graph = nx.disjoint_union(graph, graph_dotbracket)
            graph.graph['id'] = header + '_' + shape_str
            graph.graph['info'] = 'RNAshapes shape_type=%s energy_range=%s max_num=%s shape=%s energy=%s dotbracket=%s' % (
                shape_type, energy_range, max_num, shape, energy, dotbracket)
            graph.graph['sequence'] = sequence
            yield graph
    else:
        graph_global = nx.Graph()
        for shape_str, energy_str, dotbracket_str in seq_struct_list:
            graph = nx.Graph()
            if shape:
                graph_shape = seq_to_networkx('', shape_str)
                graph = nx.disjoint_union(graph, graph_shape)
            if energy:
                graph_energy = seq_to_networkx('', energy_str)
                graph = nx.disjoint_union(graph, graph_energy)
            if dotbracket:
                graph_dotbracket = seq_to_networkx('', dotbracket_str)
                graph = nx.disjoint_union(graph, graph_dotbracket)
            graph_global = nx.disjoint_union(graph_global, graph)
        graph_global.graph['id'] = header
        graph_global.graph['info'] = 'RNAshapes shape_type=%s energy_range=%s max_num=%s shape=%s energy=%s dotbracket=%s' % (
            shape_type, energy_range, max_num, shape, energy, dotbracket)
        graph_global.graph['sequence'] = sequence
        yield graph_global


def rnashapes_struct_to_eden(iterable, **options):
    assert(is_iterable(iterable)), 'Not iterable'
    for header, seq in iterable:
        try:
            for G in string_to_networkx(header, seq, **options):
                yield G
        except Exception as e:
            print e.__doc__
            print e.message
            print 'Error in: %s %s' % (header, seq)
            graph = seq_to_networkx(header, seq, **options)
            yield graph
