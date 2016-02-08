#!/usr/bin/env python

import networkx as nx
import subprocess as sp
from eden.converter.fasta import seq_to_networkx
from eden.converter.rna import sequence_dotbracket_to_graph
from eden.util import is_iterable

# TODO: use sampling if seq is too long
# use different parsing of output in this case
# check version of program


def rnashapes_wrapper(sequence, shape_type=None, energy_range=None, max_num=None, rnashapes_version=None):
    # command line
    if rnashapes_version == 2:
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
    elif rnashapes_version == 3:
        cmd = 'echo "%s" | RNAshapes --mode shapes --shapeLevel %d --relativeDeviation %d' % (sequence, shape_type, energy_range)
        out = sp.check_output(cmd, shell=True)
        # parse output
        text = out.strip().split('\n')
        # remove additional fasta-like sequence id found in RNAshapes version 3+
        text.pop(0)
        seq_info = text.pop(0).split()[1]
        struct_text = text[:max_num + 1]
        seq_struct_list = [line.split()[1] for line in struct_text]
        struct_list = [line.split()[2] for line in struct_text]
        return seq_info, seq_struct_list, struct_list
    else:
        raise Exception("Error, parsing of RNAshapes version level '{}' not implemented.".format(rnashapes_version))


def string_to_networkx(header, sequence, **options):
    # defaults
    shape_type = options.get('shape_type', 5)
    energy_range = options.get('energy_range', 10)
    max_num = options.get('max_num', 3)
    split_components = options.get('split_components', False)
    seq_info, seq_struct_list, struct_list = rnashapes_wrapper(sequence,
                                                               shape_type=shape_type,
                                                               energy_range=energy_range,
                                                               max_num=max_num,
                                                               rnashapes_version=options.get('rnashapes_version', 2))
    if split_components:
        for seq_struct, struct in zip(seq_struct_list, struct_list):
            graph = sequence_dotbracket_to_graph(seq_info=seq_info, seq_struct=seq_struct)
            graph.graph['info'] = 'RNAshapes shape_type=%s energy_range=%s max_num=%s' % (shape_type,
                                                                                          energy_range,
                                                                                          max_num)
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
        graph_global.graph['info'] = 'RNAshapes shape_type=%s energy_range=%s max_num=%s' % (shape_type,
                                                                                             energy_range,
                                                                                             max_num)
        graph_global.graph['sequence'] = sequence
        for seq_struct in seq_struct_list:
            graph = sequence_dotbracket_to_graph(seq_info=seq_info, seq_struct=seq_struct)
            graph_global = nx.disjoint_union(graph_global, graph)
        if graph_global.number_of_nodes() < 2:
            graph_global = seq_to_networkx(header, sequence, **options)
        yield graph_global


def rnashapes_to_eden(iterable, **options):
    """Transforms sequences to graphs that encode secondary structure information
    according to the RNAShapes algorithm.

    Parameters
    ----------
    sequences : iterable
        iterable pairs of header and sequence strings

    rnashapes_version : int (default 2)
        The version of RNAshapes that is in the path.
        2   e.g. RNAshapes version 2.1.6
        3   e.g. RNAshapes version 3.3.0

    shape_type : int (default 5)
        Is the level of abstraction or dissimilarity which defines a different shape.
        In general, helical regions are depicted by a pair of opening and closing brackets
        and unpaired regions are represented as a single underscore. The differences of the
        shape types are due to whether a structural element (bulge loop, internal loop, multiloop,
        hairpin loop, stacking region and external loop) contributes to the shape representation:
        Five types are implemented.
        1   Most accurate - all loops and all unpaired  [_[_[]]_[_[]_]]_
        2   Nesting pattern for all loop types and unpaired regions in external loop
        and multiloop [[_[]][_[]_]]
        3   Nesting pattern for all loop types but no unpaired regions [[[]][[]]]
        4   Helix nesting pattern in external loop and multiloop [[][[]]]
        5   Most abstract - helix nesting pattern and no unpaired regions [[][]]

    energy_range : float (default 10)
        Sets the energy range as percentage value of the minimum free energy.
        For example, when relative deviation is specified as 5.0, and the minimum free energy
        is -10.0 kcal/mol, the energy range is set to -9.5 to -10.0 kcal/mol.
        Relative deviation must be a positive floating point number; by default it is set to to 10 %.

    max_num : int (default 3)
        Is the maximum number of structures that are generated.

    split_components : bool (default False)
        If True each structure is yielded as an independent graph. Otherwise all structures
        are part of the same graph that has therefore several disconnectd components.

    example: transform a simple sequence using RNAshapes version 3+
        >>> graphs = rnashapes_to_eden([("ID", "CCCCCGGGGG")], rnashapes_version=3)
        >>> g = graphs.next()
        >>> # extract sequence from graph nodes
        >>> "".join([ value["label"] for (key, value) in g.nodes(data=True)])
        'CCCCCGGGGG'
        >>> # get vertice types
        >>> [(start, end, g.edge[start][end]["type"]) for start, end in g.edges()]
        [(0, 8, 'basepair'), (0, 1, 'backbone'), (1, 2, 'backbone'), (1, 7, 'basepair'), (2, 3, 'backbone'), (2, 6, 'basepair'), (3, 4, 'backbone'), (4, 5, 'backbone'), (5, 6, 'backbone'), (6, 7, 'backbone'), (7, 8, 'backbone'), (8, 9, 'backbone')]

    example: transform a simple sequence using RNAshapes version 3+, splitting components
        >>> graphs = rnashapes_to_eden([("ID", "CCCCCGGGGG")], split_components=True, rnashapes_version=3)
        >>> g = graphs.next()
        >>> # extract sequence from graph nodes
        >>> "".join([ value["label"] for (key, value) in g.nodes(data=True)])
        'CCCCCGGGGG'
        >>> # get dotbracket structure annotation
        >>> g.graph["structure"]
        '(((...))).'
        >>> # get vertice types
        >>> [ (start, end, g.edge[start][end]["type"]) for start, end in g.edges()]
        [(0, 8, 'basepair'), (0, 1, 'backbone'), (1, 2, 'backbone'), (1, 7, 'basepair'), (2, 3, 'backbone'), (2, 6, 'basepair'), (3, 4, 'backbone'), (4, 5, 'backbone'), (5, 6, 'backbone'), (6, 7, 'backbone'), (7, 8, 'backbone'), (8, 9, 'backbone')]

    test max_num parameter with RNAshapes version 3+
        >>> seq = "CGUCGUCGCAUCGUACGCAUGACUCAGCAUCAGACUACGUACGCAUACGUCAGCAUCAGUCAGCAUCAGCAUGCAUCACUAGCAUGCACCCCCGGGGGCACAUCGUACGUACGCUCAGUACACUGCAUGACUACGU"
        >>> graphs = rnashapes_to_eden([("ID", seq)], split_components=True, max_num=2, rnashapes_version=3)
        >>> g = graphs.next()
        >>> # get dotbracket structure annotations
        >>> len([g.graph["structure"] for g in graphs])
        2
    """

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
