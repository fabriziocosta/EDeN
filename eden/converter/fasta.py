import networkx as nx
from eden.modifier.fasta import fasta_to_fasta
from eden.util import is_iterable


def seq_to_networkx(header, seq, **options):
    """Convert sequence tuples to networkx graphs."""

    graph = nx.Graph()
    graph.graph['id'] = header
    for id, character in enumerate(seq):
        graph.add_node(id, label=character, position=id)
        if id > 0:
            graph.add_edge(id - 1, id, label='-')
    assert(len(graph) > 0), 'ERROR: generated empty graph. Perhaps wrong format?'
    graph.graph['sequence'] = seq
    return graph


def sequence_to_eden(iterable, **options):
    """Convert sequence tuples to EDeN graphs."""

    no_header = options.get('no_header', False)
    assert(is_iterable(iterable)), 'Not iterable'
    if no_header is True:
        for seq in iterable:
            graph = seq_to_networkx('NONE', seq, **options)
            yield graph
    else:
        for header, seq in iterable:
            graph = seq_to_networkx(header, seq, **options)
            yield graph


def fasta_to_sequence(input, **options):
    """Load sequences tuples from fasta file."""

    lines = fasta_to_fasta(input, **options)
    for line in lines:
        header = line
        seq = lines.next()
        if len(seq) == 0:
            raise Exception('ERROR: empty sequence')
        yield header, seq
