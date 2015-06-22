import networkx as nx
from eden.modifier.fasta import fasta_to_fasta
from eden.util import is_iterable


def seq_to_networkx(header, seq, **options):
    """Convert sequence tuples to networkx graphs."""
    G = nx.Graph()
    G.graph['id'] = header
    for id, character in enumerate(seq):
        G.add_node(id, label=character, position=id)
        if id > 0:
            G.add_edge(id - 1, id, label='-')
    assert(len(G) > 0), 'ERROR: generated empty graph. Perhaps wrong format?'
    G.graph['sequence'] = seq
    return G


def sequence_to_eden(iterable, **options):
    """Convert sequence tuples to EDeN graphs."""
    assert(is_iterable(iterable)), 'Not iterable'
    for header, seq in iterable:
        graph = seq_to_networkx(header, seq, **options)
        yield graph


def fasta_to_sequence(input, **options):
    """Load sequences tuples from fasta file."""
    lines = fasta_to_fasta(input, **options)
    for line in lines:
        header = line
        seq = lines.next()
        yield header, seq
