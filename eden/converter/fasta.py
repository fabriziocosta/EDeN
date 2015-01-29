import networkx as nx
from eden.modifier.fasta import fasta_to_fasta


def seq_to_networkx(line, **options):
    G = nx.Graph()
    for id,character in enumerate(line):
        G.add_node(id, label = character, position = id)
        if id > 0:
            G.add_edge(id-1, id, label = '-')
    assert(len(G)>0),'ERROR: generated empty graph. Perhaps wrong format?'
    return G


def fasta_to_eden(input, **options):
    lines = fasta_to_fasta(input)
    for line in lines:
        header = line
        seq = lines.next()
        G = seq_to_networkx(seq, **options)
        G.graph['ID'] = header
        yield G
