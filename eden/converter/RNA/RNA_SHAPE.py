import networkx as nx
import subprocess as sp
from eden.modifier.FASTA import FASTA


def RNAshapes_wrapper(sequence, **options):
    #defaults
    path_to_program =  options.get('path_to_program','RNAshapes')
    shape_type =  options.get('shape_type',5)
    energy_range =  options.get('energy_range',10)
    max_num =  options.get('max_num',3)
    #command line
    cmd = 'echo "%s" | %s -t %d -c %d -# %d' % (sequence,path_to_program,shape_type,energy_range,max_num)
    out = sp.check_output(cmd, shell = True)
    text = out.strip().split('\n')
    seq_info = text[0]
    seq_struct_list = [line.split()[1] for line in text[1:-1]] 
    return seq_info, seq_struct_list


def string_to_networkx(sequence, **options):
    seq_info, seq_struct_list = RNAshapes_wrapper(sequence, **options)
    G_global = nx.Graph()
    for seq_struct in seq_struct_list:
        G = nx.Graph()
        lifo = list()
        i=0;
        for c,b in zip(seq_info, seq_struct):
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
            i+=1
        G_global = nx.disjoint_union(G_global, G)
    return G_global


def RNA_SHAPE_to_eden(input = None, input_type = None, **options):
    lines = FASTA.FASTA_to_FASTA(input = input, input_type = input_type)
    for line in lines:
        header = line
        seq = lines.next()
        G = string_to_networkx(seq, **options)
        G.graph['ID'] = header
        yield G