import networkx as nx
import eden.modifier.rna.lib_forgi as lib_forgi
from sklearn.base import BaseEstimator, ClassifierMixin


class Annotator(BaseEstimator, ClassifierMixin):

    def fit(self):
        return self

    def transform(self, graphs, part_id='part_id', part_name='part_name'):
        '''

        Parameters
        ----------
        graphs  nx.graph iterator, with graph.graph['structure']
        part_id  attributename to write the part_id
        part_name attributename to write the part_name


        Returns
        -------
        it is important to see that the part_name is not equivalent to the part_id.
        all stems may be called 's', but each stem has its own id.
        '''

        for g in graphs:
            yield self._transform_single(g, part_id=part_id, part_name=part_name)

    def _transform_single(self, graph, part_id='part_id', part_name='part_name'):
        '''
        you have generated graphs  like this and want a forgi annotation now:

        from graphlearn.utils import draw
        seqs = get_sequences()
        from eden.converter.rna.rnafold import  rnafold_to_eden as rnafold
        seq=seqs.next()
        g=rnafold([seq]).next()
        draw.graphlearn(g)
        '''
        assert ('structure' in graph.graph), "i need a 'structure' -- a VALID dotbracket string!"
        struct = graph.graph['structure']

        abstrg = get_abstr_graph(struct, ignore_inserts=False)
        for n, d in abstrg.nodes(data=True):
            for nodeid in d['contracted']:
                graph.node[nodeid][part_name] = [d['label']]
                graph.node[nodeid][part_id] = [n]

        return graph


def get_abstr_graph(struct, ignore_inserts=False):
    '''
    Parameters
    ----------
    struct: basestring
        dot-bracket string
    ignore_inserts: bool
        internal loops are ignored

    Returns
    -------
        abstract graph   with "label" and "contracton" for each node.
        graph is not expanded
    '''

    bg = lib_forgi.BulgeGraph()
    bg.from_dotbracket(struct, None)
    forgi = bg.to_bg_string()
    g = make_abstract_graph(forgi, ignore_inserts)
    return g


def make_abstract_graph(forgi, ignore_inserts=False):
    '''

    Parameters
    ----------
    forgi: string
        output of forgiobject,to_bg_string()
    ignore_inserts : bool
        ignore internal loops

    Returns
    -------
        nx.graph
    '''
    g = forgi_to_graph(forgi, ignore_inserts)
    connect_multiloop(g)
    return g


def forgi_to_graph(forgi, ignore_inserts=False):
    '''

    Parameters
    ----------
    forgi: forgi string
    ignore_inserts: iignore internal loops

    Returns
    -------
        nx.graph
    '''

    def make_node_set(numbers):
        '''
        numbers: list of string
            forgi gives me stuff like """define STEM START,END,START,END"""

        Resturns
        --------
            list of int
        '''
        numbers = map(int, numbers)
        ans = set()
        while len(numbers) > 1:
            a, b = numbers[:2]
            numbers = numbers[2:]
            # should be range a,b+1 but the biologists are weired
            for n in range(a - 1, b):
                ans.add(n)
        return ans

    def get_pairs(things):
        '''
        ???????
        '''
        current = []
        for thing in things:
            if thing[0] == 'm':
                current.append(thing)
            if len(current) == 2:
                yield current
                current = []

    g = nx.Graph()
    fni = {}  # forgi name to networkx node id

    for l in forgi.split('\n')[:-1]:
        line = l.split()
        # only look at interesting lines
        if line[0] not in ['define', 'connect']:
            continue

        # parse stuff like: define s0 1 7 65 71
        if line[0] == 'define':
            # get necessary attributes for a node
            label = line[1][0]
            id = line[1]
            myset = make_node_set(line[2:])
            node_id = len(g)

            # build a node and remember its id
            g.add_node(node_id)
            fni[id] = node_id
            g.node[node_id].update({'label': label, 'contracted': myset})

        # parse stuff like this: connect s3 h2 m1 m3
        if line[0] == 'connect':
            # get nx name of the first element.
            hero = fni[line[1]]
            # connect that hero to the following elements
            for fn in line[2:]:
                g.add_edge(hero, fni[fn])

            # remember what pairs multiloop pieces we are part of
            # i assume that if a stack is part of 2 multiloops they appear in order ..
            # this assumption may be wrong so be careful
            g.node[fni[line[1]]]['multipairs'] = []
            for a, b in get_pairs(line[2:]):
                g.node[fni[line[1]]]['multipairs'].append((fni[a], fni[b]))
    if ignore_inserts:
        # repair inserts by mergind adjacent stacks
        mergelist = []
        for n, d in g.nodes(data=True):
            if d['label'] == "i":
                neighs = g.neighbors(n)
                mergelist.append((neighs[0], (n, neighs[1])))

        # merged is keeping track of already merged nodes.
        # so we always know where to look for :)
        merged = {}
        for s1, mergers in mergelist:
            while s1 not in g.nodes():
                s1 = merged[s1]
            for merger in mergers:
                while merger not in g.nodes():
                    merger = merged[merger]
                merge(g, s1, merger)
                merged[merger] = s1
            # remove resulting self-loops
            if s1 in g[s1]:
                g.remove_edge(s1, s1)
    return g


def merge(graph, node, node2):
    '''
    merge node2 into the node.
    input nodes are strings,
    node is the king
    '''
    for n in graph.neighbors(node2):
        graph.add_edge(node, n)
    graph.node[node]['contracted'].update(graph.node[node2]['contracted'])
    graph.remove_node(node2)


def connect_multiloop(g):
    merge_dict = {}
    for node, d in g.nodes(data=True):
        if d['label'] == 's':
            for a, b in g.node[node]['multipairs']:
                # finding real names... this works by walking up the
                # ladder merge history until the root is found :)
                while a not in g:
                    a = merge_dict[a]
                while b not in g:
                    b = merge_dict[b]
                if a == b:
                    continue
                merge_dict[b] = a
                merge(g, a, b)


'''

TESTING IN A NOTEBOOK:

import itertools
from eden.converter.fasta import fasta_to_sequence
from graphlearn.utils import draw
from eden.converter.rna.rnafold import  rnafold_to_eden as rnafold

%matplotlib inline


# functions to get fasta stuff
def rfam_uri(family_id):
    return 'http://rfam.xfam.org/family/%s/alignment?acc=%s&format=fastau&download=0'%(family_id,family_id)
def rfam_uri(family_id):
    return '%s.fa'%(family_id)
def get_sequences(size=9999,withoutnames=False):
    sequences = itertools.islice( fasta_to_sequence("../toolsdata/RF00005.fa"), size)
    if withoutnames:
        return [ b for (a,b) in sequences ]
    return sequences

# get a sequence
seqs = get_sequences()
# fold it
g=rnafold(seqs).next()

#g.graph.pop('structure')
# create annotater
f=forgiannotate()
# transform
g= f._transform_single(g)

#draw
draw.graphlearn(g,n_graphs_per_line=2, size=20,
                       colormap='Paired', invert_colormap=False,node_border=0.5,vertex_label='part_name',
                       secondary_vertex_label='part_id',
                       vertex_alpha=0.5, edge_alpha=0.4, node_size=200)

'''
