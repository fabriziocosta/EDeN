import forgi_bulge_graph as fbg

'''

for testing you want to do something like this:

def rfam_uri(family_id):
    return 'http://rfam.xfam.org/family/%s/alignment?acc=%s&format=fastau&download=0'%(family_id,family_id)

def get_graphs(rfam_id = 'RF00005'):
    seqs = fasta_to_sequence(rfam_uri(rfam_id))
    graphs = rnafold_to_eden(seqs, shape_type=5, energy_range=30, max_num=3)
    return graphs

g=annotate(get_graphs().next())
draw(g)

'''


def annotate(graph):
    '''
    input is a graph that has a 'structure' attribute and the first node has id 0.
    when done all the node have an 'entity' attribute that indicates the structural element it is part of.
    there is also entity_short which cuts the entity down to a single letter for prettier printing.
    '''
    # forgi can also output directly to a ssshhhhsss like string.. oh well...
    entity_lookup = {'t': 'dangling start',
                     'f': 'dangling end',
                     'i': 'internal loop',
                     'h': 'hairpin loop',
                     'm': 'multi loop',
                     's': 'stem'
                     }
    # forgi gives me the node ids as (start,end) pairs... 
    def make_node_set(numbers):
        numbers = map(int, numbers)
        ans = set()
        while len(numbers) > 1:
            a, b = numbers[:2]
            numbers = numbers[2:]
            for n in range(a - 1, b):
                ans.add(
                    n)  # should be range a,b+1 but the biologists are weired
        return ans

    # get forgi elements
    bg = fbg.BulgeGraph()
    bg.from_dotbracket(graph.graph['structure'], None)
    forgi = bg.to_bg_string()

    for line in forgi.split('\n')[:-1]:
        # if the line starts with 'define' we know that annotation follows...
        if line[0] == 'd':
            l = line.split()
            # first we see the type 
            entity = l[1][0]
            # then we see a list of nodes of that type.
            for n in make_node_set(l[2:]):
                #we mark those nodes
                graph.node[n]['entity'] = entity_lookup[entity]
                graph.node[n]['entity_short'] = entity
