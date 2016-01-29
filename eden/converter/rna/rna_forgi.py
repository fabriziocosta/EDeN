#!/usr/bin/env python

# forgi is here: https://github.com/pkerpedjiev/forgi
# i took what i needed and removed unnecessary dependencies
# in this file i try to make a wrapper

import networkx as nx
import eden.modifier.rna.lib_forgi as lib_forgi


def get_abstr_graph(struct):
    # get forgi string
    bg = lib_forgi.BulgeGraph()
    bg.from_dotbracket(struct, None)
    forgi = bg.to_bg_string()

    g = make_abstract_graph(forgi)
    return g


def make_abstract_graph(forgi):
    g = forgi_to_graph(forgi)
    connect_multiloop(g)
    return g


def forgi_to_graph(forgi):
    def make_node_set(numbers):
        '''
        forgi gives me stuff like define STEM START,END,START,END .. we take indices and output a list
        '''
        numbers = map(int, numbers)
        ans = set()
        while len(numbers) > 1:
            a, b = numbers[:2]
            numbers = numbers[2:]
            for n in range(a - 1, b):
                ans.add(n)  # should be range a,b+1 but the biologists are weired
        return ans

    def get_pairs(things):
        '''
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
    return g


def connect_multiloop(g):

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


def edge_parent_finder(abstract, graph):
    '''
    moved this here since i think that it is forgi specific
    '''
    # find out to which abstract node the edges belong
    # finding out where the edge-nodes belong, because the contractor cant possibly do this
    # draw.graphlearn_draw([abstract,graph],size=10, contract=False,vertex_label='id')

    getabstr = {contra: node for node, d in abstract.nodes(data=True) for contra in d.get('contracted', [])}
    # print getabstr
    for n, d in graph.nodes(data=True):
        if 'edge' in d:
            # if we have found an edge node...

            # lets see whos left and right of it:
            # if len is 2 then we hit a basepair, in that case we already have both neighbors
            zomg = graph.neighbors(n)
            if len(zomg) == 1:
                zomg += graph.predecessors(n)

            n1, n2 = zomg

            # case1: ok those belong to the same gang so we most likely also belong there.
            if getabstr[n1] == getabstr[n2]:
                abstract.node[getabstr[n1]]['contracted'].add(n)

            # case2: neighbors belong to different gangs...
            else:
                abstract_intersect = set(abstract.neighbors(getabstr[n1])) & \
                    set(abstract.neighbors(getabstr[n2]))

                # case 3: abstract intersect in radius 1 failed, so lets try radius 2
                if not abstract_intersect:
                    abstract_intersect = set(nx.single_source_shortest_path(abstract, getabstr[n1], 2)) & set(
                        nx.single_source_shortest_path(abstract, getabstr[n2], 2))
                    if len(abstract_intersect) > 1:
                        print "weired abs intersect..."

                for ai_node in abstract_intersect:
                    if 'contracted' in abstract.node[ai_node]:
                        abstract.node[ai_node]['contracted'].add(n)
                    else:
                        abstract.node[ai_node]['contracted'] = set([n])

    return abstract
