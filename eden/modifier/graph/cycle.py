import eden
from collections import defaultdict

from sklearn.base import BaseEstimator, ClassifierMixin
import networkx as nx


class Annotator(BaseEstimator, ClassifierMixin):
    def fit(self):
        return self

    def transform(self, graphs, part_id='part_id', part_name='part_name',
                  attribute_name='label'):
        '''
        Parameters
        ----------
        graphs  nx.graph iterator
        part_id  attributename to write the part_id
        part_name attributename to write the part_name


        Returns
        -------
        annotated graph.


        it is important to see that the part_name is not
        equivalent to the part_id.
        all circles may be called 'circle', but each
        structure has its own id.
        '''
        for g in graphs:
            yield self._transform_single(g, part_id=part_id,
                                         part_name=part_name,
                                         attribute_name=attribute_name)

    def _transform_single(self, graph, part_id='part_id',
                          part_name='part_name', attribute_name='label'):
        # letz annotateo oOoO

        # make basic annotation
        for n, d in graph.nodes(data=True):
            d['__cycle'] = list(node_to_cycle(graph, n))
            d['__cycle'].sort()
            graph.node[n][part_id] = set()
            # graph.node[n][part_name] = set()

        # process cycle annotation
        def get_name(graph, n, attribute_name):
            return _getname(graph, n)

        namedict = {}
        for n, d in graph.nodes(data=True):
            idd = _fhash(d['__cycle'])
            name = namedict.get(idd, None)
            if name is None:
                name = get_name(graph, n, attribute_name)
                namedict[idd] = name
            for nid in d['__cycle']:
                graph.node[nid][part_id].add(idd)
                # graph.node[nid][part_name].add(name)

        # transform sets to lists
        for n, d in graph.nodes(data=True):
            d[part_id] = list(d[part_id])
            d[part_name] = [namedict[_idd] for _idd in d[part_id]]
            d[part_id].sort()
            d[part_name].sort()

        return graph


def _fhash(stuff):
    return eden.fast_hash(stuff, 2 ** 20 - 1)


def node_to_cycle(graph, n, attribute_name='label', min_cycle_size=3):
    """Node to cycle.

    :param graph:
    :param n: start node
    :param min_cycle_size:
    :return:  a cycle the node belongs to

    so we start in node n,
    then we expand 1 node further in each step.
    if we meet a node we had before we found a cycle.

    there are 3 possible cases.
        - frontier hits frontier -> cycle of even length
        - frontier hits visited nodes -> cycle of uneven length
        - it is also possible that the newly found cycle doesnt contain our
        start node. so we check for that
    """

    def close_cycle(collisions, parent, root, graph):
        """We found a cycle.

        But that does not say that the root node is part of that cycle.
        """

        def extend_path_to_root(work_list, parent_dict, root, graph):
            """Extend.

            :param work_list: list with start node
            :param parent_dict: the tree like dictionary that contains each
            nodes parent(s)
            :param root: root node. probably we dont really need this since the
            root node is the orphan
            :return: cyclenodes or none

             --- mm we dont care if we get the shortest path.. that is true for
             cycle checking.. but may be a
             problem in cycle finding.. dousurururururu?
            """
            current = work_list[-1]
            while current != root:

                # if we have 1 partent, we use it
                if len(parent_dict[current]) > 1:
                    work_list.append(parent_dict[current][0])

                # otherwise we look at all of them.
                else:

                    bestparent = parent_dict[current][0]
                    bestlabel = graph.node[bestparent][attribute_name]
                    for parent in parent_dict[current]:
                        if graph.node[parent][attribute_name] < bestlabel:
                            bestlabel = graph.node[parent][attribute_name]
                            bestparent = parent
                    work_list.append(bestparent)

                current = work_list[-1]
            return work_list[:-1]

        # any should be fine. e is closing a cycle,
        # note: e might contain more than one hit but we dont care
        e = collisions.pop()
        # print 'r',e
        # we closed a cycle on e so e has 2 parents...
        li = parent[e]
        a = [li[0]]
        b = [li[1]]
        # print 'pre',a,b
        # get the path until the root node
        a = extend_path_to_root(a, parent, root, graph)
        b = extend_path_to_root(b, parent, root, graph)
        # print 'comp',a,b
        # if the paths to the root node dont overlap, the root node must be
        # in the loop
        a = set(a)
        b = set(b)
        intersect = a & b
        if len(intersect) == 0:
            paths = a | b
            paths.add(e)
            paths.add(root)
            return paths
        return False

    # START OF ACTUAL FUNCTION
    no_cycle_default = set([n])
    frontier = set([n])
    step = 0
    visited = set()
    parent = defaultdict(list)

    while frontier:
        # print frontier
        step += 1

        # give me new nodes:
        next = []
        for front_node in frontier:
            new = set(graph.neighbors(front_node)) - visited
            next.append(new)
            for e in new:
                parent[e].append(front_node)

        # we merge the new nodes.   if 2 sets collide, we found a cycle of
        # even length
        while len(next) > 1:
            # merge
            s1 = next[1]
            s2 = next[0]
            merge = s1 | s2

            # check if we havee a cycle   => s1,s2 overlap
            if len(merge) < len(s1) + len(s2):
                col = s1 & s2
                cycle = close_cycle(col, parent, n, graph)
                if cycle:
                    if step * 2 > min_cycle_size:
                        return cycle
                    return no_cycle_default

            # delete from list
            next[0] = merge
            del next[1]
        next = next[0]

        # now we need to check for cycles of uneven length => the new nodes hit
        # the old frontier
        if len(next & frontier) > 0:
            col = next & frontier
            cycle = close_cycle(col, parent, n, graph)
            if cycle:
                if step * 2 - 1 > min_cycle_size:
                    return cycle
                return no_cycle_default

        # we know that the current frontier didntclose cycles so we dont need
        # to look at them again
        visited = visited | frontier
        frontier = next
    return no_cycle_default


def _getname(graph, n, attribute_name='label'):
    # more complicated naming scheme  looks at cycle and
    # uses lexicographicaly smallest name.
    # trivial case with cycle length 1:
    if len(graph.node[n]['__cycle']) == 1:
        return graph.node[n][attribute_name]
    # first we need the nodelabels in order
    g = nx.Graph(graph.subgraph(graph.node[n]['__cycle']))
    startnode = graph.node[n]['__cycle'][0]
    neighbor = g.neighbors(startnode)[0]
    g.remove_edge(startnode, neighbor)

    result = []
    while len(g) > 1:
        neighbor = g.neighbors(startnode)[0]
        result.append(g.node[startnode][attribute_name])
        g.remove_node(startnode)
        startnode = neighbor
    result.append(g.node[startnode][attribute_name])

    #  we have the labels in order now.
    # we want to cycle until we find the lex lowest configuration
    def min_lex(li):
        def all_lex(li):
            n = len(li)
            for i in range(n):
                yield li
                li = li[1:] + [li[0]]

        il = list(li)
        il.reverse()
        return ''.join(min(min(all_lex(li)), min(all_lex(il))))

    return min_lex(result)


'''
TESTING IN A NOTEBOOK:

%matplotlib inline
# get data
from eden.converter.graph.gspan import gspan_to_eden
from itertools import islice
def get_graphs(dataset_fname, size=100):
    return  islice(gspan_to_eden(dataset_fname),size)
dataset_fname = 'toolsdata/bursi.pos.gspan'
from graphlearn.utils import draw

ca = cycleAnnotation()
graphs=get_graphs(dataset_fname,10)

for gg in graphs:
    g = ca._transform_single(gg)
    draw.graphlearn(g,n_graphs_per_line=2, size=20,
                       colormap='Paired', invert_colormap=False,
                       node_border=0.5,vertex_label='part_name',
                       secondary_vertex_label='part_id',
                       vertex_alpha=0.5, edge_alpha=0.4, node_size=200)

'''
