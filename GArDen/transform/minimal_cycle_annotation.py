import eden
from collections import defaultdict

from sklearn.base import BaseEstimator, ClassifierMixin

import logging
logger = logging.getLogger(__name__)


class AnnotateMinimalCycles(BaseEstimator, ClassifierMixin):

    def fit(self):
        return self

    def transform(self, graphs, attribute='label', part_id='part_id', part_name='part_name'):
        '''
        Parameters
        ----------
        graphs  nx.graph iterator
        part_id  attributename to write the part_id
        part_name attributename to write the part_name


        Returns
        -------
        annotated graph.


        it is important to see that the part_name is not equivalent to the part_id.
        all circles may be called 'circle', but each structure has its own id.

        '''
        try:
            for graph in graphs:
                yield self._transform_single(graph, attribute=attribute, part_id=part_id, part_name=part_name)
        except Exception as e:
            logger.debug('Failed iteration. Reason: %s' % e)
            logger.debug('Exception', exc_info=True)

    def _transform_single(self, graph, attribute='label', part_id='part_id', part_name='part_name'):
        # letz annotateo oOoO

        # process cycle annotation
        def get_name(graph, n):
            # labels = sorted([graph.node[i][attribute] for i in graph.node[n]['__cycle']])
            # return ''.join(labels)
            orig = ''.join([graph.node[i][attribute] for i in graph.node[n]['__cycle']])
            normalized = min_lex(orig)
            return normalized

        # make basic annotation
        for n, d in graph.nodes(data=True):
            d['__cycle'] = list(node_to_cycle(graph, n))
            # d['__cycle'].sort()
            graph.node[n][part_id] = set()
            # graph.node[n][part_name] = set()

        namedict = {}
        for n, d in graph.nodes(data=True):
            idd = fhash(d['__cycle'])
            name = namedict.get(idd, None)
            if name is None:
                name = get_name(graph, n)
                namedict[idd] = name
            for nid in d['__cycle']:
                graph.node[nid][part_id].add(idd)
                # graph.node[nid][part_name].add(name)

        # transform sets to lists
        for n, d in graph.nodes(data=True):
            d[part_id] = list(d[part_id])
            d[part_name] = [namedict[id] for id in d[part_id]]

        return graph


def min_lex(s):
    def all_lex(s):
        n = len(s)
        for i in range(n):
            yield s
            s = list(s)
            s_list = s[1:]
            s_list += s[0]
            s = ''.join(s_list)
    return min(all_lex(s))


def fhash(stuff):
    return eden.fast_hash(stuff, 2 ** 20 - 1)


def node_to_cycle(graph, n, min_cycle_size=3):
    """
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
        - it is also possible that the newly found cycle doesnt contain our start node. so we check for that
    """

    def close_cycle(collisions, parent, root):
        '''
            we found a cycle, but that does not say that the root node is part of that cycle..S
        '''

        def extend_path_to_root(work_list, parent_dict, root):
            """
            :param work_list: list with start node
            :param parent_dict: the tree like dictionary that contains each nodes parent(s)
            :param root: root node. probably we dont really need this since the root node is the orphan
            :return: cyclenodes or none

             --- mm we dont care if we get the shortest path.. that is true for cycle checking.. but may be a
             problem in cycle finding.. dousurururururu?
            """
            current = work_list[-1]
            while current != root:
                work_list.append(parent_dict[current][0])
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
        a = extend_path_to_root(a, parent, root)
        b = extend_path_to_root(b, parent, root)
        # print 'comp',a,b
        # of the paths to the root node dont overlap, the root node musst be in the loop
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

        # we merge the new nodes.   if 2 sets collide, we found a cycle of even length
        while len(next) > 1:
            # merge
            s1 = next[1]
            s2 = next[0]
            merge = s1 | s2

            # check if we havee a cycle   => s1,s2 overlap
            if len(merge) < len(s1) + len(s2):
                col = s1 & s2
                cycle = close_cycle(col, parent, n)
                if cycle:
                    if step * 2 > min_cycle_size:
                        return cycle
                    return no_cycle_default

            # delete from list
            next[0] = merge
            del next[1]
        next = next[0]

        # now we need to check for cycles of uneven length => the new nodes hit the old frontier
        if len(next & frontier) > 0:
            col = next & frontier
            cycle = close_cycle(col, parent, n)
            if cycle:
                if step * 2 - 1 > min_cycle_size:
                    return cycle
                return no_cycle_default

        # we know that the current frontier didntclose cycles so we dont need to look at them again
        visited = visited | frontier
        frontier = next
    return no_cycle_default
