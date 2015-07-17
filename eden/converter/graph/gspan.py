import json
import networkx as nx
from eden import util
import logging
logger = logging.getLogger(__name__)


def gspan_to_eden(input, options=dict()):
    """Take a string list in the extended gSpan format and yields NetworkX graphs.

    Args:
        input: data source, can be a list of strings, a file name or a url
    Returns:
        NetworkX graph generator
    Raises:
        Exception: if a graph is empty
    """

    header = ''
    string_list = []
    for line in util.read(input):
        if line.strip():
            if line[0] in ['g', 't']:
                if string_list:
                    yield gspan_to_networkx(header, string_list)
                string_list = []
                header = line
            else:
                string_list += [line]
    if string_list:
        yield gspan_to_networkx(header, string_list)


def gspan_to_networkx(header, lines):
    """Take a string list in the extended gSpan format and returns a NetworkX graph.

    Args:
        header: string to be used as id for the graph
        lines: string list in extended gSpan format
    Returns:
        NetworkX graph
    Raises:
        Exception: if a graph is empty
    """

    # remove empty lines
    lines = [line for line in lines if line.strip()]

    graph = nx.Graph(id=header)
    for line in lines:
        tokens = line.split()
        fc = tokens[0]

        # process vertices
        if fc in ['v', 'V']:
            id = int(tokens[1])
            label = tokens[2]
            weight = 1.0

            # uppercase V indicates no-viewpoint, in the new EDeN this is simulated via a smaller weight
            if fc == 'V':
                weight = 0.1

            graph.add_node(id, ID=id, label=label, weight=weight)

            # extract the rest of the line  as a JSON string that contains all attributes
            attribute_str = ' '.join(tokens[3:])
            if attribute_str.strip():
                attribute_dict = json.loads(attribute_str)
                graph.node[id].update(attribute_dict)

        # process edges
        elif fc == 'e':
            src = int(tokens[1])
            dst = int(tokens[2])
            label = tokens[3]
            graph.add_edge(src, dst, label=label)
            attribute_str = ' '.join(tokens[4:])
            if attribute_str.strip():
                attribute_dict = json.loads(attribute_str)
                graph.edge[src][dst].update(attribute_dict)
        else:
            logger.debug('line begins with unrecognized code: %s' % fc)
    if len(graph) == 0:
        raise Exception('ERROR: generated empty graph. Perhaps wrong format?')
    return graph


def eden_to_gspan(graphs, filename):
    """Write list of graphs to gSpan file.

    Args:
        graphs: list of NetworkX graphs
        filename: name for the gSpan file
    Raises:
        Exception: if a graph is empty
    """

    with open(filename, 'w') as f:
        for i, graph in enumerate(graphs):
            f.write('t #  %s\n' % i)

            for node, data in graph.nodes_iter(data=True):
                f.write('v %s %s\n' % (node, data['label']))

            for src, dst, data in graph.edges_iter(data=True):
                f.write('e %s %s %s\n' % (src, dst, data['label']))
