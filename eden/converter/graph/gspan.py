import json
import networkx as nx
from eden import util


def gspan_to_eden(input, options=dict()):
    """Take a string list in the extended gSpan format and yields NetworkX graphs.

    Keyword arguments:
    input -- string with data source
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
            string_list += [line]

    if string_list:
        yield gspan_to_networkx(header, string_list)


def gspan_to_networkx(header, lines):
    """Take a string list in the extended gSpan format and returns a NetworkX graph.

    Keyword arguments:
    header -- string to be used as id for the graph
    lines  -- string list in extended gSpan format
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
        if fc == 'e':
            src = int(tokens[1])
            dst = int(tokens[2])
            label = tokens[3]
            graph.add_edge(src, dst, label=label)
            attribute_str = ' '.join(tokens[4:])
            if attribute_str.strip():
                attribute_dict = json.loads(attribute_str)
                graph.edge[src][dst].update(attribute_dict)

    assert(len(graph) > 0), 'ERROR: generated empty graph. Perhaps wrong format?'
    return graph


def eden_to_gspan(graphs, filename):
    """Write list of graphs to gSpan file..

    Keyword arguments:
    graphs    -- list of NetworkX graphs
    filename  -- name for the gSpan file
    """
    f = open(filename, 'w')
    for i, graph in enumerate(graphs):
        f.write('t # ' + str(i) + '\n')

        for node, data in graph.nodes_iter(data=True):
            f.write('v ' + str(node) + ' ' + data['label'] + '\n')

        for src, dst, data in graph.edges_iter(data=True):
            f.write('e ' + str(src) + ' ' + str(dst) + ' ' + data['label'] + '\n')
