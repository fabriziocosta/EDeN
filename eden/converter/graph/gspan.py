import json
import networkx as nx
from eden import util

def gspan_to_eden(input = None, options = dict()):
    """
    Takes a string list in the extended gSpan format and yields networkx graphs.

    Parameters
    ----------
    input : string
        A pointer to the data source.

    """

    string_list = []
    for line in util.read( input ):
        if line.strip():
            if line[0] in ['g','t']:
                if string_list:
                    yield gspan_to_networkx(string_list)
                string_list = []
            string_list += [line]

    if string_list:
        yield gspan_to_networkx(string_list)


def gspan_to_networkx(string_list):
    G = nx.Graph()
    for line in string_list:
        if line.strip():
            line_list = line.split()
            fc = line_list[0]
            #process vertices
            if fc in ['v','V'] : 
                vid = int(line_list[1])
                vlabel = line_list[2]
                #lowercase v indicates active viewpoint
                if fc == 'v':
                    weight = 1
                else: #uppercase v indicates no-viewpoint
                    weight = 0.1
                G.add_node(vid, label = vlabel, weight = weight)
                #abstract vertices
                if vlabel[0] == '^':
                    G.node[vid]['nesting'] = True
                #extract the rest of the line  as a JSON string that contains all attributes
                attribute_str=' '.join(line_list[3:])
                if attribute_str.strip():
                    attribute_dict = json.loads(attribute_str)
                    G.node[vid].update(attribute_dict)
            #process edges
            if fc == 'e' : 
                srcid = int(line_list[1])
                destid = int(line_list[2])
                elabel = line_list[3]
                G.add_edge(srcid, destid, label = elabel)
                attribute_str=' '.join(line_list[4:])
                if attribute_str.strip():
                    attribute_dict=json.loads(attribute_str)
                    G.edge[srcid][destid].update(attribute_dict)
    assert(len(G) > 0 ),'ERROR: generated empty graph. Perhaps wrong format?'
    return G

