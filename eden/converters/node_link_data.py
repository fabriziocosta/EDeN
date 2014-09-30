import json
import networkx as nx
from networkx.readwrite import json_graph

def node_link_data_to_eden(name, input_type='list'):
    """
    Takes a string list in the serialised node_link_data JSON format and yields networkx graphs.

    Parameters
    ----------
    name : string
        A pointer to the data source.

    input_type : ['url','file','string_file']
        If type is 'url' then 'name' is interpreted as a URL pointing to a file.
        If type is 'file' then 'name' is interpreted as a file name.
        If type is 'string_file' then 'name' is interpreted as a file name for a file 
        that contains strings rather than integers. The set of strings are mapped to 
        unique increasing integers and the corresponding vector of integers is returned.
    """

    input_types = ['url','file','list']
    assert(input_type in input_types),'ERROR: input_type must be one of %s ' % input_types

    if input_type is 'file':
        f=open(name,'r')
    elif input_type is 'url':
        import requests
        f=requests.get(name).text.split('\n')
    elif input_type == "list":
        f = name
    return _node_link_data_to_eden(f)        


def _node_link_data_to_eden(serialized_list):
    """Takes a string list in the serialised node_link_data JSON format and yields networkx graphs."""
    for serial_data in serialized_list:
        py_obj = json.loads(serial_data)
        graph = json_graph.node_link_graph(py_obj)
        yield graph



def eden_to_node_link_data(graph_list):
    """Takes a list of networkx graphs and yields serialised node_link_data JSON strings."""
    for G in graph_list:
            json_data = json_graph.node_link_data(G)
            serial_data = json.dumps(json_data)        
            yield serial_data

