import json
from networkx.readwrite import json_graph
from eden import util


def node_link_data_to_eden(input=None, options=dict()):
    """
    Takes a string list in the serialised node_link_data JSON format and yields networkx graphs.

    Parameters
    ----------
    input : string
        A pointer to the data source.

    """

    return _node_link_data_to_eden(util.read(input))


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
