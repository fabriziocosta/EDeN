import logging
logger = logging.getLogger(__name__)


"""
preconditions are programs that ensure that a contract is fulfilled by the input stream.
postconditions are programs that ensure that a contract is fulfilled by the output stream.
They are callable functions with input iterable/output iterable and program as parameters that evaluate to
True or False.

E.g. if the precondition is 'inducible' then perform optimization of parameters using parameters_priors,
unpacking the input in pairs of (data_in, target_out). Otherwise do testing.

Provide collections of well documented programs (each with as simple an interface as possible, but not more)
Organize the collection ob the basis of the interfaces.
"""


def associate(iterable, precondition=None, postcondition=None, program=None, parameters_priors=None):
    """Map a graph to another, simple data type."""
    pass


def partition(iterable, precondition=None, postcondition=None, program=None, parameters_priors=None):
    """Map a graph to an iterator over the input graphs.
    Example: a graph to the set of graphs that are in the same part.
    Example: for a hierarchical clustering return an iterator over a tree structure: the iterator exposes
    the interface for advancing on other elements that have the same parent or advances to the parent.
    """
    pass


def compose(iterable, precondition=None, postcondition=None, program=None, parameters_priors=None):
    """Map an iterator over graphs to a graph.
    Example: receive a list of pairs of graphs and yield an iterator over composite a graph."""
    pass


def decompose(iterable, precondition=None, postcondition=None, program=None, parameters_priors=None):
    """Map a graph to an iterator over subgraphs of the input graph."""
    pass


def transform(iterable, precondition=None, postcondition=None, program=None, parameters_priors=None):
    """Map a graph to a graph.
    The postcondition can be:
    - compress : |V_out| < |V_in| or |E_out| < |E_in|
    - expand : |V_out| > |V_in| or |E_out| > |E_in|
    - preserve : the graph structure is identical but the attributes can change
    - None : no constraints
    """
    pass


def contruct(iterable, precondition=None, postcondition=None, program=None, parameters_priors=None):
    """Map a graph to several graphs.
    Example: learn probability distribution over graphs given a finite example set and sample a stream of
    graphs from the same probability distribution."""
    pass
