#!/usr/bin/env python
"""Provides interface declaration."""

from eden.util import vectorize, is_iterable
from eden.graph import Vectorizer
import random
from itertools import tee, izip
from collections import defaultdict


import logging
logger = logging.getLogger(__name__)


"""
preconditions are programs that ensure that a condition/contract is fulfilled
by the input stream given the specific program and parameters_priors.
preconditions take in input all the input to the interface.
postconditions are programs that ensure that a condition/contract is fulfilled
by the output stream given the specific program and parameters_priors.
postconditions take in input all the input to the interface and the produced
output.
pre/postconditions evaluate to True or False.

E.g. if the precondition tests the 'inducible' contract then it will check that
each data instance stores an attribute called 'target' and the program will be
adapted via optimization on the parameter space sampled using
parameters_priors. Otherwise do testing.

Provide collections of well documented programs (each with as simple an
interface as possible, but not more)
Organize the collection ob the basis of the interfaces.
"""


def sample_parameters_uniformly_at_random(parameters_priors):
    """Sample parameters in parameters dictionaries uniformly at random."""
    if parameters_priors:
        parameters = {}
        for param in parameters_priors:
            if is_iterable(parameters_priors[param]):
                value = random.choice(parameters_priors[param])
            else:
                value = parameters_priors[param]
            parameters[param] = value
        return parameters
    else:
        return None


def convert(iterable, program=None, precondition=None,
            postcondition=None, parameters_priors=None):
    """Map an input data type to a graph."""
    parameters = sample_parameters_uniformly_at_random(parameters_priors)
    if parameters:
        program.set_params(**parameters)
    return program.transform(iterable)


def associate(iterable, program=None, precondition=None,
              postcondition=None, parameters_priors=None):
    """Map a graph to an output data type."""
    pass


def partition(iterable, program=None,
              precondition=None, postcondition=None, parameters_priors=None):
    """Map a graph to an iterator over the input graphs.

    Example: a graph to the set of graphs that are in the same part.
    Example: for a hierarchical clustering return an iterator over a tree
    structure: the iterator exposes the interface for advancing on other
    elements that have the same parent or advances to the parent.
    """
    # convert iterable graphs in vector data matrix
    parameters = sample_parameters_uniformly_at_random(parameters_priors)
    vectorizer = Vectorizer(r=parameters['r'], d=parameters['d'])
    iterable, iterable_ = tee(iterable)
    data_matrix = vectorize(iterable_,
                            vectorizer=vectorizer,
                            fit_flag=False,
                            n_blocks=5,
                            block_size=None,
                            n_jobs=parameters['n_jobs'])
    predictions = program.fit_predict(data_matrix)
    partition_list = defaultdict(list)
    for prediction, graph in izip(predictions, iterable):
        partition_list[prediction].append(graph.copy())
    return partition_list


def compose(iterable, program=None,
            precondition=None, postcondition=None, parameters_priors=None):
    """Map iterator over graphs to a graph.

    Example: receive two iterators on corresponding graphs and yield an
    iterator over a composite graph.
    """
    pass


def decompose(iterable, program=None,
              precondition=None, postcondition=None, parameters_priors=None):
    """Map a graph to an iterator over subgraphs of the input graph."""
    pass


def transform(iterable, program=None,
              precondition=None, postcondition=None, parameters_priors=None):
    """Map a graph to a graph.

    The postcondition can be:
    - compress : |V_out| < |V_in| or |E_out| < |E_in|
    - expand : |V_out| > |V_in| or |E_out| > |E_in|
    - preserve : the graph structure is identical but the attributes can change
    - None : no constraints
    """
    parameters = sample_parameters_uniformly_at_random(parameters_priors)
    if parameters:
        program.set_params(**parameters)
    return program.transform(iterable)


def contruct(iterable, program=None,
             precondition=None, postcondition=None, parameters_priors=None):
    """Map a graph to iterator over similar but novel graphs.

    Example: learn probability distribution over graphs given a finite example
    set and sample a stream of graphs from the same probability distribution.
    """
    pass
