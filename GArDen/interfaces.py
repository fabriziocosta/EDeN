#!/usr/bin/env python
"""Provides interface declaration."""

from eden.util import is_iterable
import random
from itertools import tee, izip
from collections import defaultdict
from GArDen.order import OrdererWrapper

import logging
logger = logging.getLogger(__name__)


"""
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


def precondition(iterable=None, program=None):
    """Ensure that a condition/contract is fulfilled.

    The condition has to be fulfilled by the input stream given the specific
    program with the chosen parameters.

    Preconditions take in input all the input to the interface.

    Preconditions evaluate to True or False.
    """
    return True


def precond_is_classifier(iterable=None, program=None):
    """Ensure that a program can do classification."""
    if program.__class__.__name__ in ['SGDClassifier',
                                      'LogisticRegression']:
        return True
    else:
        return False


def precond_is_regressor(iterable=None, program=None):
    """Ensure that a program can do regression."""
    if program.__class__.__name__ in ['SGDRegressor']:
        return True
    else:
        return False


def precond_is_knn(iterable=None, program=None):
    """Ensure that a program can do k-nearest-neighbors."""
    if program.__class__.__name__ in ['NearestNeighbors']:
        return True
    else:
        return False


def precond_is_wrapped(iterable=None, program=None):
    """Ensure that a program is already wrapped."""
    if program.__class__.__name__ in ['KNNWrapper',
                                      'ClassifierWrapper',
                                      'RegressorWrapper']:
        return True
    else:
        return False


def postcondition(iterable=None, program=None):
    """Ensure that a condition/contract is fulfilled.

    The condition has to be fulfilled by the output stream given the specific
    program with the chosen parameters.

    Postconditions take in input all the output from the interface.

    Postconditions evaluate to True or False.
    """
    return True


def convert(iterable, program=None, precondition=precondition,
            postcondition=postcondition, parameters_priors=None):
    """Map an input data type to a graph."""
    try:
        parameters = sample_parameters_uniformly_at_random(parameters_priors)
        if parameters:
            program.set_params(**parameters)
        if precondition(iterable=iterable, program=program) is False:
            raise Exception('precondition failed')
        out_iterable = program.transform(iterable)
        if postcondition(iterable=out_iterable, program=program) is False:
            raise Exception('postcondition failed')
        for item in out_iterable:
            yield item
    except Exception as e:
        logger.debug('Error. Reason: %s' % e)
        logger.debug('Exception', exc_info=True)


def model(iterable, program=None, precondition=precondition,
          postcondition=postcondition, parameters_priors=None):
    """Induce a predictive model.

    The induction is done by optimizing the parameters and the
    hyper parameters.
    Return a biased program that can be used in the other operators.
    """
    try:
        parameters = sample_parameters_uniformly_at_random(parameters_priors)
        if parameters:
            program.set_params(**parameters)
        if precondition(iterable=iterable, program=program) is False:
            raise Exception('precondition failed')
        program = program.fit(iterable)
        if postcondition(iterable=None, program=program) is False:
            raise Exception('postcondition failed')
        return program
    except Exception as e:
        logger.debug('Error. Reason: %s' % e)
        logger.debug('Exception', exc_info=True)


def predict(iterable, program=None, precondition=precondition,
            postcondition=postcondition, parameters_priors=None):
    """Map a graph to an output data type."""
    try:
        parameters = sample_parameters_uniformly_at_random(parameters_priors)
        if parameters:
            program.set_params(**parameters)
        if precondition(iterable=iterable, program=program) is False:
            raise Exception('precondition failed')
        predictions = program.predict(iterable)
        if postcondition(iterable=predictions, program=program) is False:
            raise Exception('postcondition failed')
        return predictions
    except Exception as e:
        logger.debug('Error. Reason: %s' % e)
        logger.debug('Exception', exc_info=True)


def partition(iterable, program=None, precondition=precondition,
              postcondition=postcondition, parameters_priors=None):
    """Map a graph to an iterator over the input graphs.

    Example: a graph to the set of graphs that are in the same part.
    Example: for a hierarchical clustering return an iterator over a tree
    structure: the iterator exposes the interface for advancing on other
    elements that have the same parent or advances to the parent.
    """
    try:
        # the wrapping has to be done externally so to allow programs
        # that work on graphs to act directly
        # the wrapper provides the vectorization support
        # program = ClustererWrapper(program=program)

        parameters = sample_parameters_uniformly_at_random(parameters_priors)
        if parameters:
            program.set_params(**parameters)
        if precondition(iterable=iterable, program=program) is False:
            raise Exception('precondition failed')
        iterable, iterable_ = tee(iterable)
        predictions = program.fit_predict(iterable_)
        if postcondition(iterable=predictions, program=program) is False:
            raise Exception('postcondition failed')
        partition_dict = defaultdict(list)
        for prediction, graph in izip(predictions, iterable):
            partition_dict[prediction].append(graph.copy())
        return partition_dict
    except Exception as e:
        logger.debug('Error. Reason: %s' % e)
        logger.debug('Exception', exc_info=True)


def order(iterable, program=None, precondition=precondition,
          postcondition=postcondition, parameters_priors=None):
    """Map iterable to iterable.

    Example: receive an iterator over graphs and yield an
    iterator over the same graphs but sorted by density.
    """
    try:
        # the wrapper provides the vectorization support
        program = OrdererWrapper(program=program)

        parameters = sample_parameters_uniformly_at_random(parameters_priors)
        if parameters:
            program.set_params(**parameters)
        if precondition(iterable=iterable, program=program) is False:
            raise Exception('precondition failed')
        iterable, iterable_ = tee(iterable)
        scores = program.decision_function(iterable_)
        if postcondition(iterable=scores, program=program) is False:
            raise Exception('postcondition failed')
        for score, graph in sorted(izip(scores, iterable)):
            yield graph
    except Exception as e:
        logger.debug('Error. Reason: %s' % e)
        logger.debug('Exception', exc_info=True)


def compose(iterable, program=None, precondition=precondition,
            postcondition=postcondition, parameters_priors=None):
    """Map iterator over graphs to a graph.

    Example: receive iterator over pairs (or lists) of graphs and yield an
    iterator over a composite graph.
    """
    try:
        parameters = sample_parameters_uniformly_at_random(parameters_priors)
        if parameters:
            program.set_params(**parameters)
        if precondition(iterable=iterable, program=program) is False:
            raise Exception('precondition failed')
        out_iterable = program.transform(iterable)
        if postcondition(iterable=out_iterable, program=program) is False:
            raise Exception('postcondition failed')
        for item in out_iterable:
            yield item
    except Exception as e:
        logger.debug('Error. Reason: %s' % e)
        logger.debug('Exception', exc_info=True)


def decompose(iterable, program=None, precondition=precondition,
              postcondition=postcondition, parameters_priors=None):
    """Map a graph to an iterator over subgraphs of the input graph."""
    try:
        parameters = sample_parameters_uniformly_at_random(parameters_priors)
        if parameters:
            program.set_params(**parameters)
        if precondition(iterable=iterable, program=program) is False:
            raise Exception('precondition failed')
        out_iterable = program.transform(iterable)
        if postcondition(iterable=out_iterable, program=program) is False:
            raise Exception('postcondition failed')
        for item in out_iterable:
            yield item
    except Exception as e:
        logger.debug('Error. Reason: %s' % e)
        logger.debug('Exception', exc_info=True)


def transform(iterable, program=None, precondition=precondition,
              postcondition=postcondition, parameters_priors=None):
    """Map a graph to a graph.

    The postcondition can be:
    - compress : |V_out| < |V_in| or |E_out| < |E_in|
    - expand : |V_out| > |V_in| or |E_out| > |E_in|
    - preserve : the graph structure is identical but the attributes can change
    - None : no constraints
    """
    try:
        parameters = sample_parameters_uniformly_at_random(parameters_priors)
        if parameters:
            program.set_params(**parameters)
        if precondition(iterable=iterable, program=program) is False:
            raise Exception('precondition failed')
        out_iterable = program.transform(iterable)
        if postcondition(iterable=out_iterable, program=program) is False:
            raise Exception('postcondition failed')
        for item in out_iterable:
            yield item
    except Exception as e:
        logger.debug('Error. Reason: %s' % e)
        logger.debug('Exception', exc_info=True)


def construct(iterable, program=None, precondition=precondition,
              postcondition=postcondition, parameters_priors=None):
    """Map a graph to iterator over similar but novel graphs.

    Example: learn probability distribution over graphs given a finite example
    set and sample a stream of graphs from the same probability distribution.
    """
    try:
        parameters = sample_parameters_uniformly_at_random(parameters_priors)
        if parameters:
            program.set_params(**parameters)
        if precondition(iterable=iterable, program=program) is False:
            raise Exception('precondition failed')
        out_iterable = program.transform(iterable)
        if postcondition(iterable=out_iterable, program=program) is False:
            raise Exception('postcondition failed')
        for item in out_iterable:
            yield item
    except Exception as e:
        logger.debug('Error. Reason: %s' % e)
        logger.debug('Exception', exc_info=True)
