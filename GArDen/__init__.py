#!/usr/bin/env python
"""Provides specialized functors for graphs."""

from toolz import map
from toolz import groupby
from toolz import valmap


def transform(func, iterable):
    """transform."""
    return map(func, iterable)


def compose(func, iterable):
    """compose."""
    return valmap(func, iterable)


def decompose(func, iterable):
    """decompose."""
    return map(func, iterable)


def partition(func, iterable):
    """partition."""
    return groupby(func, iterable)


def partition_list(func, iterable):
    """partition_list."""
    # tee iterable
    # apply func to iterable
    # get a vector of results
    # zip them with the iterable
    # build a defaultdict with that
    # return a conversion to standard dict
    return groupby(func, iterable)


def rank(func, iterable):
    """rank."""
    return sorted(groupby(func, iterable))

# ------------------------------------------------------


def fit(func, iterable):
    """fit."""
    # call fit on func
    return func


def select(func, iterable):
    """select."""
    # call model selection on func
    return func

# perform checks (that can be disabled) to ensure the invariants


# consider vectorize as a decorator:
# vSGD = vectorize(SGD, params, r=2, d=3)
# from now on, data is first vectorized and then passed on to
# fit, predict, transform, decision_function
