#!/usr/bin/env python
"""Provides layout in 2D of vector instances."""

import math

import networkx as nx

import numpy as np
from numpy.linalg import inv

from sklearn.base import BaseEstimator, ClassifierMixin

import logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------


class KKEmbedder(BaseEstimator, ClassifierMixin):
    """Given a graph computes 2D embedding.

    Based on the algorithm from:
    Tomihisa Kamada, and Satoru Kawai. "An algorithm for drawing general
    undirected graphs." Information processing letters 31, no. 1 (1989): 7-15.
    """

    def __init__(self, stop_eps=1, n_iter=20,
                 init_pos=None):
        """Constructor."""
        self.stop_eps = stop_eps
        self.n_iter = n_iter
        self.init_pos = init_pos
        self.dms = []

    def _compute_all_pairs(self, graph, weight=None, normalize=False):
        lengths = nx.all_pairs_dijkstra_path_length(graph, weight=weight)
        max_length = max([max(lengths[i].values()) for i in lengths])
        if normalize:
            for i in lengths:
                for j in lengths[i]:
                    lengths[i][j] = float(lengths[i][j]) / max_length
        return lengths

    def _compute_initial_pos(self, graph):
        _radius = 1
        _offset = 0
        n = len(graph)
        pos = {id: np.array([_radius * math.cos(theta - math.pi / 2) + _offset,
                             _radius * math.sin(theta - math.pi / 2) + _offset]
                            )
               for id, theta in enumerate(
            np.linspace(0, 2 * math.pi * (1 - 1 / float(n)), num=n))}
        return pos

    def _compute_dE(self, pos=None, lengths=None, weights=None, m=None):
        dEx = 0
        dEy = 0
        d2Ex2 = 0
        d2Ey2 = 0
        d2Exy = 0
        d2Eyx = 0
        for i in pos:
            if i != m:
                xmi = pos[m][0] - pos[i][0]
                ymi = pos[m][1] - pos[i][1]
                xmi2 = xmi * xmi
                ymi2 = ymi * ymi
                xmi_ymi2 = xmi2 + ymi2
                lmi = lengths[m][i]
                kmi = weights[m][i] / (lmi * lmi)
                dEx += kmi * (xmi - (lmi * xmi) / math.sqrt(xmi_ymi2))
                dEy += kmi * (ymi - (lmi * ymi) / math.sqrt(xmi_ymi2))
                d2Ex2 += kmi * (1 - (lmi * ymi2) / math.pow(xmi_ymi2, 1.5))
                d2Ey2 += kmi * (1 - (lmi * xmi2) / math.pow(xmi_ymi2, 1.5))
                res = kmi * (lmi * xmi * ymi) / math.pow(xmi_ymi2, 1.5)
                d2Exy += res
                d2Eyx += res
        return dEx, dEy, d2Ex2, d2Ey2, d2Exy, d2Eyx

    def _compute_dm(self, pos=None, lengths=None, weights=None, m=None):
        dEx = 0
        dEy = 0
        for i in pos:
            if i != m:
                xmi = pos[m][0] - pos[i][0]
                ymi = pos[m][1] - pos[i][1]
                xmi2 = xmi * xmi
                ymi2 = ymi * ymi
                xmi_ymi2 = xmi2 + ymi2
                lmi = lengths[m][i]
                kmi = weights[m][i] / (lmi * lmi)
                dEx += kmi * (xmi - (lmi * xmi) / math.sqrt(xmi_ymi2))
                dEy += kmi * (ymi - (lmi * ymi) / math.sqrt(xmi_ymi2))
        return math.sqrt(dEx * dEx + dEy * dEy)

    def _compute_m(self, pos=None, lengths=None, weights=None, id=0):
        self.dms = np.array([self._compute_dm(pos, lengths, weights, m)
                             for m in pos])
        m = np.argsort(-self.dms)[id]
        return m

    def _compute_dxdy(self, pos=None, lengths=None, weights=None, m=None):
        dEx, dEy, d2Ex2, d2Ey2, d2Exy, d2Eyx = self._compute_dE(pos,
                                                                lengths,
                                                                weights,
                                                                m)
        A = np.array([[d2Ex2, d2Exy], [d2Eyx, d2Ey2]])
        B = np.array([[-dEx], [-dEy]])
        X = inv(A).dot(B)
        dx = X[0]
        dy = X[1]
        return dx, dy

    def _update(self, pos=None, lengths=None, weights=None):
        m = self._compute_m(pos, lengths, weights)
        dx, dy = self._compute_dxdy(pos, lengths, weights, m)
        pos[m][0] += dx
        pos[m][1] += dy
        return m

    def _scale(self, init_pos):
        _min = -0.5
        _max = 0.5
        pos = dict()
        max_x = max([init_pos[id][0] for id in init_pos])
        min_x = min([init_pos[id][0] for id in init_pos])
        max_y = max([init_pos[id][1] for id in init_pos])
        min_y = min([init_pos[id][1] for id in init_pos])
        for id in init_pos:
            x = init_pos[id][0]
            y = init_pos[id][1]
            # standardize
            x = (x - min_x) / (max_x - min_x)
            y = (y - min_y) / (max_y - min_y)
            # rescale
            x = x * (_max - _min) + _min
            y = y * (_max - _min) + _min
            pos[id] = np.array([x, y])
        return pos

    def _compute_weights(self, graph):
        weights = np.ones((len(graph), len(graph)))
        for u, v in graph.edges():
            val = graph.edge[u][v].get('weight', 1)
            weights[u][v] = val
        return weights

    def transform(self, graph, normalize=True):
        """Transform."""
        lengths = self._compute_all_pairs(graph, weight='len',
                                          normalize=normalize)
        weights = self._compute_weights(graph)
        if self.init_pos is None:
            pos = self._compute_initial_pos(graph)
        else:
            pos = self._scale(self.init_pos)
        effective_n_iter = self.n_iter * len(graph)
        for i in range(effective_n_iter):
            m = self._update(pos, lengths, weights)
            if i % 100 == 0:
                logger.debug('iteration %d/%d score:%.2f threshold:%.2f' %
                             (i, effective_n_iter, self.dms[m], self.stop_eps))
            if self.dms[m] < self.stop_eps or self.dms[m] != self.dms[m]:
                logger.debug('Stopped at iteration %d/%d with score %.2f' %
                             (i, effective_n_iter, self.dms[m]))
                break
        return pos
