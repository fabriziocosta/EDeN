"""EasyMKL is a Multiple Kernel Learning algorithm.

@author: Michele Donini
@email: mdonini@math.unipd.it

EasyMKL: a scalable multiple kernel learning algorithm
by Fabio Aiolli and Michele Donini

Paper @ http://www.math.unipd.it/~mdonini/publications.html
"""

from cvxopt import matrix, solvers, mul
import numpy as np


class GEasyMKL():
    """EasyMKL is a Multiple Kernel Learning algorithm.

    The parameter lam (lambda) has to be validated from 0 to 1.

    For more information:
    EasyMKL: a scalable multiple kernel learning algorithm
        by Fabio Aiolli and Michele Donini.

    Paper @ http://www.math.unipd.it/~mdonini/publications.html
    """

    def __init__(self, lam=0.1, tracenorm=True):
        """init."""
        self.lam = lam
        self.tracenorm = tracenorm

        self.list_Ktr = None
        self.labels = None
        self.gamma = None
        self.weights = None
        self.traces = {}

    def sum_kernels(self, list_K, weights=None):
        """Return the kernel created by averaging of all the kernels."""
        k = matrix(0.0, (list_K[0].size[0], list_K[0].size[1]))
        if weights is None:
            for ker in list_K:
                k += ker
        else:
            for w, ker in zip(weights, list_K):
                k += w * ker
        return k

    def traceN(self, k):
        """traceN."""
        return sum([k[i, i] for i in range(k.size[0])]) / k.size[0]

    def train(self, dict_list_Ktr, labels):
        """Train.

        list_Ktr : dict of param:kernel of the training examples
        labels : array of the labels of the training examples
        """
        self.list_Ktr = dict_list_Ktr
        for ik in self.list_Ktr:
            k = self.list_Ktr[ik]
            self.traces[ik] = self.traceN(k)
        if self.tracenorm:
            for ik in self.list_Ktr:
                k = self.list_Ktr[ik]
                self.list_Ktr[ik] = k / self.traceN(k)

        set_labels = set(labels)
        if len(set_labels) != 2:
            print 'The different labels are not 2'
            return None
        elif (-1 in set_labels and 1 in set_labels):
            self.labels = labels
        else:
            poslab = max(set_labels)
            self.labels = matrix(np.array([1. if i == poslab else -1.
                                           for i in labels]))

        # Sum of the kernels
        # ker_matrix = matrix(self.sum_kernels(self.list_Ktr.values())) / float(len(dict_list_Ktr))
        ker_matrix = matrix(self.sum_kernels(self.list_Ktr.values()))

        YY = matrix(np.diag(list(matrix(self.labels))))
        KLL = (1.0 - self.lam) * YY * ker_matrix * YY
        LID = matrix(np.diag([self.lam] * len(self.labels)))
        Q = 2 * (KLL + LID)
        p = matrix([0.0] * len(self.labels))
        G = -matrix(np.diag([1.0] * len(self.labels)))
        h = matrix([0.0] * len(self.labels), (len(self.labels), 1))
        A = matrix([[1.0 if lab == +1 else 0 for lab in self.labels],
                    [1.0 if lab2 == -1 else 0 for lab2 in self.labels]]).T
        b = matrix([[1.0], [1.0]], (2, 1))

        solvers.options['show_progress'] = False  # True
        sol = solvers.qp(Q, p, G, h, A, b)
        # Gamma:
        self.gamma = sol['x']

        # Weights evaluation:
        yg = mul(self.gamma.T, self.labels.T)
        self.weights = {}
        for ik in dict_list_Ktr:
            kermat = dict_list_Ktr[ik]
            b = yg * kermat * yg.T
            self.weights[ik] = b[0]

        norm2 = sum([w for w in self.weights.values()])
        for iw in self.weights:
            self.weights[iw] = self.weights[iw] / norm2

        if self.tracenorm:
            for iw in self.weights:
                self.weights[iw] = self.weights[iw] / self.traces[iw]

        return self.weights

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
    data = load_iris()
    X = data.data
    y = data.target

    # REMARK: labels and kernels are "cvxopt.matrix"
    y = matrix([-1.0 if i == 0 else 1.0 for i in y])
    K_dict = {('rbf', g): matrix(rbf_kernel(X, gamma=g))
              for g in [2**i for i in range(-5, 2)]}
    K_dict.update({('poly', d): matrix(polynomial_kernel(X, degree=d))
                   for d in range(1, 5)})

    easy = GEasyMKL()
    weights_dict = easy.train(K_dict, y)
    print weights_dict
