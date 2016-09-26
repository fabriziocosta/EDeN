import dill
try:
    from itertools import izip_longest # Python 2
except ImportError:
    from itertools import zip_longest as izip_longest # Python 3
from sklearn.base import BaseEstimator, TransformerMixin

__author__ = "Fabrizio Costa"
__copyright__ = "Copyright 2015, Fabrizio Costa"
__credits__ = ["Fabrizio Costa", "Bjoern Gruening"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Fabrizio Costa"
__email__ = "costa@informatik.uni-freiburg.de"
__status__ = "Production"

_bitmask_ = 4294967295


class AbstractVectorizer(BaseEstimator, TransformerMixin):
    """Interface declaration for the Vectorizer class."""

    def annotate(self, graphs, estimator=None, reweight=1.0, relabel=False):
        raise NotImplementedError("Should have implemented this")

    def set_params(self, **args):
        raise NotImplementedError("Should have implemented this")

    def fit(self, graphs):
        raise NotImplementedError("Should have implemented this")

    def partial_fit(self, graphs):
        raise NotImplementedError("Should have implemented this")

    def fit_transform(self, graphs):
        raise NotImplementedError("Should have implemented this")

    def transform(self, graphs):
        raise NotImplementedError("Should have implemented this")

    def transform_single(self, graph):
        raise NotImplementedError("Should have implemented this")

    def predict(self, graphs, estimator):
        raise NotImplementedError("Should have implemented this")

    def similarity(self, graphs, ref_instance=None):
        raise NotImplementedError("Should have implemented this")

    def distance(self, graphs, ref_instance=None):
        raise NotImplementedError("Should have implemented this")


def run_dill_encoded(what):
    """Use dill as replacement for pickle to enable multiprocessing on instance methods"""

    fun, args = dill.loads(what)
    return fun(*args)


def apply_async(pool, fun, args, callback=None):
    """
    Wrapper around apply_async() from multiprocessing, to use dill instead of pickle.
    This is a workaround to enable multiprocessing of classes.
    """
    return pool.apply_async(run_dill_encoded, (dill.dumps((fun, args)),), callback=callback)


def fast_hash_2(dat_1, dat_2, bitmask=_bitmask_):
    return int(hash((dat_1, dat_2)) & bitmask) + 1


def fast_hash_3(dat_1, dat_2, dat_3, bitmask=_bitmask_):
    return int(hash((dat_1, dat_2, dat_3)) & bitmask) + 1


def fast_hash_4(dat_1, dat_2, dat_3, dat_4, bitmask=_bitmask_):
    return int(hash((dat_1, dat_2, dat_3, dat_4)) & bitmask) + 1


def fast_hash(vec, bitmask=_bitmask_):
    return int(hash(tuple(vec)) & bitmask) + 1


def fast_hash_vec(vec, bitmask=_bitmask_):
    hash_vec = []
    running_hash = 0xAAAAAAAA
    for i, vec_item in enumerate(vec):
        running_hash ^= hash((running_hash, vec_item, i))
        hash_vec.append(int(running_hash & bitmask) + 1)
    return hash_vec


def chunks(iterable, n):
    """chunks."""
    iterable = iter(iterable)
    while True:
        items = []
        try:
            for i in range(n):
                it = next(iterable)
                items.append(it)
        finally:
            if items:
                yield items


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)
