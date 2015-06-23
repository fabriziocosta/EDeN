__author__ = "Fabrizio Costa"
__copyright__ = "Copyright 2015, Fabrizio Costa"
__credits__ = ["Fabrizio Costa", "Bjoern Gruening"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Fabrizio Costa"
__email__ = "costa@informatik.uni-freiburg.de"
__status__ = "Production"

import dill
from itertools import izip_longest


def run_dill_encoded(what):
    """
    Use dill as replacement for pickle to enable multiprocessing on instance methods
    """
    fun, args = dill.loads(what)
    return fun(*args)


def apply_async(pool, fun, args, callback=None):
    """
    Wrapper around apply_async() from multiprocessing, to use dill instead of pickle.
    This is a workaround to enable multiprocessing of classes.
    """
    return pool.apply_async(run_dill_encoded, (dill.dumps((fun, args)),), callback=callback)


def fast_hash(vec, bitmask):
    d = 0x01000193
    # Use the FNV algorithm from http://isthe.com/chongo/tech/comp/fnv/
    for c in vec:
        d = ((d * 0x01000193) ^ (c * 17)) & 0xffffffff
    return int(d & bitmask) + 1


def fast_hash_2(dat_1, dat_2, bitmask):
    return int((((0x01000193 ^ (dat_1 * 17)) & 0xffffffff) ^ (dat_2 * 19)) & bitmask) + 1


def fast_hash_3(dat_1, dat_2, dat_3, bitmask):
    return int((((((0x01000193 ^ (dat_1 * 17)) & 0xffffffff) ^ (dat_2 * 19)) & 0xffffffff) ^ (dat_3 * 23)) & bitmask) + 1


def fast_hash_4(dat_1, dat_2, dat_3, dat_4, bitmask):
    return int((((((((0x01000193 ^ (dat_1 * 17)) & 0xffffffff) ^ (dat_2 * 19)) & 0xffffffff) ^ (dat_3 * 23)) & 0xffffffff) ^ (dat_4 * 29)) & bitmask) + 1


def fast_hash_vec(vec, bitmask):
    hash_vec = [0] * len(vec)
    d = 0x01000193
    # Use the FNV algorithm from http://isthe.com/chongo/tech/comp/fnv/
    for i, c in enumerate(vec):
        d = ((d * 0x01000193) ^ (c * 17)) & 0xffffffff
        hash_vec[i] = int(d & bitmask) + 1
    return hash_vec


def fast_hash_vec_char(vec, bitmask):
    hash_vec = [0] * len(vec)
    d = 0x01000193
    # Use the FNV algorithm from http://isthe.com/chongo/tech/comp/fnv/
    for i, c in enumerate(vec):
        d = ((d * 0x01000193) ^ (ord(c) * 17)) & 0xffffffff
        hash_vec[i] = int(d & bitmask) + 1
    return hash_vec


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)
