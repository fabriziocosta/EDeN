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


def fast_hash_2(dat_1, dat_2, bitmask):
    d = (~((7919 + dat_1) ^ 7919)) ^ (2999 ^ dat_2 * 2999)
    return int(d & bitmask) + 1


def fast_hash_4(dat_1, dat_2, dat_3, dat_4, bitmask):
    d = ((~((7919 + dat_1) ^ 7919)) ^ (2999 ^ dat_2 * 2999)) 
    d ^= (~((d  << 11 + dat_3) ^ (d >> 5))) ^ ((d << 7) ^ (dat_4 >> 3) * d)
    return int(d & bitmask) + 1


def calc_running_hash(running_hash, list_item, counter):
    return ((~(((running_hash << 11) + list_item) ^ (running_hash >> 5))), ((running_hash << 7) ^ list_item * (running_hash >> 3)))[bool((counter & 1) == 0)]


def fast_hash(vec, bitmask):
    running_hash = 0xAAAAAAAA
    for i, list_item in enumerate(vec):
        running_hash ^= calc_running_hash(running_hash, list_item, i)
    return int(running_hash & bitmask) + 1


def fast_hash_vec(vec, bitmask):
    hash_vec = []
    running_hash = 0xAAAAAAAA
    for i, list_item in enumerate(vec):
        running_hash ^= calc_running_hash(running_hash, list_item, i)
        hash_vec.append(int(running_hash & bitmask) + 1)
    return hash_vec


def fast_hash_vec_char(vec, bitmask):
    hash_vec = []
    running_hash = 0xAAAAAAAA
    for i, list_item_char in enumerate(vec):
        list_item = ord(list_item_char)
        running_hash ^= calc_running_hash(running_hash, list_item, i)
        hash_vec.append(int(running_hash & bitmask) + 1)
    return hash_vec


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)
