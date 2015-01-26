import dill
from itertools import *

def run_dill_encoded(what):
    """
    Use dill as replacement for pickle to enable multiprocessing on instance methods
    """
    fun, args = dill.loads(what)
    return fun(*args)


def apply_async(pool, fun, args, callback):
    """
    Wrapper around apply_async() from multiprocessing, to use dill instead of pickle.
    This is a workaround to enable multiprocessing of classes.
    """
    return pool.apply_async(run_dill_encoded, (dill.dumps((fun, args)),), callback = callback)


def calc_running_hash( running_hash, list_item, counter ):
    return ((~(((running_hash << 11) + list_item) ^ (running_hash >> 5))),((running_hash << 7) ^ list_item * (running_hash >> 3)))[bool((counter & 1) == 0)]


def fast_hash( vec, bitmask ):
    running_hash = 0xAAAAAAAA
    for i, list_item in enumerate(vec):
        running_hash  ^= calc_running_hash( running_hash, list_item, i )
    return int(running_hash & bitmask) + 1


def fast_hash_vec( vec, bitmask ):
    hash_vec=[]
    running_hash = 0xAAAAAAAA
    for i, list_item in enumerate(vec):
        running_hash  ^= calc_running_hash( running_hash, list_item, i )
        hash_vec += [int(running_hash & bitmask) + 1]
    return hash_vec


def fast_hash_vec_char( vec, bitmask ):
    hash_vec=[]
    running_hash = 0xAAAAAAAA
    for i, list_item_char in enumerate(vec):
        list_item = ord(list_item_char)
        running_hash  ^= calc_running_hash( running_hash, list_item, i )
        hash_vec += [int(running_hash & bitmask) + 1]
    return hash_vec


def report_base_statistics(vec):
    from collections import Counter
    c =Counter(vec)
    msg = ''
    for k in c:
        msg += "class: %s count:%d (%0.2f)\t"% (k, c[k], c[k]/float(len(vec)))
    return msg

def grouper(iterable, n, fillvalue = None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)