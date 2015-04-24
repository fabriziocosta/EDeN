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


def serial_vectorize(graphs, vectorizer=None):
    X = vectorizer.transform(graphs)
    return X


def multiprocess_vectorize(graphs, vectorizer=None, n_blocks=5, n_jobs=8):
    graphs = list(graphs)
    import multiprocessing as mp
    size = len(graphs)
    #if n_blocks is the same or larger than size then decrease n_blocks so to have at least 10 instances per block 
    if n_blocks >= size:
        n_blocks = size / 10
    #if one block will end up containing a single instance reduce the number of blocks to avoid the case
    if size % n_blocks == 1:
        n_blocks = max(1, n_blocks - 1)
    block_size = size / n_blocks
    reminder = size % n_blocks
    if n_jobs == -1:
        pool = mp.Pool()
    else:
        pool = mp.Pool(n_jobs)
    intervals = [(s * block_size, (s + 1) * block_size) for s in range(n_blocks)]
    if reminder > 1:
        intervals += [(n_blocks * block_size, n_blocks * block_size + reminder)]
    results = [apply_async(pool, serial_vectorize, args=(graphs[start:end], vectorizer)) for start, end in intervals]
    output = [p.get() for p in results]
    pool.close()
    pool.join()
    import numpy as np
    from scipy.sparse import vstack
    X = output[0]
    for Xi in output[1:]:
        X = vstack([X, Xi], format="csr")
    return X


def vectorize(graphs, vectorizer=None, n_blocks=5, n_jobs=8):
    if n_jobs == 1:
        return serial_vectorize(graphs, vectorizer=vectorizer)
    else:
        return multiprocess_vectorize(graphs, vectorizer=vectorizer, n_blocks=n_blocks, n_jobs=n_jobs)


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
        hash_vec += [int(running_hash & bitmask) + 1]
    return hash_vec


def fast_hash_vec_char(vec, bitmask):
    hash_vec = []
    running_hash = 0xAAAAAAAA
    for i, list_item_char in enumerate(vec):
        list_item = ord(list_item_char)
        running_hash ^= calc_running_hash(running_hash, list_item, i)
        hash_vec += [int(running_hash & bitmask) + 1]
    return hash_vec


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(*args, fillvalue=fillvalue)
