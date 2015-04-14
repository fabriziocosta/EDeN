__author__ = "Fabrizio Costa, Bjoern Gruening"
__copyright__ = "Copyright 2014, Fabrizio Costa"
__credits__ = ["Fabrizio Costa", "Bjoern Gruening"]
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "Fabrizio Costa"
__email__ = "costa@informatik.uni-freiburg.de"
__status__ = "Production"


from itertools import izip_longest


def serial_vectorize(graphs, vectorizer=None):
    X = vectorizer.transform(graphs)
    return X


def multiprocess_vectorize(graphs, vectorizer=None, negative_ratio=2, n_blocks=5, n_processes=8):
    graphs = list(graphs)
    import multiprocessing as mp
    size = len(graphs)
    block_size = size / n_blocks
    pool = mp.Pool(processes=n_processes)
    results = [pool.apply_async(serial_vectorize, args=(graphs[s * block_size:(s + 1) * block_size], vectorizer)) for s in range(n_blocks - 1)]
    output = [p.get() for p in results]
    import numpy as np
    from scipy.sparse import vstack
    X = output[0]
    for Xi in output[1:]:
        X = vstack([X, Xi], format="csr")
    return X


def vectorize(graphs, vectorizer=None, multiprocess=True, n_blocks=5, n_processes=8):
    if multiprocess:
        return multiprocess_vectorize(graphs, vectorizer=vectorizer, n_blocks=n_blocks, n_processes=n_processes)
    else:
        return serial_vectorize(graphs, vectorizer=vectorizer)


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
