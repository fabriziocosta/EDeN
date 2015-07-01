from collections import defaultdict
import numpy as np


def find_smallest_positive(alist):
    # find first positive value
    minpos = -1
    for x in alist:
        if x > 0:
            minpos = x
            break
    if minpos > 0:
        # find smallest positive value
        for x in alist:
            if x > 0 and x < minpos:
                minpos = x
    return minpos


def rebase_to_smallest_positive(alist):
    base = find_smallest_positive(alist)
    if base == -1:
        return None
    else:
        return [x - base for x in alist]


def compute_maximum_subarray(score_vector=None):
    begin_temp = begin = end = 0
    start_val = score_vector[0]
    max_ending_here = max_so_far = start_val
    for pos, x in enumerate(score_vector[1:], 1):
        if max_ending_here < 0:
            max_ending_here = x
            begin_temp = pos
        else:
            max_ending_here = max_ending_here + x
        if max_ending_here > max_so_far:
            max_so_far = max_ending_here
            begin = begin_temp
            end = pos
    return begin, end


def compute_iterated_maximum_subarray(seq=None, score=None, min_subarray_size=None, max_subarray_size=None, output='minimal', margin=1):
    original_score = score
    while True:
        # find (begin,end) of subarray in each element
        begin, end = compute_maximum_subarray(score_vector=score)
        # check that the retrieved subarray is larger than min_subarray_size
        if end - begin < min_subarray_size - 1:
            break
        else:
            # extract maximum subarray
            # NOTE: in order to account for border effects we expand on the left and on the right by 'margin'
            first = max(0, begin - margin)
            # NOTE: we return + 1 for the rightmost postition to be compliant with the 'one after the end' semantics
            last = min(len(seq), end + margin + 1)
            subarray = seq[first: last]
            subarray_size = len(subarray)
            if max_subarray_size == -1 or subarray_size <= max_subarray_size:
                # store data
                acc = 0
                for x in original_score[begin: end + 1]:
                    acc += x
                if output == 'minimal':
                    subarray = {'subarray_string': ''.join(subarray)}
                else:
                    subarray = {'subarray_string': ''.join(subarray), 'subarray': subarray, 'begin': first,
                                'end': last, 'size': subarray_size, 'seq': seq, 'score': acc}
                yield subarray
            if subarray_size > max_subarray_size:
                # if the subarray is too large then rebase the score list, i.e. offset by the smallest positive value
                score = rebase_to_smallest_positive(score)
                if score is None:
                    break
            else:
                # remove current subarray by zeroing importance values of subarray
                score[first: last] = [0.0] * subarray_size
                # iterate after removal of current subarray


def extract_sequence_and_score(graph=None):
    # make dict with positions as keys and lists of ids as values
    pos_to_ids = defaultdict(list)
    for u in graph.nodes():
        if 'position' not in graph.node[u]:  # no position attributes in graph, use the vertex id instead
            raise Exception('Missing "position" attribute in node:%s %s' % (u, graph.node[u]))
        else:
            pos = graph.node[u]['position']
        # accumulate all node ids
        pos_to_ids[pos] += [u]

    # extract sequence of labels and importances
    seq = [None] * len(pos_to_ids)
    score = [0] * len(pos_to_ids)
    for pos in sorted(pos_to_ids):
        ids = pos_to_ids[pos]
        labels = [graph.node[u].get('label', 'N/A') for u in ids]
        # check that all labels for the same position are identical
        assert(sum([1 for label in labels if label == labels[0]]) == len(labels)
               ), 'ERROR: non identical labels referring to same position: %s  %s' % (pos, labels)
        seq[pos] = labels[0]
        # average all importance score for the same position
        importances = [graph.node[u].get('importance', 0) for u in ids]
        score[pos] = np.mean(importances)
    return seq, score


def compute_max_subarrays_sequence(seq=None, score=None, min_subarray_size=None, max_subarray_size=None, output='minimal', margin=1):
    # extract subarrays
    for subarray in compute_iterated_maximum_subarray(seq=seq, score=score, min_subarray_size=min_subarray_size, max_subarray_size=max_subarray_size, output=output, margin=margin):
        yield subarray


def compute_max_subarrays(graph=None, min_subarray_size=None, max_subarray_size=None, output='minimal', margin=1):
    seq, score = extract_sequence_and_score(graph)
    for subarray in compute_max_subarrays_sequence(seq=seq, score=score, min_subarray_size=min_subarray_size, max_subarray_size=max_subarray_size, output=output, margin=margin):
        yield subarray
