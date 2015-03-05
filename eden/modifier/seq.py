import random
import re
from eden import util


def null_modifier(header=None, seq=None, **options):
    yield header, seq


def seq_to_seq(iterable, modifier=null_modifier, **options):
    """
    Takes in input a list of pairs of header, sequence
    """
    for header, seq in iterable:
        for seq in modifier(header=header, seq=seq, **options):
            yield seq


def mark_modifier(header=None, seq=None, **options):
    mark_pos = options.get('position', 0.5)
    mark = options.get('mark', '@')
    pos = int(len(seq) * mark_pos)
    seq_out = seq[:pos] + mark + seq[pos:]
    yield header, seq_out


def shuffle_modifier(header=None, seq=None, **options):
    times = options.get('times', 1)
    order = options.get('order', 1)
    for i in range(times):
        kmers = [seq[i:i + order] for i in range(0, len(seq), order)]
        random.shuffle(kmers)
        seq_out = ''.join(kmers)
        yield header, seq_out
