import random


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


def split_modifier(header=None, seq=None, **options):
    step = options.get('step', 10)
    window = options.get('window', 100)
    seq_len = len(seq)
    if seq_len >= window:
        for start in range(0, seq_len, step):
            seq_out = seq[start: start + window]
            if len(seq_out) == window:
                end = int(start + len(seq_out))
                header_out = '%s %d %d' % (header, start, end)
                yield (header_out, seq_out)
