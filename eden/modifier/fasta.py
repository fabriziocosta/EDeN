import random
import re
from eden import util


def null_modifier(header=None, seq=None, **options):
    yield header
    yield seq


def fasta_to_fasta(input, modifier=null_modifier, **options):
    """
    Takes a FASTA file yields a normalised FASTA file.

    Parameters
    ----------
    input : string
        A pointer to the data source.

    normalize : bool
        If True all characters are uppercased and Ts are replaced by Us
    """
    normalize = options.get('normalize', True)
    iterable = _fasta_to_fasta(input)
    for line in iterable:
        header = line
        seq = iterable.next()
        if normalize:
            seq = seq.upper()
            seq = seq.replace('T', 'U')
        seqs = modifier(header=header, seq=seq, **options)
        for seq in seqs:
            yield seq


def _fasta_to_fasta(input):
    seq = ""
    for line in util.read(input):
        if line:
            if line[0] == '>':
                line = line[1:]
                if seq:
                    yield seq
                    seq = ""
                line_str = str(line)
                yield line_str.strip()
            else:
                line_str = line.split()
                if line_str:
                    seq += str(line_str[0]).strip()
    if seq:
        yield seq


def one_line_modifier(header=None, seq=None, **options):
    header_only = options.get('header_only', False)
    sequence_only = options.get('sequence_only', False)
    one_line = options.get('one_line', False)
    one_line_separator = options.get('one_line_separator', '\t')
    if one_line:
        yield header + one_line_separator + seq
    elif header_only:
        yield header
    elif sequence_only:
        yield seq
    else:
        raise Exception('ERROR: One of the options must be active.')


def insert_landmark_modifier(header=None, seq=None, **options):
    landmark_relative_position = options.get('landmark_relative_position', 0.5)
    landmark_char = options.get('landmark_char', '@')
    pos = int(len(seq) * landmark_relative_position)
    seq_out = seq[:pos] + landmark_char + seq[pos:]
    yield header
    yield seq_out


def shuffle_modifier(header=None, seq=None, **options):
    times = options.get('times', 1)
    order = options.get('order', 1)
    for i in range(times):
        kmers = [seq[i:i + order] for i in range(0, len(seq), order)]
        random.shuffle(kmers)
        seq_out = ''.join(kmers)
        yield header
        yield seq_out


def remove_modifier(header=None, seq=None, **options):
    regex = options.get('regex', '?')
    m = re.search(regex, seq)
    if not m:
        yield header
        yield seq


def keep_modifier(header=None, seq=None, **options):
    regex = options.get('regex', '?')
    m = re.search(regex, seq)
    if m:
        yield header
        yield seq


def update_start_end(header=None, start=None, end=None):
    startend_regex = '^(>\w*) START: *(\w*) *END: *(\w*)'
    startend_m = re.search(startend_regex, header)
    if startend_m:
        end = start + int(startend_m.group(3))
        start = start + int(startend_m.group(2))
        header = startend_m.group(1)
    return '%s START: %0.9d END: %0.9d LEN: %d' % (header, start, end, end - start)


def split_modifier(header=None, seq=None, **options):
    step = options.get('step', 10)
    window = options.get('window', 100)
    seq_len = len(seq)
    if seq_len >= window:
        for start in range(0, seq_len, step):
            seq_out = seq[start: start + window]
            if len(seq_out) == window:
                end = int(start + len(seq_out))
                header_out = update_start_end(header=header, start=start, end=end)
                yield header_out
                yield seq_out


def split_window_modifier(header=None, seq=None, **options):
    regex = options.get('regex', '?')
    window = options.get('window', 0)
    window_left = options.get('window_left', 0)
    window_right = options.get('window_right', 0)

    if window != 0:
        pattern = "(.{%d})(%s)(.{%d})" % (window, regex, window)
    else:
        pattern = "(.{%d})(%s)(.{%d})" % (window_left, regex, window_right)
    for m in re.finditer(pattern, seq):
        if m:
            start = m.start()
            end = m.end()
            header_out = update_start_end(header=header, start=start, end=end)
            yield header_out
            yield m.group(0)


def split_regex_modifier(header=None, seq=None, **options):
    regex = options.get('regex', '?')
    for m in re.finditer(regex, seq):
        if m:
            start = m.start()
            end = m.end()
            header_out = update_start_end(header=header, start=start, end=end)
            yield header_out
            yield m.group(0)


def replace_modifier(header=None, seq=None, **options):
    regex = options.get('regex', '?')
    replacement = options.get('replacement', ' ')
    seq_out = re.sub(regex, replacement, seq)
    yield header
    yield seq_out


def random_sample_modifier(header=None, seq=None, **options):
    prob = options.get('prob', 0.1)
    chance = random.random()
    if chance <= prob:
        yield header
        yield seq
