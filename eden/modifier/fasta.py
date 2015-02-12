import random
import re
from eden import util

def null_modifier(header = None, seq = None, **options):
    yield header
    yield seq


def fasta_to_fasta(input, modifier = null_modifier, **options):
    """
    Takes a FASTA file yields a normalised FASTA file.

    Parameters
    ----------
    input : string
        A pointer to the data source.
    """
    lines = _fasta_to_fasta(input)
    for line in lines:
        header_in = line
        seq_in = lines.next()
        seqs = modifier(header = header_in, seq = seq_in, **options)
        for seq in seqs:
            yield seq


def _fasta_to_fasta( input ):
    seq = ''
    for line in util.read( input ):
        _line = line.strip()
        if _line:
            if _line[0] == '>':
                #extract string from header
                header = _line 
                if seq:
                    yield prev_header
                    yield seq
                seq = ''
                prev_header = header
            else:
                seq += _line
    if seq:
        yield prev_header
        yield seq


def one_line_modifier(header = None, seq = None, **options):
    header_only = options.get('header_only',False)
    sequence_only = options.get('sequence_only',False)
    one_line = options.get('one_line',False)
    one_line_separator = options.get('one_line_separator','\t')
    if one_line:
        yield  header + one_line_separator + seq
    elif header_only:
        yield header
    elif sequence_only:
        yield seq
    else:
        raise Exception('ERROR: One of the options must be active.')    


def insert_landmark_modifier(header = None, seq = None, **options):
    landmark_relative_position = options.get('landmark_relative_position',0.5)
    landmark_char =  options.get('landmark_char','@')
    pos = int( len(seq) * landmark_relative_position )
    seq_out = seq[:pos] + landmark_char + seq[pos:]
    yield header
    yield seq_out


def shuffle_modifier(header = None, seq = None, **options):
    times =  options.get('times',1)
    order =  options.get('order',1)
    for i in range(times):
        kmers = [ seq[i:i+order] for i in range(0,len(seq),order) ]
        random.shuffle(kmers)
        seq_out = ''.join(kmers)
        yield header
        yield seq_out


def remove_modifier(header = None, seq = None, **options):
    regex =  options.get('regex','?')
    m = re.find(regex, seq)
    if not m:
        yield header
        yield seq
            

def keep_modifier(header = None, seq = None, **options):
    regex =  options.get('regex','?')
    m = re.search(regex, seq)
    if m:
        yield header
        yield seq


def split_modifier(header = None, seq = None, **options):
    step =  options.get('step',10)
    window =  options.get('window',100)
    seq_len = len(seq)
    if seq_len >= window:
        for start in range(0, seq_len, step):
            seq_out = seq[start : start + window]
            if len(seq_out) == window:
                yield '%s START: %0.9d END: %0.9d' % (header, start, int(start + len(seq_out)))
                yield seq_out


def split_window_modifier(header = None, seq = None, **options):
    regex =  options.get('regex','?')
    window =  options.get('window',0)
    window_left =  options.get('window_left',0)
    window_right =  options.get('window_right',0)
    
    if window != 0:
        pattern = "(.{%d})(%s)(.{%d})"%(window, regex, window)
    else:
        pattern = "(.{%d})(%s)(.{%d})"%(window_left, regex, window_right)
    for m in re.finditer(pattern, seq):
        if m:
            start = m.start()
            end = m.end()
            startend_regex = 'START: *(\w*) *END: *(\w*)'
            startend_m = re.search(startend_regex,header)
            if startend_m:
                end = start + startend_m.group(2)
                start = start + startend_m.group(1)
            yield '%s START: %0.9d END: %0.9d' % (header, start, end)
            yield m.group(0)


def split_regex_modifier(header = None, seq = None, **options):
    regex =  options.get('regex','?')
    for m in re.finditer(regex, seq):
        if m:
            start = m.start()
            end = m.end()
            startend_regex = 'START: *(\w*) *END: *(\w*)'
            startend_m = re.search(startend_regex,header)
            if startend_m:
                end = start + startend_m.group(2)
                start = start + startend_m.group(1)
            yield '%s START: %0.9d END: %0.9d' % (header, start, end)
            yield m.group(0)


def replace_modifier(header = None, seq = None, **options):
    regex =  options.get('regex','?')
    replacement =  options.get('replacement',' ')
    seq_out = re.sub(regex, replacement, seq)
    yield header
    yield seq_out


def random_sample_modifier(header = None, seq = None, **options):
    prob =  options.get('prob',0.1)
    chance = random.random()
    if chance <= prob:
        yield header
        yield seq