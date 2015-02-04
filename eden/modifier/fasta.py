import random
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
    remove_char =  options.get('remove_char','-')
    if not remove_char in seq:
        yield header
        yield seq
            

def keep_modifier(header = None, seq = None, **options):
    keep_char_list =  options.get('keep_char_list',['A','C','G','T','U'])
    skip = False
    for c in seq:
        if not c in keep_char_list:
            skip = True
            break
    if not skip:
        yield header
        yield seq


def split_modifier(header = None, seq = None, **options):
    step =  options.get('step',10)
    window =  options.get('window',100)
    seq_len = len(seq)

    if seq_len < window:
        yield '%s START: %0.9d WINDOW: %0.3d' % (header, 0, window)
        yield seq
    else :
        for start in range(0, seq_len, step):
            seq_out = seq[start : start + window]
            if len(seq_out) == window:
                yield '%s START: %0.9d WINDOW: %0.3d' % (header, start, window)
                yield seq_out


def split_N_modifier(header = None, seq = None, **options):
    min_length =  options.get('min_length',10)
    seq_curr = ''
    start = 0
    for i,c in enumerate(seq):
        if c != 'N':
            if c == 'T':
                l = 'U'
            else:
                l = c
            seq_curr += l
        else:
            if len(seq_curr) >= min_length:
                yield '%s START: %0.9d' % (header, start + 1 )
                yield seq_curr
            seq_curr = ''
            start = i


def random_sample_modifier(header = None, seq = None, **options):
    prob =  options.get('prob',0.1)
    chance = random.random()
    if chance <= prob:
        yield header
        yield seq