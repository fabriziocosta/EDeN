from eden import util


def null_modifier(header=None, seq=None, const=None, **options):
    yield header
    yield seq
    yield const


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
        const = iterable.next()
        if normalize:
            seq = seq.upper()
            seq = seq.replace('T', 'U')
        seqs = modifier(header=header, seq=seq, const=const, **options)
        for seq in seqs:
            yield seq


def _fasta_to_fasta(input):
    header = ""
    seq = ""
    const = ""
    for line in util.read(input):
        line = str(line).strip()
        if line == "":
            # assume the empty line indicates that next line describes the constraints
            if seq:
                yield seq
            seq = None
        elif line[0] == '>':
            if const:
                yield const
                header = ""
                seq = ""
                const = ""
            header = line
            yield header
        else:
            # remove trailing chars, split and take only first part, removing comments
            line_str = line.split()[0]
            if line_str:
                if seq is None:
                    const += line_str
                else:
                    seq += line_str
    if const:
        yield const
