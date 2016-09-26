from eden.modifier.fasta_with_constraints import fasta_to_fasta


def fasta_to_sequence(input, **options):
    lines = fasta_to_fasta(input)
    for line in lines:
        header = line
        seq = next(lines)
        constr = next(lines)
        if len(seq) == 0 or len(constr) == 0:
            raise Exception('ERROR: empty sequence or constraint')
        yield header, seq, constr
