from eden.modifier.fasta_with_constraints import fasta_to_fasta


def fasta_to_sequence(input, **options):
    lines = fasta_to_fasta(input)
    for line in lines:
        header = line
        seq = lines.next()
        const = lines.next()
        yield header, seq, const
