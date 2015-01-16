from eden.modifier.FASTA import FASTA

def FASTA_remove_to_FASTA(input = None, input_type = None, **options):
    remove_char =  options.get('remove_char','-')
    lines = FASTA.FASTA_to_FASTA(input = input, input_type = input_type)
    for line in lines:
        header = line
        seq = lines.next()
        if not remove_char in seq:
            yield '>' + header
            yield seq