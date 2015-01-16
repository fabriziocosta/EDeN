import random
from eden.modifier.FASTA import FASTA

def FASTA_split_to_FASTA(input = None, input_type = None,  **options):
    step =  options.get('step',10)
    window =  options.get('window',100)
    lines = FASTA.FASTA_to_FASTA(input = input, input_type = input_type)
    for line in lines:
        header = line
        seq = lines.next()
        seq_len = len(seq)
        for start in range(0, seq_len, step):
            subseq = seq[start : start + window]
            yield '>%s START: %d WINDOW: %d' % (header, start, window)
            yield subseq