import random
from eden.modifier.FASTA import FASTA

def FASTA_shuffle_to_FASTA(input = None, input_type = None,  **options):
    lines = FASTA.FASTA_to_FASTA(input = input, input_type = input_type)
    for line in lines:
        header = line
        seq = lines.next()
        seq_mod = [c for c in seq]
        random.shuffle(seq_mod)
        seq_shuffled = ''.join(seq_mod)        
        yield header
        yield seq_shuffled