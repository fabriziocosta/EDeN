from eden.modifier.FASTA import FASTA

def FASTA_insert_landmark_to_FASTA(input = None, input_type = None, **options):
    landmark_relative_position = options.get('landmark_relative_position',0.5)
    landmark_char =  options.get('landmark_char','@')
    lines = FASTA.FASTA_to_FASTA(input = input, input_type = input_type)
    for line in lines:
        header = line
        seq = lines.next()
        pos = int( len(seq) * landmark_relative_position )
        seq_mod = seq[:pos] + landmark_char + seq[pos:]
        yield header
        yield seq_mod