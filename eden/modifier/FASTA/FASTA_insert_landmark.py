import random
from eden.modifier.FASTA import FASTA

def FASTA_insert_landmark_to_FASTA(input = None, input_type = None, **options):
    return _FASTA_insert_landmark_to_FASTA(FASTA.FASTA_to_FASTA(input=input,input_type=input_type), **options)        


def _FASTA_insert_landmark_to_FASTA(data_str_list, **options):
    landmark_relative_position = options.get('landmark_relative_position',0.5)
    landmark_char =  options.get('landmark_char','@')
    for line in data_str_list:
        header_str = line
        line_buffer = data_str_list.next()
        pos = int( len(line_buffer) * landmark_relative_position )
        seq = line_buffer[:pos] + landmark_char + line_buffer[pos:]
        yield '>' + header_str
        yield seq