from graphprot import testfun


def test_testfun():
    testfun("hello world!")


from eden.converter.fasta import fasta_to_sequence
from eden.converter.fasta import sequence_to_eden


def test_fasta_to_sequence_graph():
    fa_fn = "test/test_fasta_to_sequence.fa"
    seq = fasta_to_sequence(fa_fn)
    gra = sequence_to_eden(seq)
