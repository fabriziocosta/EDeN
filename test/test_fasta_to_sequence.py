from eden.converter.fasta import fasta_to_sequence
from eden.util import is_iterable


class TestFastaToSequence:

    def test_fasta_to_sequence_default(self):
        """Test test_fasta_to_sequence with default parameters."""

        fa_fn = "test/test_fasta_to_sequence.fa"
        seq = fasta_to_sequence(fa_fn)
        assert(is_iterable(seq))
        (header, sequence) = seq.next()
        # header should contain the fasta header with '>' removed
        assert(header == "ID0")
        # sequence should be uppercased and all Ts should be replaced by Us
        assert(sequence == "GUGGCGUACUCACGGCCACCUUAGGACUCCGCGGACUUUAUGCCCACCAAAAAAACGAGCCGUUUCUACGCGUCCUCCGUCGCCUGUGUCGAUAAAGCAA")

    def test_fasta_to_sequence_normalized(self):
        """Test default test_fasta_to_sequence with default parameters."""

        fa_fn = "test/test_fasta_to_sequence.fa"
        seq = fasta_to_sequence(fa_fn, normalize=True)
        assert(is_iterable(seq))
        (header, sequence) = seq.next()
        # sequence should be uppercased and all Ts should be replaced by Us
        assert(sequence == "GUGGCGUACUCACGGCCACCUUAGGACUCCGCGGACUUUAUGCCCACCAAAAAAACGAGCCGUUUCUACGCGUCCUCCGUCGCCUGUGUCGAUAAAGCAA")

    def test_fasta_to_sequence_no_normalize(self):
        """Test default test_fasta_to_sequence with default parameters."""

        fa_fn = "test/test_fasta_to_sequence.fa"
        seq = fasta_to_sequence(fa_fn, normalize=False)
        assert(is_iterable(seq))
        (header, sequence) = seq.next()
        # sequence should correspond to the unmodified fasta string
        assert(sequence == "gtggcgtactcacggccaCCTTAGGACTCCGCGGACTTTATGCCCACCAAAAAAACGAGCCGTTTCTACGCGTCCTCCGTCGCCTgtgtcgataaagcaa")
