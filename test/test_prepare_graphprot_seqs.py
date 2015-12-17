# from re import search
from scripttest import TestFileEnvironment

bindir = "graphprot/"
script = "prepare_graphprot_seqs.py"
# test file environment
testdir = "test/testenv_prepare_graphprot_seqs"
# directories relative to test file environment
bindir_rel = "../../" + bindir
datadir_rel = "../../test/"

env = TestFileEnvironment(testdir)


def test_no_such_genome_id():
    """Try using an unknown genome id for automatic chromosome limits extraction."""
    run = env.run(
        bindir_rel + script,
        datadir_rel + "artificial_coords.bed",
        "unknown_genome",
        datadir_rel + "artificial_genome.fa",
        expect_error=True
    )
    # assert search("chromosome_limits", run.stdout), "Error message did not contain reference to 'chromosome_limits', was :'\n{}'".format(run.stdout)
    assert(run.returncode != 0)
