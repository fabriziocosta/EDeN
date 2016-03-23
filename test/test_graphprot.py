from scripttest import TestFileEnvironment
import re
from filecmp import cmp

bindir = "graphprot/"
script = "graphprot_seqmodel"
# test file environment
datadir = "test/"
testdir = "test/testenv_graphprot_seqmodel/"
# directories relative to test file environment
bindir_rel = "../../" + bindir
datadir_rel = "../../" + datadir

env = TestFileEnvironment(testdir)


def test_invocation_no_params():
    "Call without parameters should return usage information."
    call = bindir_rel + script
    run = env.run(
        call,
        expect_error=True)
    assert run.returncode == 2
    assert re.match("usage", run.stderr), "stderr should contain usage information: {}".format(run.stderr)


def test_simple_fit():
    "Train a model on 10 positive and 10 negative sequences using default paramters."
    outfile = "test_simple_fit.model"
    call = bindir_rel + script + " fit -p {} -n {} --output-dir ./ --model-file {} --n-iter 1".format(
        datadir_rel + "rndseqs_10_a.fa",
        datadir_rel + "rndseqs_10_b.fa",
        outfile
    )
    env.run(call)
    assert cmp(testdir + outfile, datadir + outfile), "Error: trained models don't match."
