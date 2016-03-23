from scripttest import TestFileEnvironment
import re

bindir = "graphprot/"
script = "graphprot_seqmodel"
# test file environment
testdir = "test/testenv_graphprot_seqmodel"
# directories relative to test file environment
bindir_rel = "../../" + bindir
datadir_rel = "../../test/"

env = TestFileEnvironment(testdir)


def test_invocation_no_params():
    "Call without parameters should return usage information."
    call = bindir_rel + script
    run = env.run(
        call,
        expect_error=True)
    assert run.returncode == 2
    assert re.match("usage", run.stderr), "stderr should contain usage information: {}".format(run.stderr)
