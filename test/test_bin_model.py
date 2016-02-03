import re
# from filecmp import cmp
from scripttest import TestFileEnvironment

testdir = "test/"
bindir = "bin/"
datadir = "test/data/"
testenv = testdir + 'testenv_bin_model/'

env = TestFileEnvironment(testenv)
# relative to test file environment: testenv
bindir_rel = "../../" + bindir
datadir_rel = "../../" + datadir


def test_call_without_parameters():
    "Call bin/model without any additional parameters."
    call_script = bindir_rel + 'model'
    run = env.run(*call_script.split(),
                  expect_error=True
                  )
    assert re.search("too few arguments", run.stderr), 'expecting "too few arguments" in stderr'


# def test_fit():
#     "Call bin/model in fit mode."
#     call_script = bindir_rel + 'model -p xx -n xx'
#     run = env.run(*call_script.split())
#     assert False, str(run.files_created)
