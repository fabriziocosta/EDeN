from scripttest import TestFileEnvironment
import re
# from filecmp import cmp

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


def test_invocation_nonexisting_input():
    "Call with nonexisting input file."
    outfile = "shouldcrash"
    call = bindir_rel + script + " -vvv fit -p {} -n {} --output-dir ./ --model-file {} --n-iter 1".format(
        datadir_rel + "does_not_exist",
        datadir_rel + "does_not_exist",
        outfile,
    )
    run = env.run(
        call,
        expect_error=True,
    )
    assert run.returncode != 0


def test_fit_optimization_fail():
    outfile = "test_simple_fit.model"
    call = bindir_rel + script + " -vvv fit -p {} -n {} --output-dir ./ --model-file {} --n-iter 2 --n-inner-iter-estimator 2".format(
        datadir_rel + "does_not_exist",
        datadir_rel + "does_not_exist",
        outfile,
    )
    # graphprot/graphprot_seqmodel -vvv fit -p test/graphprot_seqmodel_test_fit_no_solution.fa -n test/graphprot_seqmodel_test_fit_no_solution.fa --output-dir manualtest --n-iter 2 --n-inner-iter-estimator 2
    run = env.run(
        call,
        expect_error=True
    )
    # script should give non-zero return code
    assert run.returncode != 0
    # script should not create any files
    assert len(run.files_created.keys()) == 0


def test_simple_fit():
    "Train a model on 10 positive and 10 negative sequences using default paramters."
    outfile = "test_simple_fit.model"
    call = bindir_rel + script + " -vvv fit -p {} -n {} --output-dir ./ --model-file {} --n-iter 1".format(
        datadir_rel + "PARCLIP_MOV10_Sievers_10seqs.train.positives.fa",
        datadir_rel + "PARCLIP_MOV10_Sievers_10seqs.train.negatives.fa",
        outfile
    )
    # ../../graphprot/graphprot_seqmodel -vvv fit -p ../../test/PARCLIP_MOV10_Sievers_100seqs.train.positives.fa -n ../../test/PARCLIP_MOV10_Sievers_100seqs.train.negatives.fa --output-dir ./ --model-file test_simple_fit.model --n-iter 1
    env.run(call)
    call = bindir_rel + script + " -vvv estimate -p {} -n {} --output-dir ./ --model-file {} --cross-validation".format(
        datadir_rel + "PARCLIP_MOV10_Sievers_10seqs.train.positives.fa",
        datadir_rel + "PARCLIP_MOV10_Sievers_10seqs.train.negatives.fa",
        outfile
    )
    # ../../graphprot/graphprot_seqmodel -vvv estimate -p ../../test/PARCLIP_MOV10_Sievers_1kseqs.train.positives.fa -n ../../test/PARCLIP_MOV10_Sievers_1kseqs.train.negatives.fa --output-dir ./ --model-file test_simple_fit.model --cross-validation
    run = env.run(
        call,
        expect_stderr=True,
    )
    stdout = open(testdir + outfile + ".cv.out", "w")
    stdout.write(run.stdout)


def test_predict_simpletask():
    "Fit model and do prediction of training data using default parameters."
    model = "test_predict_simpletask.model"
    call = bindir_rel + script + " -vvv fit -p {} -n {} --output-dir ./ --model-file {} --n-iter 1".format(
        datadir_rel + "simple_positives.fa",
        datadir_rel + "simple_negatives.fa",
        model,
    )
    env.run(call,)
    call = bindir_rel + script + " -vvv predict --input-file {} --model-file {} --output-dir {}".format(
        datadir_rel + "simple_positives.fa",
        model,
        "test_predict_simpletask",
    )
    run = env.run(call)
    assert "test_predict_simpletask/predictions.txt" in run.files_created
    for line in run.files_created["test_predict_simpletask/predictions.txt"].bytes.split("\n"):
        try:
            prediction, margin, id = line.split()
            assert float(margin) >= 0.4, "Error: all margins should be at leat 0.4, the margin for id {} is '{}' in {}.".format(id, margin, run.files_created["test_predict_simpletask/predictions.txt"].bytes)
        except ValueError:
            pass


def test_predict():
    "Predict class of some sequences."
    model = "test_predict.model"
    call = bindir_rel + script + " -vvv fit -p {} -n {} --output-dir ./ --model-file {} --n-iter 1".format(
        datadir_rel + "PARCLIP_MOV10_Sievers_10seqs.train.positives.fa",
        datadir_rel + "PARCLIP_MOV10_Sievers_10seqs.train.negatives.fa",
        model
    )
    # ../../graphprot/graphprot_seqmodel -vvv fit -p ../../test/PARCLIP_MOV10_Sievers_100seqs.train.positives.fa -n ../../test/PARCLIP_MOV10_Sievers_100seqs.train.negatives.fa --output-dir ./ --model-file test_simple_fit.model --n-iter 1
    env.run(call)
    call = bindir_rel + script + " -vvv predict --input-file {} --model-file {} --output-dir {}".format(
        datadir_rel + "PARCLIP_MOV10_Sievers_10seqs.train.positives.fa",
        model,
        "test_predict",
    )
    run = env.run(call)
    assert "test_predict/predictions.txt" in run.files_created


def test_priors_weight_fail_allzero():
    "Fit model reweighting by priors, set prior weight extra high to produce exclusively zero weights."
    # lowest prior is p=0.00031274442646757
    # weights w > 1/p are guaranteed to produce zero weights exclusively (-> 3.179)
    model = "test_priors_weight_fail_allzero.model"
    call = bindir_rel + script + " -vvv fit -p {} -n {} --output-dir ./ --model-file {} --n-iter 1 --kmer-probs {} --kmer-weight 3200".format(
        datadir_rel + "PARCLIP_MOV10_Sievers_10seqs.train.positives.fa",
        datadir_rel + "PARCLIP_MOV10_Sievers_10seqs.train.negatives.fa",
        model,
        datadir_rel + "test_graphprot_priors.txt",
    )
    run = env.run(
        call,
        expect_error=True)
    assert run.returncode != 0


def test_priors_weight():
    "Fit model reweighting by priors, set prior weight extra high to produce exclusively zero weights."
    model = "test_priors.model"
    call = bindir_rel + script + " -vvv fit -p {} -n {} --output-dir ./ --model-file {} --n-iter 1 --kmer-probs {}".format(
        datadir_rel + "PARCLIP_MOV10_Sievers_10seqs.train.positives.fa",
        datadir_rel + "PARCLIP_MOV10_Sievers_10seqs.train.negatives.fa",
        model,
        datadir_rel + "test_graphprot_priors.txt",
    )
    run = env.run(call,)
    assert model in run.files_created


def test_predictprofile():
    "Predict nucleotide-wise margins of some sequences."
    model = "test_predict_profile.model"
    call = bindir_rel + script + " -vvv fit -p {} -n {} --output-dir {} --model-file {} --n-iter 1".format(
        datadir_rel + "PARCLIP_MOV10_Sievers_10seqs.train.positives.fa",
        datadir_rel + "PARCLIP_MOV10_Sievers_10seqs.train.negatives.fa",
        "test_predict_profile",
        model
    )
    env.run(call)
    call = bindir_rel + script + " -vvv predict_profile --input-file {} --model-file {} --output-dir {}".format(
        datadir_rel + "PARCLIP_MOV10_Sievers_10seqs.train.positives.fa",
        "test_predict_profile/" + model,
        "test_predict_profile",
    )
    run = env.run(call)
    assert "test_predict_profile/profile.txt" in run.files_created


def test_predictprofile_with_priors():
    "Predict nucleotide-wise margins of some sequences."
    model = "test_predict_profile_with_priors.model"
    call = bindir_rel + script + " -vvv fit -p {} -n {} --output-dir {} --model-file {} --n-iter 1 --kmer-probs {} --kmer-weight 200".format(
        datadir_rel + "PARCLIP_MOV10_Sievers_10seqs.train.positives.fa",
        datadir_rel + "PARCLIP_MOV10_Sievers_10seqs.train.negatives.fa",
        "test_predict_profile_with_priors",
        model,
        datadir_rel + "test_graphprot_priors.txt",
    )
    env.run(call)
    call = bindir_rel + script + " -vvv predict_profile --input-file {} --model-file {} --output-dir {}".format(
        datadir_rel + "PARCLIP_MOV10_Sievers_10seqs.train.positives.fa",
        "test_predict_profile_with_priors/" + model,
        "test_predict_profile_with_priors",
    )
    run = env.run(call)
    assert "test_predict_profile_with_priors/profile.txt" in run.files_created
