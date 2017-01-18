import os


def test_notebooks():
    notebooks = ['sequence_example.ipynb']
    for notebook in notebooks:
        cmd = 'wget -q https://raw.githubusercontent.com/fabriziocosta/EDeN_examples/master/%s' % notebook
        os.system(cmd)
        cmd = 'jupyter nbconvert  --stdout --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=300 %s > /dev/null' % notebook
        res = os.system(cmd)
        os.system('rm -f %s' % notebook)
        assert res == 0
