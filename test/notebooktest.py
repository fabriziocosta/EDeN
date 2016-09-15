import os


def test_notebooks():
    notebooks = ['Nearest_Neighbors_and_Gram_Matrix.ipynb']
    for notebook in notebooks:
        cmd = 'wget https://raw.githubusercontent.com/fabriziocosta/EDeN_examples/master/%s' % notebook
        os.system(cmd)
        cmd = 'jupyter nbconvert --to=html --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=300 %s' % notebook
        res = os.system(cmd)
        os.system('rm -f %s' % notebook)

        assert res == 0
