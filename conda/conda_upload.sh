# Only need to change these two variables
PKG_NAME=eden
USER=dmaticzka

OS=linux-64
mkdir ~/conda-bld
export CONDA_BLD_PATH=~/conda-bld
# update version from tags
python setup.py sdist
export VERSION=`python setup.py --version | grep --only-matching '^[0-9\.]\+'`_`date +%Y.%m.%d`
conda config --set anaconda_upload yes
conda config --set force yes
conda build . \
--user $USER \
--token $CONDA_UPLOAD_TOKEN
