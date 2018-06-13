# Only need to change these two variables
PKG_NAME=misc_scripts
USER=dmaticzka

OS=linux-64
mkdir ~/conda-bld
conda config --set anaconda_upload no
export CONDA_BLD_PATH=~/conda-bld
# update version from tags
python setup.py sdist
export VERSION=`python setup.py --version`_`date +%Y.%m.%d`
conda build . && \
anaconda \
-t $CONDA_UPLOAD_TOKEN upload \
-u $USER \
-l nightly \
--force \
$CONDA_BLD_PATH/$OS/$PKG_NAME-$VERSION-py`echo ${TRAVIS_PYTHON_VERSION} | tr -d '.'`_0.tar.bz2
