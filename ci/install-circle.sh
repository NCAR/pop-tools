#!/bin/bash

set -e
set -eo pipefail

conda config --set always_yes true --set changeps1 false --set quiet true
conda config --add channels conda-forge
conda create --name=${ENV_NAME} python=${PYTHON} --quiet
conda env update -f ci/environment.yml
source activate ${ENV_NAME}
python -m pip install --no-deps --quiet -e .
conda list -n ${ENV_NAME}
