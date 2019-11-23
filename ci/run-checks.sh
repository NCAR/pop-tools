#!/bin/bash

set -e
set -eo pipefail

source activate ${ENV_NAME}
pytest -n 4 --junitxml=test-reports/junit.xml --cov=./ --verbose
echo "[Upload coverage]"
codecov
