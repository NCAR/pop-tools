#!/bin/bash

set -e
set -eo pipefail

echo "Code Styling with (black, flake8, isort)"

source activate ${ENV_NAME}

echo "[flake8]"
flake8 pop_tools

echo "[black]"
black --check --line-length=100 -S pop_tools

echo "[isort]"
isort --recursive -w 100 --check-only pop_tools
