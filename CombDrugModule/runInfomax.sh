#!/bin/bash

# Activate conda environment
ANACONDA_PATH=$(dirname "$(dirname `which conda`)")
source ${ANACONDA_PATH}/etc/profile.d/conda.sh
conda activate infomax

MOD_DIR=$(dirname "$(realpath $0)")

python ${MOD_DIR}/runInfomax.py $1
