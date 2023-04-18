#!/bin/bash

# Activate conda environment
ANACONDA_PATH=$(dirname "$(dirname `which conda`)")
source ${ANACONDA_PATH}/etc/profile.d/conda.sh
conda activate e3fp_env

MOD_DIR=$(dirname "$(realpath $0)")

python ${MOD_DIR}/runE3FP.py $1 $2
