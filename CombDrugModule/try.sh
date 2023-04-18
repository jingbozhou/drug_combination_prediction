#!/bin/bash

ANACONDA_PATH=$(dirname "$(dirname `which conda`)")

echo ${ANACONDA_PATH}/etc/profile.d/conda.sh
