#!/bin/bash

# Fresh install with "rm .micromamba/micromamba .done-*"

set -e

nj=$(nproc)
home=$PWD
venv_dir=$PWD/venv
source ./env.sh

mark=.done-asrbn-requirements
if [ ! -f $mark ]; then
  echo " == Installing ASRBN python libraries =="

  pip3 install -r anonymization/pipelines/asrbn/requirements.txt  || exit 1
  touch $mark
fi

