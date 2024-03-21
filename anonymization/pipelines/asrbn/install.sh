#!/bin/bash

# Fresh install with "rm .micromamba/micromamba .done-*"

set -e

nj=$(nproc)
home=$PWD
venv_dir=$PWD/venv
source ./env.sh

compute_and_write_hash "anonymization/pipelines/asrbn/requirements.txt"  # SHA256: 83b977d3e6716e665c5673810d0b39a1bbc422580c0ce6372da4431c04e8cfa4
trigger_new_install ".done-asrbn-requirements"

mark=.done-asrbn-requirements
if [ ! -f $mark ]; then
  echo " == Installing ASRBN python libraries =="
  pip3 install -r anonymization/pipelines/asrbn/requirements.txt  || exit 1
  touch $mark
fi
