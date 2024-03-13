#!/bin/bash

# Fresh install with "rm .micromamba/micromamba .done-*"

set -e

nj=$(nproc)
home=$PWD
venv_dir=$PWD/venv
source ./env.sh

hash_check=".install-hash-ASRBN"
stored_hash=$(echo $(cat $hash_check 2> /dev/null) || echo "empty")
current_hash=$(sha256sum "$0" | awk '{print $1}')
if [ "$current_hash" != "$stored_hash" ]; then
  echo "ASRBN install script has been modified. Triggering new installation..."
  \rm -f .done-asrbn-requirements || true
  echo "$current_hash" > $hash_check
fi

mark=.done-asrbn-requirements
if [ ! -f $mark ]; then
  echo " == Installing ASRBN python libraries =="
  pip3 install -r anonymization/pipelines/asrbn/requirements.txt  || exit 1
  touch $mark
fi
