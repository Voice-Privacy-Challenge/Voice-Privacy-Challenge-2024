#!/bin/bash

# Fresh install with "rm .micromamba/micromamba .done-*"

set -e

nj=$(nproc)
home=$PWD
venv_dir=$PWD/venv
source ./env.sh

compute_and_write_hash "anonymization/modules/nac/coqui_tts/requirements.txt"  # SHA256: 81d2267383004a6f3176cf92e65659e5e81845b78c0cc2d4709c6efb010d7500
trigger_new_install "exp/nac_mappings .done-nac-requirements"

# Download nac mappings
if [ ! -d exp/nac_mappings ]; then
    echo "Download pre-computed speaker mappings for the NAC pipeline"
    unzip_location="exp/nac_mappings"
    zip_file="./anonymization/modules/nac/speaker_mappings.zip"
    curl -q -L -o $zip_file https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2024/releases/download/nac_mappings.zip/nac_speaker_mappings.zip
    unzip $zip_file -d $unzip_location
    rm $zip_file
fi


mark=.done-nac-requirements
if [ ! -f $mark ]; then
  echo " == Installing NAC libraries =="
  coqui_tts_dir="./anonymization/modules/nac/coqui_tts"
  [ -d $coqui_tts_dir ] && yes | rm -rf $coqui_tts_dir
  git clone https://github.com/m-pana/nac-requirements.git $coqui_tts_dir
  pip install -e $coqui_tts_dir
  touch $mark
fi
