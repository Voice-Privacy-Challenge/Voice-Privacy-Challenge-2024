#!/bin/bash

# Fresh install with "rm .micromamba/micromamba .done-*"

set -e

nj=$(nproc)
home=$PWD
venv_dir=$PWD/venv
source ./env.sh

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
  dir="./anonymization/modules/nac/coqui_tts"
  [ -d $dir ] && yes | rm -rf $dir
  git clone https://github.com/m-pana/nac-requirements.git $dir
  pip install -e $dir
  touch $mark
fi