#!/bin/bash

# Fresh install with "rm .micromamba/micromamba .done-*"

set -e

nj=$(nproc)
home=$PWD
venv_dir=$PWD/venv
source ./env.sh

ESPAK_VERSION=1.51.1

compute_and_write_hash "anonymization/pipelines/sttts/requirements.txt"  # SHA256: 68b108ebe942a9de604440d8a1278710ee29e559942702c366b0de414edf890a
trigger_new_install "exp/sttts_models .done-sttts-requirements .done-espeak"

# Download GAN pre-models only if perform GAN anonymization
if [ ! -d exp/sttts_models ]; then
    echo "Download pretrained models of GAN-based speaker anonymization system..."
    mkdir -p exp/sttts_models
    wget -q -O exp/sttts_models/anonymization.zip https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v2.0/anonymization.zip
    wget -q -O exp/sttts_models/asr.zip https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v2.0/asr.zip
    wget -q -O exp/sttts_models/tts.zip https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/v2.0/tts.zip
    unzip -oq exp/sttts_models/asr.zip -d exp/sttts_models
    unzip -oq exp/sttts_models/tts.zip -d exp/sttts_models
    unzip -oq exp/sttts_models/anonymization.zip -d exp/sttts_models
    rm exp/sttts_models/*.zip
fi


mark=.done-sttts-requirements
if [ ! -f $mark ]; then
  echo " == Installing STTTS python libraries =="
  pip3 install -r anonymization/pipelines/sttts/requirements.txt  || exit 1
  touch $mark
fi


mark=.done-espeak
if [ ! -f $mark ]; then
  echo " == Installing G2P espeak-ng =="
  wget https://github.com/espeak-ng/espeak-ng/archive/$ESPAK_VERSION/espeak-ng-$ESPAK_VERSION.tar.gz
  \rm espeak-ng-$ESPAK_VERSION -rf || true
  tar -xvzf ./espeak-ng-$ESPAK_VERSION.tar.gz
  \rm ./espeak-ng-$ESPAK_VERSION.tar.gz
  cd espeak-ng-$ESPAK_VERSION
  ./autogen.sh || true # First time fails?
  ./autogen.sh

  sed -i "s|.*define PATH_ESPEAK_DATA.*|\#define PATH_ESPEAK_DATA \"${venv_dir}/share/espeak-ng-data\"|" src/libespeak-ng/speech.h
  sed -i "58d" src/libespeak-ng/speech.h
  sed -i "59d" src/libespeak-ng/speech.h

  ./configure
  make -j $nj src/espeak-ng src/speak-ng
  make -j $nj

  make DESTDIR="$venv_dir" install

  yes | cp -rf ${venv_dir}/usr/local/* ${venv_dir} || true

  # espeak-ng --voices
  pip3 install phonemizer
  python3 -c "import phonemizer; phonemizer.phonemize('Good morning', language='en-gb')"
  python3 -c "import phonemizer; phonemizer.phonemize('Guten Morgen', language='de')"
  python3 -c "import phonemizer; phonemizer.phonemize('Bonjour', language='fr-fr')"

  cd $home
  touch $mark
fi
