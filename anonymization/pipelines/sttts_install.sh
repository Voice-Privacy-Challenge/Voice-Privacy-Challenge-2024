#!/bin/bash

# Fresh install with "rm .micromamba/micromamba .done-*"

set -e

nj=$(nproc)

home=$PWD
ESPAK_VERSION=1.51.1

venv_dir=$PWD/venv
export MAMBA_ROOT_PREFIX=".micromamba"  # Local install of micromamba (where the libs/bin will be cached)

source ./env.sh


mark=.done-sttts-requirements
if [ ! -f $mark ]; then
  echo " == Installing python libraries =="

  pip3 install -r anonymization/pipelines/sttts_requirements.txt  || exit 1
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
