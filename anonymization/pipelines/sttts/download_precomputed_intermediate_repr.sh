#!/bin/bash

set -e

nj=$(nproc)
home=$PWD
venv_dir=$PWD/venv
source ./env.sh

# Download precomputed intermediate results (transcription, prosody, speaker embeddings) to avoid slow recomputation if not necessary
if [ ! -d exp/anon_pipeline_sttts/transcription/asr_branchformer_tts-phn_en/train-clean-360 ]; then
    echo "Download precomputed intermediate results of GAN-based speaker anonymization system..."
    mkdir -p exp/anon_pipeline_sttts
    wget -q -O exp/anon_pipeline_sttts/intermediate_representations_libri.zip https://github.com/DigitalPhonetics/speaker-anonymization/releases/download/intermediate_representations/intermediate_representations_libri.zip
    unzip -oq exp/anon_pipeline_sttts/intermediate_representations_libri.zip -d exp/anon_pipeline_sttts
    rm exp/anon_pipeline_sttts/intermediate_representations_libri.zip
fi
