#!/bin/bash

set -e

source env.sh

anon_config=anon_dsp.yaml

force_compute=
# force_compute='--force_compute True'

# Generate anonymized audio (libri dev+test set & libri-360h)
python run_anonymization.py --config ${anon_config} ${force_compute}

# Perform libri dev+test pre evaluation using pretrained ASR/ASV models
python run_evaluation.py --config eval_pre.yaml

# Train post ASV using anonymized libri-360 and perform libri dev+test post evaluation
# ASV training takes 2hours
python run_evaluation.py --config eval_post.yaml
