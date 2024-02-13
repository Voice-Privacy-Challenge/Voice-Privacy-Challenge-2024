#!/bin/bash

source env.sh

# Generate b2 anonymized audio (libri dev+test set & libri-360h)
python run_anonymization_dsp.py --config anon_dsp.yaml

# Perform libri dev+test pre evaluation using pretrained ASV models
python run_evaluation.py --config eval_pre_from_anon_datadir.yaml

# Train post ASV using anonymized libri-360 and perform libri dev+test post evaluation
# ASV training takes 2hours
python run_evaluation.py --config eval_post_scratch_from_anon_datadir.yaml 
