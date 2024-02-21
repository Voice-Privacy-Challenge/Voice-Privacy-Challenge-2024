#!/bin/bash

set -e

source env.sh

### Variables

# Anonymization pipeline
anon_config=anon_mcadams.yaml
# anon_config=anon_sttts.yaml

force_compute=
# force_compute='--force_compute True'

# Modify run_evaluation(s) configs, see below
eval_overwrite=

### Anonymization + Evaluation:

# find the $anon_suffix (data/dataset_$anon_suffix) = to where the anonymization produces the data files
anon_suffix=$(python3 -c "from hyperpyyaml import load_hyperpyyaml; f = open('./configs/${anon_config}'); print(load_hyperpyyaml(f, None).get('anon_suffix', ''))")
if [[ $anon_suffix ]]; then
  eval_overwrite="{\"anon_data_suffix\": \"$anon_suffix\"}"
fi

# Generate anonymized audio (libri dev+test set & libri-360h)
python run_anonymization.py --config ${anon_config} ${force_compute}

# Perform libri dev+test pre evaluation using pretrained ASR/ASV models
python run_evaluation.py --config eval_pre.yaml --overwrite "${eval_overwrite}"

# Train post ASV using anonymized libri-360 and perform libri dev+test post evaluation
# ASV training takes 2hours
python run_evaluation.py --config eval_post.yaml --overwrite "${eval_overwrite}"
