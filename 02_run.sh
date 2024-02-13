#!/bin/bash

set -e

source env.sh

#generate b2 anonymized audio (libri dev+test set - 25min & libri-360h)
python run_anonymization_dsp.py --config anon_dsp.yaml

#perform libri dev+test pre evaluation using pretrained ASV/ASR models
#ASV-8mins, ASR- hours if eval_batchsize=3, ASR- hours if evel_batchsize=2 (default)
python run_evaluation.py --config eval_pre_from_anon_datadir.yaml


#train post ASV using anonymized libri-360 and perform libri dev+test post evaluation
#ASV training takes 2hours
python run_evaluation.py --config eval_post_scratch_from_anon_datadir.yaml 
