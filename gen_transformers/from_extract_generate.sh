#!/bin/bash
set -e
export ROUGE_HOME=/home/griffin/faith-sum/eval/ROUGE-1.5.5/

EXTRACT_EXPERIMENT="add_doc_bart_large_cnn"
ABSTRACT_EXPERIMENT=$1
ABSTRACT_MODEL=$2
SPLIT="test"


DEVICE=$3
if [[ $3 -eq 0 ]] ; then
  DEVICE=0
fi


MAX_EXAMPLES=$4
if [[ $4 -eq 0 ]] ; then
  MAX_EXAMPLES=1000
fi


echo "Generating Conditional Abstracts..."

echo "Abstracting from single extract beam"
python gen_from_extract.py --extract_experiment $EXTRACT_EXPERIMENT --split $SPLIT --max_examples $MAX_EXAMPLES \
  --abstract_experiment $ABSTRACT_EXPERIMENT --hf_model $ABSTRACT_MODEL --decode_method beam --num_candidates 1 \
  --gpu_device $DEVICE

echo "Abstracting from 16 beams"
python gen_from_extract.py --extract_experiment $EXTRACT_EXPERIMENT --split $SPLIT --max_examples $MAX_EXAMPLES \
  --abstract_experiment $ABSTRACT_EXPERIMENT --hf_model $ABSTRACT_MODEL --decode_method beam --num_candidates 16 \
  --gpu_device $DEVICE

echo "Abstracting from 16 diverse beams"
python gen_from_extract.py --extract_experiment $EXTRACT_EXPERIMENT --split $SPLIT --max_examples $MAX_EXAMPLES \
  --abstract_experiment $ABSTRACT_EXPERIMENT --hf_model $ABSTRACT_MODEL --decode_method diverse --num_candidates 16 \
  --gpu_device $DEVICE
