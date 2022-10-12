#!/bin/bash
set -e

SUMMARY_STYLE="extract"

EXTRACT_EXPERIMENT="add_doc_bart_large_cnn_oracle_drop"
ABSTRACT_EXPERIMENT="extract_indicators"
EXTRACT_MODEL="facebook/bart-large"
ABSTRACT_MODEL="facebook/bart-large"
SPLIT="test"
MAX_EXAMPLES=1000
DEVICE=1

#echo "Generating single beam"
#python generate.py --wandb_name $EXTRACT_EXPERIMENT --summary_style extract --split $SPLIT --max_examples $MAX_EXAMPLES \
#  --hf_model $EXTRACT_MODEL --decode_method beam --num_return_sequences 1
#
#echo "Generating 16 beams"
#python generate.py --wandb_name $EXTRACT_EXPERIMENT --summary_style extract --split $SPLIT --max_examples $MAX_EXAMPLES \
#  --hf_model $EXTRACT_MODEL --decode_method beam --num_return_sequences 16
#
echo "Generating 16 diverse beams"
python generate.py --wandb_name $EXTRACT_EXPERIMENT --summary_style extract --split $SPLIT --max_examples $MAX_EXAMPLES \
  --hf_model $EXTRACT_MODEL --decode_method diverse --num_return_sequences 16

echo "Generating Abstracts from the Extracts we've generated..."
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
