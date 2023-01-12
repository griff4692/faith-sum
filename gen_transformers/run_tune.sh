#!/bin/bash
set -e

SPLIT="validation"
NCANDS=16

DEVICE=$1
EXTRACT_EXPERIMENT=$2
ABSTRACT_EXPERIMENT=$3
MAX_EXAMPLES=$4

SHARED_ARGS="--device $DEVICE --split $SPLIT --max_examples $MAX_EXAMPLES --decode_method beam"
GEN_ARGS="--num_return_sequences $NCANDS --wandb_name $EXTRACT_EXPERIMENT --summary_style extract -use_hf_rouge"
python generate.py $SHARED_ARGS $GEN_ARGS
TUNE_ARGS="--abstract_experiment $ABSTRACT_EXPERIMENT --extract_experiment $EXTRACT_EXPERIMENT --num_candidates $NCANDS"
python tune_lp.py $SHARED_ARGS $TUNE_ARGS
