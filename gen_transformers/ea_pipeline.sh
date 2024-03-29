#!/bin/bash
set -e
export ROUGE_HOME=~/faith-sum/eval/ROUGE-1.5.5/

BATCH_SIZE=32
N_CAND=16

DEVICE=$1
SPLIT=$2
EXTRACT_EXPERIMENT=$3
ABSTRACT_EXPERIMENT=$4

SHARED_ARGS="--device $DEVICE --split $SPLIT --decode_method beam"

GEN_ARGS="--num_return_sequences $N_CAND --wandb_name $EXTRACT_EXPERIMENT --batch_size $BATCH_SIZE --summary_style extract --summary_style extract -use_hf_rouge"
python generate.py $SHARED_ARGS $GEN_ARGS

echo "Now generating abstracts from these extracts"
FROM_EXTRACT_ARGS="--extract_experiment $EXTRACT_EXPERIMENT --abstract_experiment $ABSTRACT_EXPERIMENT --num_candidates $N_CAND --num_return_sequences 1"
python gen_from_extract.py $SHARED_ARGS $FROM_EXTRACT_ARGS

echo "Fini!"
