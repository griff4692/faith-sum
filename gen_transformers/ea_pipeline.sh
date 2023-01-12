#!/bin/bash
set -e

BATCH_SIZE=32
N_CAND=16

DEVICE=$1
SPLIT=$2
EXTRACT_EXPERIMENT=$3
ABSTRACT_EXPERIMENT=$4


if [ $5 -eq 0 ]; then
  MAX_EXAMPLES=9999999
else
  MAX_EXAMPLES=$5
fi

if [ $6 -eq 0 ]; then
  CHUNK=""
else
  CHUNK=$6
fi

if [ $7 -eq 0 ]; then
  NUM_CHUNKS=8
else
  NUM_CHUNKS=$7
fi

SHARED_ARGS="--device $DEVICE --split $SPLIT --decode_method beam"

GEN_ARGS="--num_return_sequences $N_CAND --wandb_name $EXTRACT_EXPERIMENT --batch_size $BATCH_SIZE --summary_style extract -use_hf_rouge --chunk ${CHUNK} --num_chunks ${NUM_CHUNKS}"
python generate.py $SHARED_ARGS $GEN_ARGS --max_examples $MAX_EXAMPLES

echo "Now generating abstracts from these extracts"
FROM_EXTRACT_ARGS="--extract_experiment $EXTRACT_EXPERIMENT --abstract_experiment $ABSTRACT_EXPERIMENT --num_candidates $N_CAND --num_return_sequences 1 --chunk ${CHUNK} --num_chunks ${NUM_CHUNKS}"
python gen_from_extract.py $SHARED_ARGS $FROM_EXTRACT_ARGS -add_abstract_experiment

echo "Fini!"
