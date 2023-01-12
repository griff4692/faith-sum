#!/bin/bash
set -e

NCANDS=16
BSIZE=32

DEVICE=$1
DATASET=$2
EXP=$3
METHOD=$4  # "beam" "diverse"
SPLIT=$5

SHARED="--device $DEVICE --dataset $DATASET --experiment $EXP --summary_style abstract --num_return_sequences $NCANDS --decode_method $METHOD --batch_size ${BSIZE}"

if [ $6 -eq 0 ]; then
  python generate.py $SHARED --split $SPLIT
else
  NUM_CHUNKS=8
  echo "Running chunk $6 / ${NUM_CHUNKS}"
  python generate.py $SHARED --split $SPLIT --chunk $6 --num_chunks $NUM_CHUNKS
fi
