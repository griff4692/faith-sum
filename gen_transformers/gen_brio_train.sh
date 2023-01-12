#!/bin/bash
set -e

NUM_CHUNKS=8
NCANDS=16

DATASET=$1
EXP=$2
METHOD=$3  # "beam" "diverse"

SHARED="--split train --summary_style abstract --dataset $DATASET --experiment $EXP --num_return_sequences $NCANDS --decode_method $METHOD --num_chunks $NUM_CHUNKS"

python generate.py $SHARED --chunk 0 & \
python generate.py $SHARED --chunk 1 & \
python generate.py $SHARED --chunk 2 & \
python generate.py $SHARED --chunk 3 & \
python generate.py $SHARED --chunk 4 & \
python generate.py $SHARED --chunk 5 & \
python generate.py $SHARED --chunk 6 & \
python generate.py $SHARED --chunk 7
