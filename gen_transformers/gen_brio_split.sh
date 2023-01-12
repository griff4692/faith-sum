#!/bin/bash
set -e

DATASET=$1
EXP=$2
METHOD=$3  # "beam" "diverse"
SPLIT=$4
NCANDS=16

SHARED="--dataset $DATASET --experiment $EXP --summary_style abstract --num_return_sequences $NCANDS --decode_method $METHOD"
python generate.py $SHARED --split $SPLIT
