#!/bin/bash
set -e

DEVICE=$1
DATASET=$2
ABSTRACT_EXPERIMENT=$3

python sample_extracts.py --dataset $DATASET
python generate_ea_targets.py --dataset $DATASET --device $DEVICE --