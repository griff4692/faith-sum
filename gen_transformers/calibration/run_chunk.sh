#!/bin/bash
set -e

DEVICE=$1
ABSTRACT_EXPERIMENT=$2

SHARED="--device $DEVICE --abstract_experiment $ABSTRACT_EXPERIMENT"

if [ -z "$3" ]; then
  python generate_ea_targets.py $SHARED
else
  CHUNK=$3
  NUM_CHUNKS=$4
  echo "Running chunk $CHUNK / ${NUM_CHUNKS}"
  python generate_ea_targets.py $SHARED --chunk $CHUNK --num_chunks $NUM_CHUNKS
fi
