#!/bin/bash
set -e

ABSTRACT_EXPERIMENT=$1
NUM_CHUNKS=$2


if [ $NUM_CHUNKS -eq 8 ]; then
  python generate_ea_targets.py --abstract_experiment $ABSTRACT_EXPERIMENT --num_chunks $NUM_CHUNKS --device 0 --chunk 0 & \
  python generate_ea_targets.py --abstract_experiment $ABSTRACT_EXPERIMENT --num_chunks $NUM_CHUNKS --device 1 --chunk 1 & \
  python generate_ea_targets.py --abstract_experiment $ABSTRACT_EXPERIMENT --num_chunks $NUM_CHUNKS --device 2 --chunk 2 & \
  python generate_ea_targets.py --abstract_experiment $ABSTRACT_EXPERIMENT --num_chunks $NUM_CHUNKS --device 3 --chunk 3 & \
  python generate_ea_targets.py --abstract_experiment $ABSTRACT_EXPERIMENT --num_chunks $NUM_CHUNKS --device 4 --chunk 4 & \
  python generate_ea_targets.py --abstract_experiment $ABSTRACT_EXPERIMENT --num_chunks $NUM_CHUNKS --device 5 --chunk 5 & \
  python generate_ea_targets.py --abstract_experiment $ABSTRACT_EXPERIMENT --num_chunks $NUM_CHUNKS --device 6 --chunk 6 & \
  python generate_ea_targets.py --abstract_experiment $ABSTRACT_EXPERIMENT --num_chunks $NUM_CHUNKS --device 7 --chunk 7 & \
else
  python generate_ea_targets.py --abstract_experiment $ABSTRACT_EXPERIMENT --num_chunks $NUM_CHUNKS --device 0 --chunk 0 & \
  python generate_ea_targets.py --abstract_experiment $ABSTRACT_EXPERIMENT --num_chunks $NUM_CHUNKS --device 1 --chunk 1 & \
  python generate_ea_targets.py --abstract_experiment $ABSTRACT_EXPERIMENT --num_chunks $NUM_CHUNKS --device 2 --chunk 2 & \
  python generate_ea_targets.py --abstract_experiment $ABSTRACT_EXPERIMENT --num_chunks $NUM_CHUNKS --device 3 --chunk 3 & \
fi

wait

echo "Fini! Now merging ${NUM_CHUNKS} chunks into a single file..."
python merge_chunks.py --split $1 --extract_experiment $2 --abstract_experiment $3 --num_chunks $NUM_CHUNKS
