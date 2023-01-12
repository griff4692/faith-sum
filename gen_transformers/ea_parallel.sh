#!/bin/bash
set -e

#SPLIT=$1
#EXTRACT_EXPERIMENT=$2
#ABSTRACT_EXPERIMENT=$3
#MAX_EXAMPLES=$4
NUM_CHUNKS=$5


if [ $NUM_CHUNKS -eq 8 ]; then
  bash ea_pipeline.sh 0 $1 $2 $3 $4 0 $NUM_CHUNKS & \
  bash ea_pipeline.sh 1 $1 $2 $3 $4 1 $NUM_CHUNKS & \
  bash ea_pipeline.sh 2 $1 $2 $3 $4 2 $NUM_CHUNKS & \
  bash ea_pipeline.sh 3 $1 $2 $3 $4 3 $NUM_CHUNKS & \
  bash ea_pipeline.sh 4 $1 $2 $3 $4 4 $NUM_CHUNKS & \
  bash ea_pipeline.sh 5 $1 $2 $3 $4 5 $NUM_CHUNKS & \
  bash ea_pipeline.sh 6 $1 $2 $3 $4 6 $NUM_CHUNKS & \
  bash ea_pipeline.sh 7 $1 $2 $3 $4 7 $NUM_CHUNKS & \
else
  bash ea_pipeline.sh 0 $1 $2 $3 $4 0 $NUM_CHUNKS & \
  bash ea_pipeline.sh 1 $1 $2 $3 $4 1 $NUM_CHUNKS & \
  bash ea_pipeline.sh 2 $1 $2 $3 $4 2 $NUM_CHUNKS & \
  bash ea_pipeline.sh 3 $1 $2 $3 $4 3 $NUM_CHUNKS &
fi

wait

echo "Fini! Now merging ${NUM_CHUNKS} chunks into a single file..."
python merge_chunks.py --split $1 --extract_experiment $2 --abstract_experiment $3 --num_chunks $NUM_CHUNKS
