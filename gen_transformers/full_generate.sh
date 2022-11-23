#!/bin/bash
set -e
export ROUGE_HOME=~/faith-sum/eval/ROUGE-1.5.5/

EXTRACT_EXPERIMENT="add_doc_bart_large_cnn"
ABSTRACT_EXPERIMENT="bart_large_cnn_from_extract_e3"
HF_MODEL="facebook/bart-large-cnn"
SPLIT="train"
DATASET="cnn_dailymail"

N_CAND=16
DEVICE=$1
CHUNK=$2
NUM_CHUNKS=$3
BATCH_SIZE=64

SHARED_ARGS="--hf_model $HF_MODEL --gpu_device $DEVICE --dataset $DATASET --chunk $CHUNK --split $SPLIT --decode_method beam"

GEN_ARGS="--num_return_sequences $N_CAND --wandb_name $EXTRACT_EXPERIMENT --batch_size $BATCH_SIZE --num_chunks $NUM_CHUNKS --summary_style extract --extract_method generate --summary_style extract -use_hf_rouge"
python generate.py $SHARED_ARGS $GEN_ARGS

echo "Now generating abstracts from these extracts"

FROM_EXTRACT_ARGS="--extract_experiment $EXTRACT_EXPERIMENT --abstract_experiment $ABSTRACT_EXPERIMENT --num_candidates $N_CAND --num_return_sequences 1"
python gen_from_extract.py $SHARED_ARGS $FROM_EXTRACT_ARGS

echo "Fini!"
