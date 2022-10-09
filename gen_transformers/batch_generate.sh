#!/bin/bash

BATCH_SIZE=8
GPU_DEVICE=$2
MAX_EXAMPLES=1000
SPLIT='test'
NUM_RETURN=16

if [[ $1 == "bart_large_cnn" ]]
then
  WANDB_NAME="null"
  EXPERIMENT="bart_large_cnn"
  HF_MODEL="facebook/bart-large-cnn"
  SUMMARY_STYLE="abstract"
elif [[ $1 == "gen_abstract_full" ]]
then
  WANDB_NAME="gen_abstract_full"
  HF_MODEL="facebook/bart-base"
  EXPERIMENT=$WANDB_NAME
  SUMMARY_STYLE="abstract"
else
  WANDB_NAME="add_doc"
  HF_MODEL="facebook/bart-base"
  EXPERIMENT=$WANDB_NAME
  SUMMARY_STYLE="extract"
fi

echo $MAX_EXAMPLES
echo $SPLIT
echo $NUM_RETURN
echo $WANDB_NAME
echo $EXPERIMENT
echo $HF_MODEL
echo $SUMMARY_STYLE
echo $BATCH_SIZE

#echo "Beam 1..."
#python generate.py --wandb_name $WANDB_NAME --summary_style $SUMMARY_STYLE --hf_model $HF_MODEL \
#  --batch_size $BATCH_SIZE --split $SPLIT --max_examples $MAX_EXAMPLES --gpu_device $GPU_DEVICE \
#  --decode_method beam --num_return_sequences 1 --experiment $EXPERIMENT

#echo "Beam ${NUM_RETURN}..."
#python generate.py --wandb_name $WANDB_NAME --summary_style $SUMMARY_STYLE --hf_model $HF_MODEL \
#  --batch_size $BATCH_SIZE --split $SPLIT --max_examples $MAX_EXAMPLES --gpu_device $GPU_DEVICE \
#  --decode_method beam --num_return_sequences $NUM_RETURN --experiment $EXPERIMENT

echo "Nucleus ${NUM_RETURN}..."
python generate.py --wandb_name $WANDB_NAME --summary_style $SUMMARY_STYLE --hf_model $HF_MODEL \
  --batch_size 1 --split $SPLIT --max_examples $MAX_EXAMPLES --gpu_device $GPU_DEVICE \
  --decode_method nucleus --num_return_sequences $NUM_RETURN --experiment $EXPERIMENT

#echo "Diverse ${NUM_RETURN}..."
#python generate.py --wandb_name $WANDB_NAME --summary_style $SUMMARY_STYLE --hf_model $HF_MODEL \
#  --batch_size $BATCH_SIZE --split $SPLIT --max_examples $MAX_EXAMPLES --gpu_device $GPU_DEVICE \
#  --decode_method diverse --num_return_sequences $NUM_RETURN --experiment $EXPERIMENT
