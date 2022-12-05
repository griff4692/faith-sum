#!/bin/bash
set -e

DEVICE=$1
FT_PATH="/nlp/projects/faithsum/weights/add_doc_bart_large_cnn/faith_sum/3ne4vd5n/checkpoints/epoch=0-step=17944.ckpt"

MAX_EPOCH=1  # It's quick
MAX_BRIO_CANDS=8
PER_DEVICE_BS=8

SHARED="--max_epochs ${MAX_EPOCH} --brio_score_mode score --per_device_train_bs ${PER_DEVICE_BS} --max_brio_candidates ${MAX_BRIO_CANDS} --save_top_k 3 --dataset cnn_dailymail --gpu_device ${DEVICE} --hf_model facebook/bart-large-cnn -add_brio_loss --extract_method generate --val_metric_mode max --val_monitor_metric rank_corel --copy_bart_class_dropout 0.1"

echo "Baseline"
BASE_ARGS="${SHARED} --mle_weight 0.1 --brio_weight 1.0 --pretrained_path ${FT_PATH} --experiment cnn_score_brio_baseline"
echo $BASE_ARGS
python main.py $BASE_ARGS

echo "Low MLE"
LOW_MLE_ARGS="${SHARED} --mle_weight 0.1 --brio_weight 1.0 --pretrained_path ${FT_PATH} --experiment cnn_score_brio_low_mle"
echo $LOW_MLE_ARGS
python main.py $LOW_MLE_ARGS

echo "Low Contrast Weight"
LOW_CW_ARGS="${SHARED} --mle_weight 1.0 --brio_weight 0.1 --pretrained_path ${FT_PATH} --experiment cnn_score_brio_low_cw"
echo $LOW_CW_ARGS
python main.py $LOW_CW_ARGS

echo "From Scratch"
SCRATCH_ARGS="$SHARED --mle_weight 1.0 --brio_weight 1.0 --experiment cnn_score_brio_no_pretrain"
echo $SCRATCH_ARGS
python main.py $SCRATCH_ARGS
