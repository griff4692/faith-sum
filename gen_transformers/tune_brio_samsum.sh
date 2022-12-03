#!/bin/bash
set -e


FT_PATH="/nlp/projects/faithsum/weights/samsum_extract_generator/faith_sum/2olz18tl/checkpoints/epoch\=1-step\=1611.ckpt"

DEVICE=$1
LR=3e-5
MAX_EPOCH=2  # It's quick
MAX_BRIO_CANDS=8
PER_DEVICE_BS=4

SHARED="--max_epochs ${MAX_EPOCH} --per_device_train_bs ${PER_DEVICE_BS} --save_top_k 0 --max_brio_candidates ${MAX_BRIO_CANDS} --dataset samsum --lr ${LR} --gpu_device ${DEVICE} --hf_model lidiya/bart-large-xsum-samsum --pretrained_path $FT_PATH -add_brio_loss --extract_method generate --val_metric_mode max --val_monitor_metric extract_mean_f1 --copy_bart_class_dropout 0.1"

CWS=(0.1 1.0 10.0)
MWS=(0.1 0.5 1.0)
LPS=(0.0 0.5 1.0 4.0)
SCALES=(0.001 0.01 0.1 1.0)

for CW in "${CWS[@]}"
do
  for MW in "${MWS[@]}"
  do
    for LP in "${LPS[@]}"
    do
      for SCALE in "${SCALES[@]}"
      do
        EXP_ARGS="--brio_scale ${SCALE} --mle_weight ${MW} --brio_weight ${CW} --brio_length_penalty ${LP}"
        EXP_NAME="samsum_brio_mw_${MW}_cw_${CW}_sc_${SCALE}_lp_${LP}"
        echo "Starting Training for ${EXP_NAME}"
        echo "${SHARED} ${EXP_ARGS}"
        exit
        python main.py $SHARED $EXP_ARGS --experiment $EXP_NAME
      done
    done
  done
done
