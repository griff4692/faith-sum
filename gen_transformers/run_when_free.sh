GPU=$1
STRAT=$2
EXP=$3
free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $GPU | grep -Eo [0-9]+)

while [ $free_mem -lt 10000 ]; do
    free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $GPU | grep -Eo [0-9]+)
    echo $free_mem
    sleep 5
done

ARGS="--experiment $EXP --contrast_intra_sample_strategy $STRAT -contrast --contrast_ckpt primera_ft_chemistry -use_mixed_methods --max_num_rank 4 --max_num_positive 2 --max_num_negative 2 --reference_status remove --positive_methods reference --negative_methods none --contrast_objective margin_rank --max_target_length 512 --contrast_metrics relevance --gradient_accumulation_steps 8 --dataset chemistry --hf_model primera --validate_every_n_steps 1000 --max_train_steps 10000 --mle_weight 0.1 --contrast_weight 1.0 --margin_scale 0.1 --length_penalty 2.0 -save_every_time"
echo $ARGS
CUDA_VISIBLE_DEVICES=$GPU python run.py $ARGS
