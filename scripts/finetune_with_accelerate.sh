export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_SIZE=7B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_no_offloading_accelerate.conf \
    open_instruct/finetune.py \
    --model_name_or_path /paratera5-data/private/liuziyi/models/Llama-2-7b-chat-hf \
    --use_flash_attn \
    --tokenizer_name /paratera5-data/private/liuziyi/models/Llama-2-7b-chat-hf \
    --use_slow_tokenizer \
    --train_file data/processed/openchat_sharegpt_v3/openchat_sharegpt_v3_data.jsonl \
    --max_seq_length 4096 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir output/llama2_test_sharegpt_${MODEL_SIZE}/ \
    --with_tracking \
    --report_to tensorboard \
    --add_extra_id \
    --max_train_steps 2 \
    --logging_steps 1 #&> llama2_binary_sharegpt_sft.out &
    # --overwrite_cache \