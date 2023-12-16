export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_SIZE=7B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

nohup deepspeed open_instruct/finetune_trainer.py \
    --deepspeed ds_configs/stage3_no_offloading.conf \
    --model_name_or_path /paratera5-data/private/liuziyi/mygit/open-instruct/output/hf_train/llama2_7B_sharegpt_extraid \
    --tokenizer_name /paratera5-data/private/liuziyi/mygit/open-instruct/output/hf_train/llama2_7B_sharegpt_extraid \
    --use_flash_attn True \
    --use_fast_tokenizer False \
    --train_file /paratera5-data/private/liuziyi/mygit/open-instruct/data/processed/openchat_sharegpt_v3/openchat_sharegpt_v3_data.jsonl \
    --max_seq_length 4096 \
    --preprocessing_num_workers 64 \
    --do_train \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --evaluation_strategy "no" \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --num_train_epochs 3 \
    --output_dir output/hf_train/llama2_7B_sharegpt_extraid \
    --bf16 \
    --tf32 True \
    --torch_dtype bfloat16 \
    --overwrite_cache \
    --add_extra_id \
    --resume_from_checkpoint output/hf_train/llama2_7B_sharegpt_extraid/checkpoint-1298 \
    --report_to "tensorboard" &> hf_sharegpt_extraid.out &
    # --overwrite_output_dir \
    # --overwrite_cache \
