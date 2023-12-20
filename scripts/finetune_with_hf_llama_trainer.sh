export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_SIZE=7B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

nohup deepspeed open_instruct/finetune_trainer.py \
    --deepspeed ds_configs/stage3_no_offloading.conf \
    --model_name_or_path /paratera5-data/private/liuziyi/models/Llama-2-7b-chat-hf \
    --tokenizer_name /paratera5-data/private/liuziyi/models/Llama-2-7b-chat-hf \
    --use_flash_attn True \
    --use_fast_tokenizer False \
    --train_file /paratera5-data/private/liuziyi/mygit/open-instruct/data/processed/ability/all_data_with_math.jsonl \
    --max_seq_length 4096 \
    --preprocessing_num_workers 128 \
    --do_train \
    --seed 42 \
    --ddp_timeout 1800000 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --evaluation_strategy "no" \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --num_train_epochs 3 \
    --output_dir output/hf_train/llama_ability_math \
    --bf16 \
    --tf32 True \
    --torch_dtype bfloat16 \
    --add_extra_id \
    --overwrite_output_dir \
    --cache_dir /paratera5-data/private/liuziyi/cache/llama_math \
    --report_to "tensorboard" &> hf_llama_ability_math.out &
    # --overwrite_output_dir \
    # --overwrite_cache \
    # --resume_from_checkpoint /paratera5-data/private/liuziyi/mygit/open-instruct/output/hf_train/llama_ability/checkpoint-6000 \
