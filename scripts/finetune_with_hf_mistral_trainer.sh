export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_SIZE=7B
NUM_GPUS=8
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

deepspeed open_instruct/finetune_trainer.py \
    --deepspeed ds_configs/stage3_no_offloading.conf \
    --model_name_or_path /storage/home/lanzhenzhongLab/liuziyi/models/mistralai/Mistral-7B-v0.1 \
    --tokenizer_name /storage/home/lanzhenzhongLab/liuziyi/models/mistralai/Mistral-7B-v0.1 \
    --use_flash_attn True \
    --use_fast_tokenizer False \
    --train_file /storage/home/lanzhenzhongLab/liuziyi/mygit/open-instruct/data/processed/ability/reasoning/metamathqa_74.0k.jsonl \
    --max_seq_length 4096 \
    --preprocessing_num_workers 128 \
    --do_train \
    --ddp_timeout 180000 \
    --seed 42 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --evaluation_strategy "no" \
    --logging_steps 1 \
    --save_strategy steps \
    --save_total_limit 2 \
    --save_steps 500 \
    --num_train_epochs 3 \
    --output_dir output/hf_train/mistral_metamath_74k \
    --bf16 \
    --tf32 True \
    --torch_dtype bfloat16 \
    --overwrite_output_dir \
    --cache_dir /storage/home/lanzhenzhongLab/liuziyi/cache/mistral_metamath_74k \
    --report_to "tensorboard" #&> hf_mistral_metamath_74k.out &
    # --overwrite_output_dir \
    # --overwrite_cache \
    # --resume_from_checkpoint output/hf_train/llama2_7B_sharegpt_extraid/checkpoint-2596 \
