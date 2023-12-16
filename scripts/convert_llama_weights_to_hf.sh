LLAMA_FOLDER=/paratera5-data/private/liuziyi/mygit/open-instruct/output/llama2_sharegpt_7B

for MODEL_SIZE in 7B; do
    echo "Converting Llama ${MODEL_SIZE} to HuggingFace format"
    python -m transformers.models.llama.convert_llama_weights_to_hf \
    --input_dir $LLAMA_FOLDER/ \
    --model_size $MODEL_SIZE \
    --output_dir /paratera5-data/private/liuziyi/mygit/open-instruct/output/llama2_sharegpt_7B/hf_model
done