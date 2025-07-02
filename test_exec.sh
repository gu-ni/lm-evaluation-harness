models=(
    # "meta-llama/Meta-Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-Math-7B-Instruct"
    "Qwen/Qwen2-Math-7B-Instruct"
    "deepseek-math-7b-instruct"
    "MathGenie/MathCoder2-Llama-3-8B"
    "google/gemma-2-9b-it"
    "google/gemma-2-27b-it"
    "nvidia/AceReason-Nemotron-1.1-7B"
    "mistralai/Mistral-Small-24B-Instruct-2501"
)

for model_name in "${models[@]}"; do
    echo "===> Evaluating $model_name"
    lm_eval --model vllm \
        --model_args pretrained=$model_name \
        --tasks gsm8k \
        --device cuda:0 \
        --batch_size auto \
        --log_samples \
        --output_path base
done

# for ((i=1; i<=100000; i++)); do
#     echo "$i"
#     python /home/work/users/PIL_ghj/LLM/code/generate_qa_datasets_copy.py
# done
