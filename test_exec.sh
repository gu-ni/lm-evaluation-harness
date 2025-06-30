models=(
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
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
        --output_path results/${model_name//\//_}.json
done