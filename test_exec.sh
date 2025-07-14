model_args_list=(
    "pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct"
    "pretrained=Qwen/Qwen2.5-Math-7B-Instruct"
    "pretrained=Qwen/Qwen2-Math-7B-Instruct"
    # "pretrained=deepseek-math-7b-instruct"
    "pretrained=MathGenie/MathCoder2-Llama-3-8B"
    "pretrained=google/gemma-2-9b-it"
    "pretrained=google/gemma-2-27b-it"
    # "pretrained=nvidia/AceReason-Nemotron-1.1-7B,max_gen_toks=20000"
    "pretrained=mistralai/Mistral-Small-24B-Instruct-2501"
)

for model_args in "${model_args_list[@]}"; do
    echo "===> Evaluating $model_args"
    lm_eval --model vllm \
        --model_args $model_args \
        --tasks aime \
        --device cuda:0 \
        --batch_size auto \
        --log_samples \
        --output_path base
done

for ((i=1; i<=100000; i++)); do
    echo "$i"
    python /home/work/users/PIL_ghj/LLM/code/generate_qa_datasets_copy.py
done
