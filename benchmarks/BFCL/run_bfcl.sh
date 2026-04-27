#!/bin/bash
# Run BFCL generate + evaluate for a given model
# Usage: bash run_bfcl.sh <model_name> <model_path>
# Example: bash run_bfcl.sh toolrl-grpo-qwen-1.5b /app/models/toolrl-grpo-qwen-1.5b/actor/global_step_90
#
# Skipped categories (require external API keys):
#   web_search_base, web_search_no_snippet — require SERPAPI_API_KEY
#
# All other categories run fully locally (no internet needed):
#   non-live AST, live AST, multi_turn (local Python sim), memory (faiss + sentence-transformers)

MODEL_NAME=$1
MODEL_PATH=$2

CATEGORIES="simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum"

echo "=== BFCL Generate: $MODEL_NAME ==="
bfcl generate \
    --model "$MODEL_NAME" \
    --test-category "$CATEGORIES" \
    --backend vllm \
    --local-model-path "$MODEL_PATH" \
    --num-gpus 4 \
    --gpu-memory-utilization 0.85 \
    --allow-overwrite

echo "=== BFCL Evaluate: $MODEL_NAME ==="
bfcl evaluate \
    --model "$MODEL_NAME" \
    --test-category "$CATEGORIES"

echo "=== Done: $MODEL_NAME ==="
