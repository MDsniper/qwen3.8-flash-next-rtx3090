#!/usr/bin/env bash
# Generate the MoE routing profile (qwen38-merged.csv) used by --moe-cache-profile.
# Runs llama-moe-trace over two contrasting workloads (code + chat) and merges them.
# One-time step per model; takes a few minutes on a warm page cache.
set -euo pipefail

MODEL_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
LLAMA_BIN="${LLAMA_BIN:-$HOME/llama.cpp/build/bin/llama-moe-trace}"
M="$MODEL_DIR/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"

cd "$MODEL_DIR"

MOE_TRACE_OUT=qwen38-code.csv "$LLAMA_BIN" -m "$M" \
  -ngl 99 -ncmoe 99 -fa 1 -c 4096 -n 512 \
  -p "Write a Python function that merges two sorted lists, with type hints and unit tests."

MOE_TRACE_OUT=qwen38-chat.csv "$LLAMA_BIN" -m "$M" \
  -ngl 99 -ncmoe 99 -fa 1 -c 4096 -n 512 \
  -p "Hey! Can you help me plan a relaxing weekend trip? I like hiking and good food."

cat qwen38-code.csv qwen38-chat.csv > qwen38-merged.csv
wc -l qwen38-*.csv
echo "Profile written to $MODEL_DIR/qwen38-merged.csv"
