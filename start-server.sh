#!/usr/bin/env bash
# Qwen3.8-Flash-Next 177B UD-IQ3_XXS — thecodacus/llama.cpp (perf branch)
# Verified configuration for NVIDIA RTX 3090 24GB + 128GB system RAM.
#
# What each flag group does:
#   --moe-cache-profile/-slots : keep the 48 hottest routed experts per layer in VRAM
#                                (~4.3 GB pack; decode runs hot experts on GPU, cold on CPU)
#   -md ... --spec-type draft-mtp : MTP speculative decoding head (~1.3-1.7x decode)
#   -ngl 99 --n-cpu-moe 99       : attention/dense on GPU, all MoE experts in system RAM
#   --load-mode mmap             : model pages stay in page cache, shared across restarts
#   -fa on -ctk q8_0 -ctv q8_0   : flash attention + 8-bit KV cache (64K ctx fits VRAM)
#   -np 1                        : single slot — all 64K context to one conversation
set -euo pipefail
cd "$(dirname "$0")/.."

exec "${LLAMA_SERVER:-$HOME/llama.cpp/build/bin/llama-server}" \
  -m UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  --moe-cache-profile qwen38-merged.csv --moe-cache-slots 48 \
  -md MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  -ngld 0 --spec-type draft-mtp --spec-draft-n-max 1 \
  -ngl 99 --n-cpu-moe 99 --no-sched-async-cpu -t 6 \
  --load-mode mmap -fit off -fa on -ctk q8_0 -ctv q8_0 \
  -c 65536 -np 1 --cache-reuse 256 \
  -b 2048 -ub 512 --jinja --host 0.0.0.0 --port 8080
