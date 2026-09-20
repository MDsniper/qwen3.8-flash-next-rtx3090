#!/usr/bin/env bash
# Build thecodacus/llama.cpp (perf branch) with CUDA for RTX 3090 (sm_86).
# Prereqs (Ubuntu/Debian): sudo apt-get install -y build-essential cmake curl git
# CUDA toolkit with nvcc + a working nvidia driver (tested: CUDA 13.2, driver 595.84).
set -euo pipefail

DEST="${1:-$HOME/llama.cpp}"

if [ ! -d "$DEST/.git" ]; then
  git clone --depth 1 https://github.com/thecodacus/llama.cpp.git "$DEST"
fi
git -C "$DEST" fetch --depth 1 origin perf
git -C "$DEST" checkout perf

cmake -S "$DEST" -B "$DEST/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build "$DEST/build" --config Release -j "$(nproc)" \
  --target llama-server llama-moe-trace

echo
echo "Built:"
ls -la "$DEST/build/bin/llama-server" "$DEST/build/bin/llama-moe-trace"
