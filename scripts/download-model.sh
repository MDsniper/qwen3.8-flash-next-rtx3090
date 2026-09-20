#!/usr/bin/env bash
# Download Qwen3.8-Flash-Next UD-IQ3_XXS (82 GB, 3 shards) + MTP draft head (2.8 GB)
# from unsloth/Qwen3.8-Flash-Next-GGUF, with resume support and byte-exact size checks.
set -euo pipefail

DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
REPO="https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main"
mkdir -p "$DEST/UD-IQ3_XXS" "$DEST/MTP"

declare -A FILES=(
  ["UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf"]=10946624
  ["UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00002-of-00003.gguf"]=49567921344
  ["UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00003-of-00003.gguf"]=32382955968
  ["MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"]=2786568256
)

# parallel downloads, resumable
pids=()
for f in "${!FILES[@]}"; do
  have=$(stat -c%s "$DEST/$f" 2>/dev/null || echo 0)
  if [ "$have" -eq "${FILES[$f]}" ]; then
    echo "[skip] $f (already complete)"
    continue
  fi
  echo "[get ] $f"
  curl -sL -C - -o "$DEST/$f" "$REPO/$f" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

# verify
fail=0
for f in "${!FILES[@]}"; do
  have=$(stat -c%s "$DEST/$f" 2>/dev/null || echo 0)
  if [ "$have" -ne "${FILES[$f]}" ]; then
    echo "[FAIL] $f: expected ${FILES[$f]} bytes, got $have"
    fail=1
  else
    echo "[ok  ] $f"
  fi
done
exit $fail
