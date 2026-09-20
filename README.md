# Qwen3.8-Flash-Next 177B on a single RTX 3090 (24 GB)

Deploy [Qwen3.8-Flash-Next](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) `UD-IQ3_XXS` (82 GB, 512 experts/layer, 10 active) on one NVIDIA RTX 3090 + 128 GB system RAM, using [thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp) (`perf` branch) for its two killer features:

- **MoE expert cache** — the 48 hottest routed experts per layer live in VRAM (~4.3 GB); decode runs hot experts on GPU, cold remainder on CPU, merged bit-exactly.
- **MTP speculative decoding** — the model's own multi-token-prediction head (`shared-Q8_0`) drafts, the main model verifies. Output unchanged, ~1.3–1.7× faster.

This is the exact configuration that was deployed and verified end-to-end on an RTX 3090 machine (see [Verified results](#verified-results)).

## Requirements

| | |
|---|---|
| GPU | NVIDIA RTX 3090 24 GB (sm_86); any ≥16 GB CUDA card should work with fewer `--moe-cache-slots` |
| RAM | 128 GB recommended (model mmaps 82 GB; 96 GB is the hard floor) |
| Disk | ~90 GB for model + draft head |
| OS | Linux, CUDA toolkit (`nvcc`) + driver (tested: CUDA 13.2, driver 595.84) |
| Misc | `git`, `cmake`, `build-essential`, `curl`, `python3` |

> **Why not stock llama.cpp?** Mainline has no MTP graph for the `qwen4exp` architecture, no cross-model tensor borrowing, and no `--spec-type draft-mtp`. You need the codacus `perf` fork (or [unslothai/llama.cpp releases](https://github.com/unslothai/llama.cpp/releases) / [ggml-org#28243](https://github.com/ggml-org/llama.cpp/pull/28243) for MTP alone — but those lack the expert cache).

## Quick start

```bash
git clone https://github.com/MDsniper/qwen3.8-flash-next-rtx3090.git
cd qwen3.8-flash-next-rtx3090

# 1. Build thecodacus/llama.cpp perf branch with CUDA (~10 min)
./scripts/build-llama-cpp.sh              # → ~/llama.cpp/build/bin/llama-server

# 2. Download model shards + MTP draft head (~85 GB, parallel + resumable)
./scripts/download-model.sh "$PWD"        # → ./UD-IQ3_XXS/ ./MTP/ (byte-exact size checks)

# 3. Generate the MoE routing profile (one-time, ~2 min warm / ~20 min cold)
./scripts/generate-moe-profile.sh "$PWD"  # → ./qwen38-merged.csv

# 4. Serve
./start-server.sh                         # → http://0.0.0.0:8080

# 5. Verify (health + MTP stats + real 47K-token deep-context recall test)
./scripts/verify.sh
```

Model files land next to the scripts, so the repo dir is self-contained:

```
qwen3.8-flash-next-rtx3090/
├── UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-0000{1,2,3}-of-00003.gguf   (82 GB)
├── MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf                            (2.8 GB)
├── qwen38-merged.csv                                                      (routing profile)
└── start-server.sh
```

## The server command

```bash
llama-server \
  -m UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
  --moe-cache-profile qwen38-merged.csv --moe-cache-slots 48 \
  -md MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  -ngld 0 --spec-type draft-mtp --spec-draft-n-max 1 \
  -ngl 99 --n-cpu-moe 99 --no-sched-async-cpu -t 6 \
  --load-mode mmap -fit off -fa on -ctk q8_0 -ctv q8_0 \
  -c 65536 -np 1 --cache-reuse 256 \
  -b 2048 -ub 512 --jinja --host 0.0.0.0 --port 8080
```

Flag-by-flag:

| Flag | Why |
|---|---|
| `-ngl 99 --n-cpu-moe 99` | attention/dense/embeddings on GPU; all 512×48 routed experts stay in system RAM |
| `--moe-cache-profile ... --moe-cache-slots 48` | VRAM-resident hot-expert pack (48 layers × 48 slots ≈ 4.3 GB on CUDA0) |
| `-md ...shared-Q8_0.gguf -ngld 0 --spec-type draft-mtp --spec-draft-n-max 1` | MTP head on CPU (it's tiny), drafts 1 token ahead; verification is exact |
| `--no-sched-async-cpu` | deterministic CPU/GPU split scheduling (drop it for +4–5% with spec decoding — see Tuning) |
| `-t 6` | 6 CPU threads for the cold-expert chain; leave cores for the OS on an 8-core part |
| `--load-mode mmap` | 82 GB never dirties RAM; page cache shared across restarts (second load ≈ 8 s) |
| `-fit off` | we know it fits; don't let the auto-fitter shrink our context |
| `-fa on -ctk q8_0 -ctv q8_0` | flash attention + 8-bit KV → 64K context fits beside the expert pack |
| `-c 65536 -np 1 --cache-reuse 256` | full 64K to a single slot (`cache-reuse` auto-disables under spec decoding — harmless warning) |
| `-b 2048 -ub 512 --jinja` | fast prefill batches; model's own chat template (reasoning-preserving) |

Confirm the expert cache engaged — start once with `-v` and look for:

```
init_moe_expert_cache: expert cache: 48 layers x 48 slots, 4347.66 MiB uploaded to CUDA0
```

(A second `no CPU-resident MoE layers` line refers to the tiny MTP draft model — expected with `-ngld 0`.) Without `-v`, llama-core INFO logs are suppressed and you won't see it.

## Verified results

RTX 3090 24 GB · Ryzen 7 2700X (8c) · 128 GB RAM · CUDA 13.2 · codacus `perf` @ `27c54b4`:

| Metric | Value |
|---|---|
| VRAM | 11.3 / 24 GB (model GPU part + 4.3 GB expert pack + 64K q8_0 KV + buffers) |
| RAM | ~90 GB resident (mmap page cache) |
| Decode (short prompt) | **15.5 t/s** with MTP drafts accepted 127/171 |
| Decode (4K context) | **15.3 t/s** |
| Decode (47K context) | 5.8 t/s (attention-bound at depth; measured under `-v` logging overhead) |
| Prefill | 22 t/s cold-start small batch · **114–128 t/s** at 4K–47K batches |
| Load time | ~7 s warm page cache / ~6 min cold |
| 64K proof | 47,022-token prompt; correct recall of a fact at ~token 42K; coherent output, no looping |

## Usage

OpenAI-compatible API:

```bash
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Hello!"}],
  "max_tokens": 500
}'
```

> **Reasoning model quirk:** the answer streams into `reasoning_content` first; `content` stays empty until thinking finishes. Budget ≥300 `max_tokens` for even trivial questions, or disable with `--no-reasoning-preserve`.

## Tuning (optional, on top of the verified config)

- **Drop `--no-sched-async-cpu`** — overlaps CPU cold-chain with GPU hot-chain; +4–5% with spec decoding per the fork README, outputs stay bit-identical.
- **More slots** — `--moe-cache-slots` is the main knob; VRAM headroom here supports well past 48 (README measured 24.4 t/s @ 56 slots). The pack is all-or-nothing: an oversized request logs `pack allocation failed - expert cache disabled` and falls back to baseline. Leave ~900 MB VRAM free beyond the pack.
- **Context vs slots** — KV grows with `-c`; compressing KV (`-ctk/-ctv`) frees VRAM that converts directly into slots.
- **q4** — `UD-IQ4_XS` (~100 GB) also fits a 128 GB box with `--n-cpu-moe 99`; expect a small decode-speed cost from wider RAM expert traffic. Not part of the verified config here.
- **Regenerate the profile** if your workload changes character entirely; a wrong-workload profile still helps (+28% worst case measured) but a merged code+chat profile recovers nearly all of the win.

## Troubleshooting

| Symptom | Fix |
|---|---|
| No `init_moe_expert_cache` line | run once with `-v` — core INFO is suppressed otherwise; then check the warnings below |
| `cannot open profile '...'` | start from the repo dir (`start-server.sh` cds for you) |
| `pack allocation failed` | warning names the max slots that fit — set that minus headroom |
| OOM at context creation | drop a few slots or shrink `-c`; the pack fits but KV/compute allocate after |
| `cache_reuse is not supported` warning | expected with spec decoding; flag is ignored, everything else works |
| Slow first load | cold mmap of 82 GB; subsequent loads reuse page cache (~7 s) |

## Credits

- Model: [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (Unsloth Dynamic IQ3_XXS + MTP heads)
- Fork: [thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp) `perf` branch — MoE expert cache, MTP spec decoding, qwen4exp support
- Also see: [OptLlama wiki](https://github.com/generelschwerz/llama.cpp/wiki)

## License

[MIT](LICENSE) — matching [llama.cpp](https://github.com/ggml-org/llama.cpp) and the [codacus fork](https://github.com/thecodacus/llama.cpp), which this repo's scripts build and configure but do not redistribute. Model weights are fetched directly from [unsloth's Hugging Face repo](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) and remain under their own license.
