# Inference

Full reference for [`scripts/tpu_infer.py`](../scripts/tpu_infer.py), the
multi-host inference entry point. See the [README](../README.md#infer) for
the quick-start example. For a deep dive on the splash-attention vs dense
trade-off and end-to-end timing, see
[`inference-optimization.md`](inference-optimization.md). For the design of
the KV-cache path, see [`kv-cache-design.md`](kv-cache-design.md).

## Three implementations

All three produce **byte-identical output at matching settings.** Pick one
based on shape and length:

| `INFER_IMPL=` | What it does | When to use |
|---|---|---|
| `fast` (default) | fixed-shape `fori_loop` with step- and block-level early breaks | short/medium gen, especially with `INFER_SPLASH=1` |
| `kv_fast` | KV-cached, decode only the active block per step | long gen (≥1024 tokens) |
| `legacy` | Python-loop reference path with host-side breaks | debugging only — slow on TPU |

## Splash attention

`INFER_SPLASH=1` swaps `jax.nn.dot_product_attention` for a Pallas splash
kernel matched to the block-causal mask. On Qwen3-8B at `GEN_LENGTH=1024`
it's **3.5× faster** *and* fixes a latent dense-kernel quality bug at
non-128-aligned sequence lengths.

| Variable | Meaning | Default |
|---|---|---|
| `INFER_SPLASH` | enable splash for the `fast` block-causal mask | `0` |
| `INFER_SPLASH_BLOCK` | splash tile size (auto-rounds to 128-multiple ≤ this) | `512` |
| `INFER_KV_DTYPE` | KV-cache dtype for `kv_fast`: `bf16` (2× HBM savings) or `fp32` | `bf16` |

## Multi-prompt and temperature sweeps

You can amortize the restore cost across many generations:

```bash
# Multiple prompts, one per line:
PROMPTS_FILE=./prompts.txt … python3 scripts/tpu_infer.py

# Temperature sweep with 1 warmup + 2 measured runs each:
PROMPT='Once upon a time' \
TEMPS='0.0,0.3,0.5,0.7,1.0' \
WARMUP_RUNS=1 MEASURED_RUNS=2 \
… python3 scripts/tpu_infer.py
```

## Inference env vars

| Variable | Meaning | Default |
|---|---|---|
| `RESUME_DIR` / `RESUME_STEP` | which checkpoint to restore | — / latest |
| `MODEL_NAME` | tokenizer + HF config source | `Qwen/Qwen3-8B` |
| `PROMPT` | input prompt | `Once upon a time` |
| `PROMPTS_FILE` | one prompt per line (overrides `PROMPT`; `#` lines = comments) | — |
| `GEN_LENGTH` | tokens to generate | `32` |
| `BLOCK_LENGTH` | DMax block size | `32` |
| `STEPS` | max denoising steps per block | `8` |
| `INFER_IMPL` | `fast` / `kv_fast` / `legacy` | `fast` |
| `FAST_BUCKET_LENGTH` | compile window for `fast`; ignored by `kv_fast` | `4096` |
| `THRESHOLD` | left-to-right confidence cutoff (math `0.5`, code `0.65`, others `0.9`–`0.95`) | `0.95` |
| `CONFIDENCE_STOP` | block-level early-exit confidence | `0.9` |
| `TOP_K` | soft-mix top-k (`3` is more coherent on undertrained ckpts) | `1` |
| `TEMPERATURE` | gumbel-max temperature (`0.0` = greedy) | `0.0` |
| `TEMPS` | comma-separated sweep, e.g. `0.0,0.3,0.5` | — |
| `SEED` | RNG seed (only used when `TEMPERATURE > 0`) | none |
| `WARMUP_RUNS` | throwaway generates before timing | `0` |
| `MEASURED_RUNS` | timed generates; reports median | `1` |
| `SUPPRESS_MASK_TOKEN` | force-disable mask logits at argmax | `0` |
| `MASK_TOKEN_ID` / `EOS_TOKEN_ID` | overrides | tokenizer default |
| `TP` | inference mesh's TP axis | `8` |
| `RESTORE_OPTIMIZER` | also restore optimizer state | `0` |
| `INFER_SPLASH` / `INFER_SPLASH_BLOCK` / `INFER_KV_DTYPE` | see above | `0` / `512` / `bf16` |

## Throughput

TPU v4-32, Qwen3-8B, `TOP_K=3`:

| Config | NFE | seconds |
|---|---|---|
| `fast`, GEN=128 | 114 | 32.1 |
| `kv_fast`, GEN=128 | 119 | 65.9 |
| `fast`, GEN=1024 | 1010 | 124.7 |
| `kv_fast`, GEN=1024 | 1043 | 76.6 |

TPU v5e-64, Qwen3-8B DMax, `TOP_K=3`, `GEN=1024` (full writeup in
[`inference-optimization.md`](inference-optimization.md)):

| Config | seconds | tok/s | quality |
|---|---|---|---|
| `fast` (dense attention) | 142.3 | 7.2 | ⚠️ collapses into punctuation |
| **`fast` + splash** | **40.9** | **25.1** | ✅ coherent |
| `kv_fast` (fp32 KV) | 52.9 | 19.4 | ✅ coherent |
| `kv_fast` (bf16 KV) | 52.1 | 19.6 | ✅ bit-identical |

## Single-host CLI

For laptop / single-TPU-host work without GCS:

```bash
# Generate from a base model (no DMax training needed)
python scripts/dmax_generate.py \
  --model Qwen/Qwen3-0.6B \
  --prompt "Solve 37 * 48." \
  --gen-length 256 --block-length 32 --steps 32 \
  --threshold 0.5 --top-k 3 --impl fast

# Generate from a saved single-host trainer checkpoint
python scripts/dmax_generate_checkpoint.py \
  --checkpoint-dir ./out-dmax/checkpoint-1000 \
  --prompt "Solve 37 * 48." \
  --gen-length 256 --impl kv_fast
```

## Gotchas

**Output collapses into punctuation with `TOP_K=1`.** SPD feeds the previous
step's distribution back as a soft mix; at `TOP_K=1` a low-confidence
single token gives a bad signal. Try `TOP_K=3`.

**First-run generation is slow.** It includes model restore + XLA compile.
Subsequent runs with identical shapes hit the cached graph and are ~5× faster.
