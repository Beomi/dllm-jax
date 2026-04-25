# Inference Optimization: DMax + Qwen3-8B on TPU v5e-64

Summary of an inference tuning pass on `tpu-v5e-64-us` (us-central1-a)
for `scripts/tpu_infer.py` generating from a
tinystories-trained DMax checkpoint
(`dmax-8b-tinystories-forward8k-3epoch-.../checkpoint_41400`).

## Headline

| Config | impl | attn | KV cache | median s | tok/s | nfe | gen tokens | quality |
|--|--|--|--|--|--|--|--|--|
| A (pre-patch) | fast | dense | — | 142.33 | 7.2 | 1028 | 1024 | **GARBAGE** |
| **B (ship)** | **fast** | **splash** | — | **40.86** | **25.1** | 128 | 175 | ✅ coherent, EOS |
| C | kv_fast | dense (fp32) | fp32 | 52.90 | 19.4 | 95 | 143 | ✅ coherent, EOS |
| D | kv_fast | dense (fp32) | bf16 | 52.12 | 19.6 | 95 | 143 | ✅ bit-identical to C |

All runs at `Qwen3-8B`, `PROMPT='Once upon a time'`, `GEN_LENGTH=1024`,
`BLOCK_LENGTH=32`, `STEPS=32`, `TOP_K=3`, `THRESHOLD=0.5`,
`CONFIDENCE_STOP=0.9`, `TP=8`, greedy (`TEMPERATURE=0`), 1 warmup + 2
measured runs, median.

**fast+splash (B) is 3.49× faster than the dense-fast baseline AND
fixes a latent quality bug.** It's the recommended default for this
model family at these shapes.

**bf16 KV cache (C→D) is a wash: 1.5% faster, byte-identical output.**
At `B=1 × q=32 × k=1152` the attention isn't bandwidth-bound so the
cache dtype doesn't matter. The flag is still exposed (`INFER_KV_DTYPE`)
so you can fall back to fp32 if you need the extra precision.

## The latent bug in fast+dense (Config A)

The `fast` path under `jax.nn.dot_product_attention` with a
block-causal mask at `total_length=1056` (prompt=4 + gen=1024 rounded
up to 33×32 blocks) produces degenerate output:

> …Lily was happy to help and she went to the laundry room.
> When she got there, she saw a big pile of clothes. She started to
> fold them and put them away. She was very careful and did a
> her....,,,,,, and,,,,,,,,,,,,,,,,,,,,,,......  *(900+ more tokens of
> commas and periods, runs out max_nfe)*

First ~80 tokens look normal; then the generation collapses into
punctuation filler and never reaches EOS, running to `nfe=1024`. The
same model, same prompt, same hyperparams under splash attention
produces a clean TinyStory. `kv_fast` (C, D) also produces clean
output. So the bug is specific to the
`fast + dot_product_attention + total_length=1056` combination.

Likely cause: numerical reduction order in the dense kernel at that
non-128-aligned sequence length pushes the intermediate denoising
trajectory into a repetition attractor. Splash's block-sparse kernel
uses different reduction order AND my integration pads `num_blocks`
so `total_length=1152` (128-aligned), so splash dodges the trap both
numerically and by shape.

## Code changes (all on branch `splash_attn`)

### `dllm_jax/dmax.py`

- `install_block_causal_splash_for_inference(mesh, num_heads, total_length, block_length, splash_block=None)`
  — builds `NumpyMask(block_q >= block_kv)`, wraps as `MultiHeadMask`,
  constructs a forward-only `make_splash_mha_single_device`, and
  registers it as `dllm_jax.models._MASKED_FLASH_ATTN_FN` under a
  `shard_map(mesh, in_specs=(P(None, 'tp', None, None),)*3)`. Auto-
  picks the largest splash tile that is a multiple of 128 and divides
  `total_length`.
- `_infer_mesh_from_model(model)` — walks `nnx.state(model)` leaves and
  unwraps `nnx.Variable` objects to read `.sharding.mesh`. The naive
  `jax.tree_util.tree_leaves(model)` returns Variable wrappers, not
  raw arrays, so the mesh lookup needs this extra hop.
- `_try_autoinstall_block_causal_splash(model, total_length, block_length)`
  — gated by `INFER_SPLASH=1`. Called from `dmax_generate_spd_fast`
  right before the `@nnx.jit` trace.
- `dmax_generate_spd_fast`: when `INFER_SPLASH=1`, pads `num_blocks`
  up to `128 // block_length` multiple so `total_length % 128 == 0`
  (TPU MXU lane alignment). A few extra blocks of compute, correct
  output.
- `_resolve_kv_cache_dtype(model)` for `dmax_generate_spd_kv_fast`.
  Default changed from `jnp.float32` to the model's compute dtype
  (bf16). `INFER_KV_DTYPE=fp32` reverts.

### `dllm_jax/models.py`

- `checkpoint_name` markers on `q`/`k`/`v` (post-RoPE) in
  `SelfAttention._project_qkv` and on the `gate*up` product in
  `DenseMLP.__call__`. Inert unless a named remat policy selects them
  (see `REMAT_POLICY` in `tpu_train.py`) — added for the training
  path but harmless for inference.

### `scripts/tpu_infer.py`

- `WARMUP_RUNS` + `MEASURED_RUNS` env vars, median reporting.
- `TEMPS="0.0,0.3,0.5,..."` env var runs a temperature sweep in a
  single process (shares the restore cost).

## Env knobs (on `scripts/tpu_infer.py`)

| Var | Default | Effect |
|--|--|--|
| `INFER_IMPL` | `fast` | `fast` / `kv_fast` / `legacy`. Use `fast` with splash for best throughput. |
| `INFER_SPLASH` | `0` | `1` enables splash kernel on the `fast` block-causal mask. Requires sharded model. |
| `INFER_SPLASH_BLOCK` | `512` | Splash tile size. Auto-adjusts to the largest multiple of 128 that divides `total_length`. |
| `INFER_KV_DTYPE` | `bf16` | KV-cache dtype for `kv_fast`. `fp32` restores the pre-patch precision path (no measurable speed gain). |
| `WARMUP_RUNS` | `0` | Throwaway generates before the measured ones. Each pays compile; helps stabilize measurements across seeds. |
| `MEASURED_RUNS` | `1` | Number of timed generates; reports median. |
| `TEMPS` | — | Comma-separated temperatures for a single-process sweep (e.g. `0.0,0.5,1.0`). |

## Temperature sweep (fast+splash, seed=42)

| temp | median s | tok/s | nfe | gen | quality |
|--|--|--|--|--|--|
| **0.0 (greedy)** | 41.28 | **24.8** | 128 | 175 | ✅ full coherent story, EOS |
| 0.3 | 163.19 | 6.3 | 1130 | 1024 | ~80 good tokens, then repeating `, the the to..` |
| 0.5 | 165.22 | 6.2 | 1146 | 1024 | ~80 good tokens, then repeating `the with. They...` |
| 0.7 | 164.83 | 6.2 | 1144 | 1024 | ~110 good tokens, then repeating `lesson heron with and..` |
| 1.0 | 76.70 | 13.4 | 416 | 382 | ~80 good tokens, then gibberish; random EOS mid-way |
| 1.5 | 166.04 | 6.2 | 1152 | 1024 | ~20 good tokens, then multilingual soup (`官方微信荤`, `훙`, `эту`, `ซึ่งเป็น`) |

### Findings

1. **Only greedy (temp=0) is usable** for this checkpoint on this
   model. Any stochastic temperature collapses the block-diffusion
   denoising into repetition after 20–110 tokens.
2. The `nfe` column reflects whether the block-level convergence
   checks (`all_confident ≥ 0.9`, `same_as_previous`) fire. Greedy
   converges fast (≈4 steps/block avg); stochastic sampling defeats
   both checks and runs out max `nfe`.
3. At `temp=1.5` the sampler pulls rare tokens from Qwen3's
   multilingual tail (Chinese / Korean / Thai / Russian / Spanish).
4. **Top-p / top-k sampling is not implemented.** Without them, any
   non-greedy temp is unusable on this checkpoint. The existing
   `top_k` parameter is a different thing — a soft-embedding mix for
   the intermediate denoising states, not a sampling-time filter.

### Recommended fix for usable sampling

Add top-p (nucleus) sampling in `dllm_jax/dmax.py:_sample_x0`,
thread `TOP_P` env var through the inference script. Rough
implementation (~10 lines):

```python
if top_p is not None and 0.0 < top_p < 1.0:
    sorted_logits = jnp.sort(logits_f32, axis=-1)[..., ::-1]
    sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
    cum = jnp.cumsum(sorted_probs, axis=-1)
    cutoff = jnp.sum(cum < top_p, axis=-1, keepdims=True)
    threshold = jnp.take_along_axis(sorted_logits, cutoff, axis=-1)
    logits_f32 = jnp.where(logits_f32 < threshold, -jnp.inf, logits_f32)
```

Use the pre-truncation `active_probs` for the `THRESHOLD` /
`CONFIDENCE_STOP` early-exit machinery so nucleus truncation doesn't
artificially inflate confidence.

## What's next

In decreasing ROI:

1. **Top-p sampling** — unblocks any non-greedy generation. 20–30 min
   (patch `_sample_x0`, thread `TOP_P` through the generate variants
   + scripts + README). Required before shipping a user-facing
   interface with temperature.
2. **Splash attention in the `kv_fast` prefill** — easy half of the
   kv_fast splash integration. Prefill shape is static; only the
   per-block decode has a dynamic mask. Prefill is a sizeable
   fraction of long-prompt latency. ~30 min.
3. **Splash attention in the `kv_fast` decode** — requires either
   pre-compiling N splash kernels (one per block position) and
   dispatching via `jax.lax.switch`, or expressing the
   `k_pos < block_end` gate as a runtime segment mask that splash
   accepts. Benefit uncertain — at `q=32 × k=1152`, splash tiles are
   underfilled and per-tile launch overhead may dominate. Current
   benchmark already shows `fast+splash` beats `kv_fast+dense`, so
   this is low priority unless we scale to very long contexts.
4. **Share jit compile cache across temperature values** — pass
   `temperature` as a traced `jax.Array` rather than a closure
   constant so one compile serves all temps. Saves ~30 s per extra
   temp in a sweep. Cosmetic for benchmarks, not a user-facing
   improvement.

## Reproduce

### Single run

```bash
gcloud compute tpus tpu-vm ssh tpu-v5e-64-us --zone=us-central1-a --worker=all \
  --command="cd ~/dllm-jax && \
    PYTHONPATH=~/dllm-jax:\${PYTHONPATH:-} \
    RESUME_DIR=gs://dllm-jax-us-central1/checkpoints/<run_name> \
    RESUME_STEP=41400 \
    MODEL_NAME=Qwen/Qwen3-8B PROMPT='Once upon a time' \
    GEN_LENGTH=1024 BLOCK_LENGTH=32 STEPS=32 \
    TOP_K=3 THRESHOLD=0.5 CONFIDENCE_STOP=0.9 TP=8 \
    INFER_IMPL=fast INFER_SPLASH=1 INFER_SPLASH_BLOCK=512 \
    WARMUP_RUNS=1 MEASURED_RUNS=2 \
    python3 scripts/tpu_infer.py"
```

### Temperature sweep

Same as above but replace the last two lines with:

```
TEMPS=0.0,0.3,0.5,0.7,1.0,1.5 SEED=42 \
WARMUP_RUNS=0 MEASURED_RUNS=1 \
```
