# MFU Optimization: Qwen3-8B + DMax OPUT on TPU v5e-64

Summary of a MFU tuning pass on `tpu-v5e-64-eu` (europe-west4-b) for
`scripts/tpu_v6e_smoke.py` training **Qwen3-8B with full DMax OPUT**
(on_policy_ratio=0.5 rollout kept, block-diffusion attention mask,
synthetic data, random init, AdamW, aggressive `jax.remat`).

## Headline

| Config | MFU (attn-adj) | tok/s | step s | vs start |
|--|--|--|--|--|
| starting point | 2.9% | 6,570 | 10.0 | 1.0× |
| **final (EE)** | **14.7%** | **32,834** | **8.0** | **5.1×** |

All comparisons are at `MODEL_NAME=Qwen/Qwen3-8B`, `MAX_LEN=4096`,
`PEAK_TFLOPS_PER_CHIP=197` (12.6 PFLOPS bf16 peak on v5e-64),
preserved training semantics (full OPUT rollout — changing semantics
is separated below).

The reported MFU still **under-counts DMax work by ~2×** because
`tpu_v6e_smoke.py:312` credits only one forward at `seq=MAX_LEN`
while DMax actually runs fwd+bwd on `seq=2·MAX_LEN` plus a rollout
forward on `seq=2·MAX_LEN`. Real hardware MFU at EE is therefore
roughly **~29% of v5e bf16 peak** — i.e., the compiled graph reaches
about a third of MXU peak on these shapes.

## Optimization ladder (full OPUT semantics preserved)

| # | Change | MFU | tok/s | step s | Notes |
|--|--|--|--|--|--|
| 0 | Starting — TP=8 hardcoded, dense attn, B=16 L=4096 | 2.9% | 6,570 | 10.0 | baseline |
| 1 | `TP=2 × FSDP=32` mesh (P) | 7.2% | 16,019 | 8.17 | mesh re-shaping freed HBM; larger matmuls; B=32 fit |
| 2 | `splash_attention` for block-diffusion mask (Z) | 8.9% | 19,809 | 6.61 | replaced dense `jax.nn.dot_product_attention` on 2L×2L mask |
| 3 | Scale to B=64 using splash's O(seq) attn memory (BB) | 9.2% | 20,469 | 12.8 | doubled batch — was OOM at B=64 with dense attn |
| 4 | **Tune splash `BlockSizes` to 512 + `use_fused_bwd_kernel=True` (EE)** | **14.7%** | **32,834** | **8.0** | kernel-level tile + bwd fusion — biggest single win |

### Why each step helped

1. **TP=2 × FSDP=32 mesh** — `tpu_v6e_smoke.py` hard-coded `TP = 8`.
   On v5e-64's 8×8 torus that forces FSDP=8, so optimizer state shards
   only 8-way and B must be a multiple of 8. Dropping TP to 2 (with
   `mesh_utils.create_device_mesh(..., allow_split_physical_axes=True)`
   because `(32, 2)` doesn't factor onto 8×8) bumps FSDP to 32, which
   (a) makes the per-chip optimizer/grad/param footprint ~4× smaller
   and (b) enlarges the matmul size per chip (hidden/TP=2048 vs 512),
   better amortizing MXU tile overhead.

2. **splash_attention for the block-diffusion mask** — the DMax path
   was silently **not** using flash attention: `dllm_jax/models.py:367`
   gated flash on `attention_mask is None`, and DMax always passes a
   non-None block-diffusion mask. The fallback was a dense
   `jax.nn.dot_product_attention` over the full 2L×2L mask. Switched
   to `jax.experimental.pallas.ops.tpu.splash_attention`, with the
   block-diffusion pattern wrapped as `NumpyMask` inside a
   `MultiHeadMask([mask] * heads_per_tp)`. Splash runs a block-sparse
   flash that skips masked regions and never materializes the score
   matrix.

3. **Scale to B=64 with splash** — dense attention's 2L×2L score
   materialization was the reason B=64 L=4096 OOMed previously
   (R: 28.84 GB > 15.75 GB). splash is O(seq), so B=64 fit and the
   mesh could amortize its fixed costs over 2× the tokens.

4. **Tune splash tile sizes + fused bwd** — the single biggest kernel
   change. `splash_attention_kernel.BlockSizes` default is
   `block_q=block_kv=block_kv_compute=128` with a separate `dkv` and
   `dq` backward pass (`use_fused_bwd_kernel=False`). 128×128 is much
   smaller than what v5e wants — at the per-chip shapes we run
   (8192 × 128), each tile barely saturates the MXU and the
   per-kernel launch overhead dominates. Bumping to 512 (8× more
   work per launch) and enabling the fused bwd kernel (dkv + dq in a
   single pass) drops step time 12.8 s → 8.0 s (−37%).

## Knobs added

All exposed as env vars on `scripts/tpu_v6e_smoke.py`:

| Env | Default | Effect |
|--|--|--|
| `TP` | 8 | tensor-parallel axis size; 2 or 4 usually better for v5e-64 |
| `SPLASH_BLOCK` | 512 | splash tile size (was 128 default) |
| `SPLASH_FUSED_BWD` | 1 | enable fused `dkv`/`dq` backward kernel |
| `DMAX_ON_POLICY_RATIO` | 0.5 | skips rollout fwd when 0 (see ablation below) |

## Peak config (EE) — reproduce

```bash
RUN_NAME=qwen3-8b-dmax-v5e-ee \
MODEL_NAME=Qwen/Qwen3-8B DATASET=tinystories \
MAX_LEN=4096 GLOBAL_BATCH=64 \
TP=2 \
SPLASH_BLOCK=512 SPLASH_FUSED_BWD=1 \
DMAX_ENABLE=1 DMAX_ON_POLICY_RATIO=0.5 \
DMAX_NOISE_LOW=0.75 DMAX_NOISE_HIGH=0.75 DMAX_BLOCK_SIZE=32 \
PEAK_TFLOPS_PER_CHIP=197 \
NUM_STEPS=0 NUM_EPOCHS=3 WANDB_LOG=1 \
python3 scripts/tpu_v6e_smoke.py
```

## What did NOT help

| Tried | Result |
|--|--|
| XLA async-collective flags (`xla_tpu_enable_async_collective_fusion`, etc.) | 0.0 pp — XProf showed comm is only ~14% of step time, not the bottleneck. |
| Longer seq at same batch (`L=7168`, `L=8192`) at TP=2 | L=7168 was slower (tok/s dropped); L=8192 OOMed (program 13.24 GB > 12.63 GB free — splash mask 16384² too big). |
| Bigger batch past B=64 at TP=2 (`B=96`, `B=128`) | OOM — program size exceeds HBM. |
| `OPTIMIZER=adafactor` at B=64 (bf16 factored opt state) | OOMed anyway — activations, not optimizer state, are the blocker at B=64 with splash defaults. |
| Skipping rollout forward unconditionally in DMax path | Changes OPUT semantics to off-policy-only, not a valid DMax result (see ablation table). |

## XProf breakdown (captured on a pre-tuned splash config)

Sampled steady-state steps 4–6, 4 chips on worker 0:

| Bucket | % of device time |
|--|--|
| fusion (matmul + elementwise epilogues) | 77% |
| while loops (remat recompute) | 7.5% |
| collective-permute | 6.0% |
| all-gather (FSDP) | 5.9% |
| all-reduce | 2.5% |
| copy / reshape / reduce | <1% |

So comm is ~14% combined (not bound) and MXU utilization inside the
fusions is the real ceiling. The splash tile tuning in step 4 above
attacks exactly that — bigger tiles raise per-fusion MXU efficiency.

## Ablation — non-OPUT configs (throughput-only)

These change training semantics and are **not** DMax OPUT results;
kept for attribution.

| Tag | Config | MFU | tok/s | Why reported |
|--|--|--|--|--|
| I | plain MDLM (DMax off), B=8 L=4096 TP=8 | 6.7% | 14,960 | Confirms the 2.9% starting ceiling is really ~6.7% HW × DMax accounting factor. |
| W | DMax TP=2 B=32 L=4096, `on_policy_ratio=0` + rollout fwd guard | 8.4% | 18,874 | Demonstrates the rollout fwd was ~⅓ of compute — but removing it drops OPUT semantics. |
| AA | W + splash attention | 10.1% | 22,605 | Same, also not OPUT. |

The W/AA rollout-skip patch sits behind a `DMAX_ON_POLICY_RATIO > 0`
guard in `loss_fn` — at the default 0.5 it's a no-op and OPUT runs
normally.

## Code changes (all uncommitted, applied on the TPU)

**`scripts/tpu_v6e_smoke.py`**:
- Line 87: `TP = int(os.environ.get("TP", "8"))` + split-axes mesh fallback.
- After flash install: build `_block_diffusion_mask_numpy` once at init,
  wrap in `MultiHeadMask`, construct splash with tuned `BlockSizes`,
  register as `dllm_models._MASKED_FLASH_ATTN_FN` (shard_map'd over the
  fsdp×tp mesh with vmap over batch).
- `loss_fn`: rollout forward now guarded by `DMAX_ON_POLICY_RATIO > 0`.
- Profiler hook env: `JAX_PROFILE_DIR`, `JAX_PROFILE_START_STEP`,
  `JAX_PROFILE_STEPS`.

**`dllm_jax/models.py`**:
- New `_MASKED_FLASH_ATTN_FN` global. `_attention` now routes to it
  when `attention_mask is not None`, keeping dense `dot_product_attention`
  as the final fallback.

## What's next if you want to push past 14.7%

In decreasing ROI:

1. **Fix the DMax MFU formula** — purely cosmetic but the headline
   number should reflect reality. `tpu_v6e_smoke.py:312`:
   ```python
   dmax_mult = 2 if DMAX_ENABLE else 1
   TRAIN_FLOPS_PER_TOKEN_DENSE = 6 * EST_PARAMS * dmax_mult
   TRAIN_FLOPS_PER_TOKEN_ATTN  = 12 * layers * MAX_LEN * H * dmax_mult
   ```
   Reported would go 14.7% → ~29%.
2. **Fused SwiGLU Pallas kernel** — merges
   `down_proj(silu(gate_proj) * up_proj)` into one kernel. Avoids
   materializing the `(B, 2L, intermediate)` activation in HBM.
   Mostly reduces the 7.5% remat `while` cost. Estimated +1-3 pp MFU.
   Multi-day build.
3. **Looser remat policy** — `save_only_these_names(["qkv", "gate_up"])`
   with `checkpoint_name(...)` in the model. Would cut remat
   recompute (~7.5% today) at the cost of HBM — risky given we're
   near the B=64 ceiling.
4. **Architectural: fold `[noised; clean]` into batch dim** instead of
   seq dim. Halves attention quadratic cost and the activation
   footprint. Requires rewriting the block-diffusion mask to run as
   two separate per-stream attentions. Biggest potential lever but
   a real rewrite.

## Artefacts

- `/tmp/sweep_results.md` — full sweep table (all configs, incl. OOMs).
- `/tmp/xprof_summary.md` — XProf trace analysis + top ops.
- `/tmp/xprof-P/plugins/profile/*/` — raw xplane.pb + trace.json.gz
  for TensorBoard.
