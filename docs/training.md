# Training

Full reference for [`scripts/tpu_train.py`](../scripts/tpu_train.py), the
multi-host training entry point. See the [README](../README.md#train) for the
quick-start example.

## Datasets

| `DATASET=` | Source | Use |
|---|---|---|
| `tinystories` | `roneneldan/TinyStories` (streamed) | small-scale sanity / smoke |
| `wikipedia` | `wikimedia/wikipedia 20231101.en` (streamed) | pretrain-style packing |
| `openthoughts` | `open-thoughts/OpenThoughts-114k` (streamed) | SFT (chat-templated) |
| `parquet` | local `*.parquet` files via `DATASET_PATH=` | bring-your-own |
| `synthetic` | random ints | regression / shape testing |

`tinystories`, `wikipedia`, and `parquet` use **token-stream packing**:
documents are tokenized, joined by EOS, and chunked to `MAX_LEN`.
`openthoughts` is a chat-templated SFT set; by default it also packs the
full chat-templated text. Set `SFT_TRAIN_ON_ANSWERS_ONLY=1` to mask prompt
tokens with `-100` and supervise only assistant turns (per-example padded
batches; some rows >`MAX_LEN` are dropped).

To inspect what your SFT pipeline actually feeds the model:

```bash
python3 scripts/inspect_sft_data.py --model Qwen/Qwen3-8B --max-len 4096 --rows 2
```

## DMax / OPUT training

`DMAX_ENABLE=1` switches the script from MDLM-style noising to DMax's
high-noise, on-policy, block-diffusion objective.

| Variable | Meaning | Default |
|---|---|---|
| `DMAX_ENABLE` | turn on DMax (otherwise MDLM-style) | `0` |
| `DMAX_BLOCK_SIZE` | tokens per noising block | `32` |
| `DMAX_ON_POLICY_RATIO` | fraction of steps using model's own greedy preds | `0.5` |
| `DMAX_NOISE_LOW` / `DMAX_NOISE_HIGH` | uniform noise range | `0.75` / `0.75` |

When DMax is on, the script also installs a Pallas **splash attention**
kernel matched to the block-diffusion mask:

| Variable | Meaning | Default |
|---|---|---|
| `SPLASH_BLOCK` | tile size (must be a multiple of 128 dividing `MAX_LEN`) | `512` |
| `SPLASH_FUSED_BWD` | fused backward kernel | `1` |
| `DISABLE_SPLASH_ATTN` | fall back to dense attention (debugging) | `0` |

## Optimizer

Both AdamW and Adafactor are wired up:

| Variable | Meaning | Default |
|---|---|---|
| `OPTIMIZER` | `adamw` or `adafactor` | `adamw` |
| `PEAK_LR` | peak learning rate (post-warmup) | `1e-4` |
| `WARMUP_STEPS` | linear ramp from 0 → `PEAK_LR` | `5` |

> ⚠️ **AdamW is the recommended default** for diffusion-LM training on
> pretrained init. Adafactor's `scale_by_param_block_rms` misbehaves on
> bidirectional objectives over causal-LM weights — loss descends and then
> climbs back to ~12 around step 60 on Qwen3-8B for both MDLM and DMax/OPUT.
> Use Adafactor only when HBM forces it (e.g. 128k context on a single
> chip), and budget the divergence risk accordingly.

## Learning-rate schedule

By default the schedule is *constant after warmup*. To decay:

| Variable | Meaning | Default |
|---|---|---|
| `LR_SCHEDULE` | `constant` or `cosine` | `constant` |
| `LR_DECAY_STEPS` | cosine: total decay steps after warmup | `0` |
| `LR_DECAY_ALPHA` | cosine: final LR = `PEAK_LR * alpha` | `0.1` |

## Mask-token warm start

DMax reuses a token id as the MASK token (default `vocab_size - 1`, an
*untrained* reserved slot on Qwen3). Without intervention, its row in the
embedding matrix is essentially random — observed to drift the noised
forward toward uniform predictions around step 150.

To seed the mask row with the **mean** of all other input/output embedding
rows:

| Variable | Meaning | Default |
|---|---|---|
| `MASK_EMBED_INIT` | `mean` (warm start) or `none` (skip) | `mean` |
| `MASK_TOKEN_ID` | which row to seed | `vocab_size - 1` |

> ⚠️ On **v5e-64**, set `MASK_EMBED_INIT=none` for now. The pre-shard
> `.at[].set()` on replicated embed + lm_head leaks ~9.5GB HBM and blocks
> the training program from loading.

You can also point at a pretrained Qwen3 special token like
`MASK_TOKEN_ID=151662` (`<|fim_pad|>`) to inherit "fill in the missing
piece" semantics for free.

## Memory & throughput knobs

| Variable | Meaning | Default |
|---|---|---|
| `REMAT_POLICY` | gradient checkpointing policy | `nothing_saveable` |
| `MAX_LEN` | context length | `16384` |
| `GLOBAL_BATCH` | global batch (across all chips) | `8` |
| `PEAK_TFLOPS_PER_CHIP` | for MFU calc; v4=275, v5e=197, v5p=459, v6e=918 | `918` |

The throughput methodology is in [`mfu-optimization.md`](mfu-optimization.md).
Reference numbers land between ~38–48% MFU after splash + remat tuning.

## Checkpointing

Sharded Orbax DCP checkpoints write to GCS by default:

```
gs://${CHECKPOINT_BUCKET_PREFIX}-${region}/checkpoints/${RUN_NAME}/
├── checkpoint_500/
├── checkpoint_500/commit_success.txt
├── checkpoint_1000/
└── …
```

| Variable | Meaning | Default |
|---|---|---|
| `CHECKPOINT_STEPS` | save every N steps | `500` |
| `CHECKPOINT_KEEP` | retain the last N | `2` |
| `CHECKPOINT_BUCKET_PREFIX` | regional auto-detect prefix | `dllm-jax` |
| `CHECKPOINT_BUCKET` | fixed bucket (overrides prefix) | — |
| `CHECKPOINT_DIR` | full directory (overrides both) | — |
| `LOCAL_CHECKPOINT_DIR` | fallback if GCS not configured | `/tmp/dllm-jax-checkpoints` |
| `CHECKPOINT_ON_FINISH` | always write a final checkpoint | `0` |

The script also enables Orbax barrier shims by default
(`CHECKPOINT_ORBAX_SYNC_DIRS=1`, `CHECKPOINT_ORBAX_SIGNAL_FALLBACK=1`)
so distributed GCS writes use JAX multi-host barriers — needed on the
JAX-0.6.x / Orbax-0.11.x TPU VM stack.

## Resume

Resuming is one variable: point at the parent directory and optionally pin
a step.

```bash
RESUME_DIR=gs://dllm-jax-us-east1/checkpoints/old-run \
RESUME_STEP=0 \
… python3 scripts/tpu_train.py
```

`RESUME_STEP=0` (default) scans `commit_success.txt` markers and picks the
latest committed step. Set `RESUME_STEP=<N>` to pin a specific one.

### Three resume modes

| Scenario | Settings |
|---|---|
| Continue training, same data, same hardware | `RESUME_DIR=…` (defaults are correct) |
| **Continue onto a new dataset** (e.g. switch to `openthoughts`) | `RESUME_DIR=…  RESUME_RESET_STEP=1` |
| **Pretrain → SFT** (different optimizer family or lr) | `RESUME_DIR=…  RESUME_RESTORE_OPTIMIZER=0  RESUME_RESET_STEP=1` |

| Variable | Meaning | Default |
|---|---|---|
| `RESUME_DIR` / `RESUME_FROM` | parent dir containing `checkpoint_<step>/` | — |
| `RESUME_STEP` | specific step (`0` = latest) | `0` |
| `RESUME_RESTORE_OPTIMIZER` | `0` = restore weights only, fresh optimizer | `1` |
| `RESUME_RESET_STEP` | `1` = zero global_step / epoch counters | `0` |

Switching hardware (v4 → v5e, v6e → v4, …) is supported transparently by
orbax: the script re-shards on restore using the *current* mesh. The only
thing you need to update is `TP`.

## Sharding and hardware sizing

The 2D mesh is `(fsdp, tp)` with `fsdp = jax.device_count() // TP`.

| Hardware | Chips | Recommended `TP` | Mesh shape | Notes |
|---|---|---|---|---|
| TPU v4-32 | 16 | `8` | `(2, 8)` | validated for Qwen3-8B training and inference |
| TPU v5e-64 | 64 | `2` | `(32, 2)` | `TP=8` here forces FSDP=8 and ~4× optimizer-state HBM; prefer `TP=2` |
| TPU v6e-256 | 256 | `8` | `(32, 8)` | high MFU with splash + remat |
| Single-host (≤ 3B params) | any | `1` | `(N,)` 1D FSDP | `P("fsdp", None)` direct TPU init |

For models ≥ 3B you generally want CPU-first init
(`jax.default_device(jax.devices("cpu")[0])`), gradient checkpointing on
every transformer layer, and Pallas flash attention. The smoke script does
all three automatically.

## Gotchas

**Adafactor diverges around step 60.** See the optimizer note above —
default to AdamW unless HBM forces otherwise.

**Optimizer rebind after split/merge.** If you manually `nnx.split` and
`nnx.merge` the optimizer in a hand-written script, rebind `model` or
you'll get silent zero-progress:

```python
opt_gdef, opt_state = nnx.split(optimizer)
opt_state = jax.tree.map(fsdp_sharding, opt_state)
optimizer = nnx.merge(opt_gdef, opt_state)
model = optimizer.model  # CRITICAL — without this, grads land on a stale model
```

The built-in `trainers.py` and `tpu_train.py` already do this; only
relevant if you're rolling your own loop.

**flax 0.10.7 vs 0.12+.** Python 3.10 TPU VMs pin flax to 0.10.7. The
`_nnx_list = getattr(nnx, "List", list)` shim in `models.py` and the
positional `optimizer.update(grads)` call are deliberate compat choices.
Don't "modernize" them without testing on the older stack.

**Stale TPU checkout.** `ImportError: cannot import name '...'` usually
means the TPU copy is out of date. Re-`scp` `dllm_jax/` and the target
script.

**`gs://` / TensorFlow warnings on restore.** Harmless. Restore uses
Orbax's GCS client, not `tf.io.gfile`.
