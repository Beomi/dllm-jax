# dllm-jax

A standalone **JAX backend for Diffusion Language Modeling (dLLM)** with **zero
PyTorch or CUDA dependency**. Designed for TPU training of MDLM, BD3LM, Dream,
and EditFlow objectives on pretrained HuggingFace checkpoints.

## Features

- **Torch-free weight loading** — `safetensors` + `huggingface_hub` + numpy, no `torch.load`
- **TPU-first** — Pallas flash attention via `shard_map`, 1D FSDP and 2D FSDP+TP
- **Pretrained init** — load Qwen3, Llama, and other HF causal LMs directly into Flax NNX
- **Five training objectives** — MDLM, BD3LM, Dream, DMax/OPUT, EditFlow
- **Clean API** — public exports, no stub boilerplate
- **DMax / OPUT end-to-end** — port of [`czg1225/DMax`](https://github.com/czg1225/DMax)
  training + Soft Parallel Decoding inference, with a KV-cached fast path
  matching reference's `cache='prefix'` setting

`transformers` is used only for `AutoConfig` / `AutoTokenizer` (works without torch).

## Prerequisites

### Google Cloud CLI (`gcloud`)

```bash
# Install gcloud CLI (see https://cloud.google.com/sdk/docs/install)
# After installation:
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# Create a GCS bucket for checkpoints (pick your TPU region):
gcloud storage buckets create gs://YOUR_BUCKET_NAME --location=us-east1
```

### Environment setup

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

Key variables in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `CHECKPOINT_BUCKET_PREFIX` | Prefix for regional buckets (`gs://{prefix}-{region}`) | `dllm-jax` |
| `CHECKPOINT_BUCKET` | Fixed bucket name (overrides regional auto-detection) | — |
| `WANDB_API_KEY` | Weights & Biases API key ([get one here](https://wandb.ai/authorize)) | — |
| `WANDB_LOG` | Enable W&B logging (`1` to enable) | `0` |
| `MODEL_NAME` | HuggingFace model ID | `Qwen/Qwen3-8B` |

See [`.env.example`](.env.example) for the full list.

> **Tip:** Instead of setting `WANDB_API_KEY` in `.env`, you can run `wandb login`
> on each TPU worker. Never commit credentials to the repository.

## Installation

```bash
pip install -e .

# TPU runtime
pip install -e '.[tpu]'

# Dev (pytest)
pip install -e '.[dev]'
```

Requires Python >= 3.10, `jax >= 0.4.20`, `flax >= 0.10.0`,
`orbax-checkpoint`, `gcsfs <= 2026.2.0`, `optax >= 0.2.0`.

### TPU VM packaging caveat

Some Python 3.10 TPU VM images have an older packaging stack. On those hosts,
`pip install -e '.[tpu]'` can fail with a missing `build_editable` hook, and
`pip install '.[tpu]'` can misread the project metadata as `UNKNOWN-0.0.0`
without installing dependencies. In that case, run from the synced checkout
with `PYTHONPATH` and install TPU dependencies explicitly:

```bash
python3 -m pip install --user -U 'jax[tpu]' \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
  'flax>=0.10.0,<0.11' orbax-checkpoint 'gcsfs<=2026.2.0' 'fsspec<=2026.2.0' \
  'optax>=0.2.0' numpy 'transformers>=4.40.0' safetensors \
  huggingface_hub datasets wandb
```

#### Verified TPU versions

The training and inference paths are validated on TPU v4-32 (`us-central2-b`)
with this stack — pin to these if `pip install '.[tpu]'` surfaces version
drift:

| Package | Version |
|---------|---------|
| Python | 3.10.12 |
| jax / jaxlib | 0.6.2 |
| libtpu | 0.0.17 |
| flax | 0.10.7 |
| optax | 0.2.8 |
| orbax-checkpoint | 0.11.34 |
| transformers | 5.5.3 |
| safetensors | 0.7.0 |
| datasets | 4.8.4 |
| gcsfs / fsspec | 2025.3.2 |
| huggingface_hub | 1.10.1 |
| numpy | 2.2.6 |

The 0.10.7 flax version forces the `_nnx_list = getattr(nnx, "List", list)`
compat shim noted under **Gotchas**; newer flax (0.12+) on JAX 0.7+ should
also work but hasn't been re-verified end-to-end on this repo.

### Regional GCS checkpoints for TPU runs

`scripts/tpu_v6e_smoke.py` saves sharded Orbax DCP checkpoints every
`CHECKPOINT_STEPS` steps (default: 500). By default it detects the TPU
zone, derives the region, and writes to a matching bucket named
`gs://${CHECKPOINT_BUCKET_PREFIX}-${region}`. For example, a TPU in
`us-east1-d` writes under `gs://dllm-jax-us-east1/checkpoints/${RUN_NAME}`.

```bash
PYTHONPATH=/path/to/dllm-jax \
  RUN_NAME=my-run-name \
  CHECKPOINT_STEPS=500 CHECKPOINT_KEEP=2 \
  MODEL_NAME=Qwen/Qwen3-8B DATASET=tinystories \
  MAX_LEN=16384 GLOBAL_BATCH=8 \
  NUM_STEPS=0 NUM_EPOCHS=3 WANDB_LOG=1 \
  python3 scripts/tpu_v6e_smoke.py
```

Override detection with `TPU_REGION=us-east1`, `CHECKPOINT_BUCKET=gs://...`,
or a full `CHECKPOINT_DIR=gs://...`. Hub uploads are only used for local
checkpoint directories; `gs://` is the durable distributed checkpoint target.
On TPU VM images with JAX 0.6.x and Orbax 0.11.x, the script enables
`CHECKPOINT_ORBAX_SYNC_DIRS=1` and `CHECKPOINT_ORBAX_SIGNAL_FALLBACK=1` by
default to use JAX multihost barriers for distributed GCS checkpoint writes.

## Quick Start

```python
from transformers import AutoTokenizer
from dllm_jax import build_model_from_pretrained, MDLMConfig, MDLMTrainer, LinearAlphaScheduler

model, config = build_model_from_pretrained("Qwen/Qwen3-0.6B", task="llada")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

trainer = MDLMTrainer(
    model=model,
    tokenizer=tokenizer,
    args=MDLMConfig(output_dir="./out", max_steps=1000, learning_rate=1e-4),
    train_dataset=dataset,
    data_collator=collator,
    scheduler=LinearAlphaScheduler(),
)
trainer.train()
```

## DMax / OPUT

`dllm_jax` includes a JAX/Flax port of the DMax training and inference
stack from [`czg1225/DMax`](https://github.com/czg1225/DMax):

- **OPUT training** — fixed high-noise masking, two-stream `[noised | clean]`
  layout with block-diffusion attention, per-step on-policy rollout that
  replaces masked tokens with the model's own greedy predictions, gradient
  accumulation support.
- **Soft Parallel Decoding (SPD) inference** with three implementations
  that produce byte-exact identical outputs at matching settings:
  - `dmax_generate_spd` — Python-loop reference path with host-side early
    breaks. Slow on TPU, useful for debugging.
  - `dmax_generate_spd_fast` — fixed-shape fori_loop compiled path; step-level
    and block-level early breaks via `jax.lax.while_loop`. Default for
    short/medium generations.
  - `dmax_generate_spd_kv_fast` — KV-cached variant matching reference's
    `cache='prefix'` path. Each step only projects K/V for the active block
    and attention runs over the cached prefix. ~1.6× speedup at 1024-token
    generation; overhead dominates at 128-token generation.
- **Reference knobs** — `top_k` (soft-mix top-k, reference default 1),
  `temperature` + gumbel sampling, `seed`, `threshold`, `confidence_stop`,
  `suppress_mask_token`, and post-EOS fill matching reference's `early_stop`.

### Training

```python
from dllm_jax import DMaxConfig, DMaxTrainer, DMaxDataCollator

trainer = DMaxTrainer(
    model=model,
    tokenizer=tokenizer,
    args=DMaxConfig(
        output_dir="./out-dmax",
        max_steps=1000,
        learning_rate=2e-6,
        noise_range_low=0.75,
        noise_range_high=0.75,
        on_policy_ratio=0.5,
        block_size=32,
        gradient_accumulation_steps=4,  # optional
    ),
    train_dataset=dataset,
    data_collator=DMaxDataCollator(tokenizer=tokenizer, label_pad_token_id=-100),
)
trainer.train()
```

### Inference

```python
from dllm_jax import dmax_generate_spd_fast, dmax_generate_spd_kv_fast

# Fast path (default for short/medium gen)
output = dmax_generate_spd_fast(
    model,
    input_ids,
    tokenizer=tokenizer,
    gen_length=512,
    block_length=32,
    steps=32,
    threshold=0.5,        # reference math eval default
    confidence_stop=0.9,  # reference block-level break
    top_k=3,              # soft mix aggregates top 3 candidates
    temperature=0.0,      # 0.0 = greedy; >0.0 = gumbel sampling
)
print(tokenizer.decode(output.generated_tokens[0], skip_special_tokens=True))

# KV-cache path (wins on long generations)
output = dmax_generate_spd_kv_fast(
    model, input_ids, tokenizer=tokenizer,
    gen_length=2048, block_length=32, steps=32,
    threshold=0.5, top_k=3,
)
```

`output.nfe` counts actual forward passes. For `fast` it matches
reference's `num_forwards`; for `kv_fast` it is `fast_nfe + num_active_blocks`
(the extra is the post-block hard-write pass that replaces soft K/V with
hard K/V in the cache — reference's cross-block update).

### CLI entry points

```bash
# Train
python scripts/dmax_train.py \
  --model Qwen/Qwen3-0.6B \
  --dataset Zigeng/DMax-LLaDA-2.0-Mini-Math-Trajectories \
  --max-steps 1000

# Generate (pretrained base model)
python scripts/dmax_generate.py \
  --model Qwen/Qwen3-0.6B \
  --prompt "Solve 37 * 48." \
  --gen-length 256 --block-length 32 --steps 32 \
  --threshold 0.5 --top-k 3 --impl fast

# Generate (from a saved trainer checkpoint)
python scripts/dmax_generate_checkpoint.py \
  --checkpoint-dir ./out-dmax/checkpoint-1000 \
  --prompt "Solve 37 * 48." \
  --gen-length 256 --impl kv_fast
```

### TPU multi-host inference

[`scripts/tpu_dmax_infer_checkpoint.py`](scripts/tpu_dmax_infer_checkpoint.py)
restores a distributed Orbax DCP checkpoint (GCS or local) on every worker,
reshards across the inference mesh (which may differ from the training
mesh), and runs `dmax_generate_spd_{fast,kv_fast,legacy}` end-to-end. All
configuration is via environment variables:

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME \
  --zone=$ZONE --worker=all \
  --command="cd ~/dllm-jax && \
    RESUME_DIR=gs://$BUCKET/checkpoints/$RUN_NAME \
    MODEL_NAME=Qwen/Qwen3-8B \
    PROMPT='Once upon a time' \
    GEN_LENGTH=1024 BLOCK_LENGTH=32 STEPS=32 \
    INFER_IMPL=kv_fast TOP_K=3 THRESHOLD=0.5 CONFIDENCE_STOP=0.9 \
    TP=8 \
    python3 scripts/tpu_dmax_infer_checkpoint.py"
```

Omit `RESUME_STEP` and the script scans `commit_success.txt` files under
`RESUME_DIR` and picks the latest committed step. Set `RESUME_STEP=<N>` to
pin a specific step.

#### Environment variables

| Variable | Meaning | Default |
|-|-|-|
| `RESUME_DIR` / `RESUME_FROM` | checkpoint parent directory (contains `checkpoint_<step>/`) | — (required) |
| `RESUME_STEP` | specific step to restore | latest committed |
| `MODEL_NAME` | tokenizer + HF config source | `Qwen/Qwen3-8B` |
| `PROMPT` | input prompt | `Once upon a time` |
| `GEN_LENGTH` | tokens to generate | `32` |
| `BLOCK_LENGTH` | DMax block size | `32` |
| `STEPS` | max denoising steps per block | `8` |
| `INFER_IMPL` | `fast`, `kv_fast`, or `legacy` | `fast` |
| `FAST_BUCKET_LENGTH` | compile window for `fast`; ignored by `kv_fast` | `4096` |
| `THRESHOLD` | left-to-right confidence cutoff — reference math eval `0.5`, code `0.65`, other benchmarks `0.9`–`0.95` | `0.95` |
| `CONFIDENCE_STOP` | block early-exit confidence | `0.9` |
| `TOP_K` | soft-mix top-k (reference default `1`; `3` is more coherent on undertrained ckpts) | `1` |
| `TEMPERATURE` | gumbel-max sampling temperature; `0.0` = greedy | `0.0` |
| `SEED` | RNG seed (only needed when `TEMPERATURE > 0`) | none |
| `SUPPRESS_MASK_TOKEN` | set `1` to force-disable mask-token logits during argmax | `0` |
| `MASK_TOKEN_ID` / `EOS_TOKEN_ID` | overrides for the model's mask / EOS id | tokenizer default |
| `TP` | tensor-parallel axis size; `fsdp` is derived as `jax.device_count() // TP` | `8` |
| `RESTORE_OPTIMIZER` | restore optimizer state (useful for resumed training, not inference) | `0` |

#### Measured throughput (TPU v4-32, Qwen3-8B, `TOP_K=3`)

| Config | nfe | generate_seconds |
|-|-|-|
| `fast`, GEN=128 | 114 | 32.1 |
| `kv_fast`, GEN=128 | 119 | 65.9 |
| `fast`, GEN=1024 | 1010 | 124.7 |
| `kv_fast`, GEN=1024 | 1043 | 76.6 |

At short generations `kv_fast` pays more XLA compile cost than it saves; at
`GEN=1024` the block-local forward wins (~1.6× faster than `fast`). `nfe`
for `kv_fast` is `fast_nfe + num_active_blocks`; the extra forwards are the
post-block hard-write that overwrites soft K/V with hard K/V in the cache
(the JAX analogue of reference's cross-block update).

#### Troubleshooting

- **Stale TPU checkout** — `ImportError: cannot import name 'dmax_generate_spd_kv_fast'` or similar means the TPU copy is out of date; re-`scp` `dllm_jax/` and the target script.
- **Orbax signal-contract hang on restore** — handled by default; the script flips `CHECKPOINT_ORBAX_SIGNAL_FALLBACK=1` when the JAX distributed client isn't initialized. Set the env var to `0` to disable the shim explicitly.
- **`gs://` / TensorFlow warnings** — harmless; restore uses Orbax's GCS client, not `tf.io.gfile`.
- **First-run generation is "slow"** — includes model restore + XLA compile. Subsequent runs with identical shapes hit the cached graph.
- **Output collapses into punctuation with `TOP_K=1`** — SPD feeds the previous step's distribution back as a soft mix; at `TOP_K=1` a low-confidence single token gives a bad signal. Try `TOP_K=3`.

The KV-cache design is documented in [`docs/kv_cache_design.md`](docs/kv_cache_design.md).
A narrative write-up of the whole end-to-end port — HF checkpoint →
OPUT training → KV-cached SPD on TPU — is in
[`docs/porting-dmax-to-tpu.md`](docs/porting-dmax-to-tpu.md).

## Checkpoints

Two save/load paths are supported depending on scale:

- **Single-host trainer checkpoints** (`DMaxTrainer.save_model`, etc.) write
  pickle files (`save_only_model=True`) or Flax training `Checkpoints` to a
  local directory. These pair with `restore_model_checkpoint` and the local
  `scripts/dmax_generate_checkpoint.py` script.
- **Distributed TPU DCP checkpoints** (Orbax `PyTreeCheckpointHandler` /
  `StandardCheckpointer`) are the durable save path for multi-host v4/v5/v6
  training via `scripts/tpu_v6e_smoke.py`. Checkpoints are sharded across
  workers, written directly to GCS (`gs://${CHECKPOINT_BUCKET_PREFIX}-${region}`),
  and committed with `commit_success.txt` markers. Resume and inference read
  the latest committed step via `latest_committed_gcs_step`.

  ```bash
  # Train with DCP checkpoints every 500 steps, keep last 2
  PYTHONPATH=$(pwd) \
    RUN_NAME=my-run CHECKPOINT_STEPS=500 CHECKPOINT_KEEP=2 \
    python3 scripts/tpu_v6e_smoke.py
  ```

  Inference from the resulting GCS checkpoints is covered above under
  [TPU multi-host inference](#tpu-multi-host-inference).

## Package Layout

```
dllm_jax/
├── models.py       # GenericDecoderLM (+ call_cached for KV cache), GenericEncoderLM, EditFlowModel
├── trainers.py     # MDLMTrainer, BD3LMTrainer, DreamTrainer, DMaxTrainer, EditFlowTrainer
│                   # (gradient accumulation, cached LR schedule)
├── configs.py      # ModelArguments, DataArguments, TrainingArguments, MDLMConfig, DMaxConfig, ...
├── dmax.py         # DMax SPD: dmax_generate_spd, dmax_generate_spd_fast, dmax_generate_spd_kv_fast
├── schedulers.py   # LinearAlpha, CosineAlpha, LinearKappa, CubicKappa, CosineKappa
├── data.py         # DMaxDataCollator, DreamSFTCollator, EditFlowCollator, NoAttentionMaskWrapper, ...
├── checkpoints.py  # restore_model_checkpoint (single-host pickle / Flax Checkpoints)
├── weights.py      # Torch-free safetensors -> NNX weight loader
└── utils.py        # resolve_with_base_env, parse_spec, get_default_logger, ...

scripts/
├── dmax_train.py                   # single-host DMax OPUT CLI
├── dmax_tinystories_train.py       # small-scale DMax sanity training
├── dmax_generate.py                # single-host SPD generation CLI (base model)
├── dmax_generate_checkpoint.py     # single-host SPD generation from a saved checkpoint
├── tpu_dmax_infer_checkpoint.py    # multi-host Orbax DCP restore + SPD generation
├── tpu_v4_32_train_3epoch.py       # multi-host DMax training on TPU v4-32
├── run_tpu_v4_32_3epoch.sh         # wrapper for the v4-32 training launch
└── tpu_v6e_smoke.py                # MDLM smoke trainer with DCP checkpointing

docs/
├── tpu_v4_32_ondemand_inference.md # runbook for TPU inference from GCS checkpoints
├── kv_cache_design.md              # design notes for the KV-cached SPD path
└── porting-dmax-to-tpu.md          # narrative write-up of the whole end-to-end port
```

## Sharding

| Model Size | Strategy    | Mesh Shape           | Notes                                     |
|------------|-------------|----------------------|-------------------------------------------|
| <= 3B      | 1D FSDP     | `(ndev,)` fsdp       | Direct TPU init, `P("fsdp", None)`        |
| 3B - 8B+   | 2D FSDP+TP  | `(ndev/tp, tp)` fsdp,tp | CPU init → shard, `P("fsdp", "tp")`   |

For example, TPU v4-32 with `TP=8` uses `(2, 8)`; TPU v5e-64 uses `(8, 8)`.
Large models additionally need CPU-first init
(`jax.default_device(jax.devices("cpu")[0])`), gradient checkpointing via
`jax.remat` on each transformer layer, and Pallas flash attention.

## Gotchas

**AdamW > Adafactor for pretrained-init MDLM.** Adafactor's
`scale_by_param_block_rms` misbehaves on bidirectional objectives over causal
LM weights — loss descends then climbs back after ~60 steps. Use
`optax.adamw(b1=0.9, b2=0.95, wd=0.01)` (the library default).

**Optimizer rebind after split/merge.** If you manually `nnx.split` and
`nnx.merge` the optimizer in a hand-written TPU script, rebind `model` to
avoid silent zero-progress:

```python
opt_gdef, opt_state = nnx.split(optimizer)
opt_state = jax.tree.map(fsdp_sharding, opt_state)
optimizer = nnx.merge(opt_gdef, opt_state)
model = optimizer.model  # CRITICAL
```

The built-in `trainers.py` never splits the optimizer, so it is safe.

**flax 0.10.7 vs 0.12+.** Python 3.10 TPU VMs pin flax to 0.10.7. `models.py`
uses a `_nnx_list = getattr(nnx, "List", list)` compat shim so the same code
runs on both. `optimizer.update(grads)` is called positionally (0.10.7 API).

## License

Apache-2.0
