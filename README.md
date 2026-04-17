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

### Regional GCS checkpoints for TPU runs

`scripts/tpu_v6e_smoke.py` saves Flax checkpoints every `CHECKPOINT_STEPS`
steps (default: 500). By default it detects the TPU zone, derives the
region, and writes to the matching bucket named `gs://${CHECKPOINT_BUCKET_PREFIX}-${region}`.
For example, a TPU in `us-east1-d` writes under
`gs://dllm-jax-us-east1/checkpoints/${RUN_NAME}`.

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

`dllm_jax` includes a JAX/Flax implementation of the DMax OPUT training loop
and soft parallel decoding path from [`czg1225/DMax`](https://github.com/czg1225/DMax).
OPUT uses fixed high-noise masking, block-diffusion attention, and optional
on-policy replacement of masked tokens with the model's own greedy predictions.

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
    ),
    train_dataset=dataset,
    data_collator=DMaxDataCollator(tokenizer=tokenizer, label_pad_token_id=-100),
)
trainer.train()
```

Generate with DMax soft parallel decoding:

```python
from dllm_jax import dmax_generate_spd

output = dmax_generate_spd(
    model,
    input_ids,
    tokenizer=tokenizer,
    gen_length=512,
    block_length=32,
    steps=32,
    threshold=0.95,
)
print(tokenizer.decode(output.generated_tokens[0], skip_special_tokens=True))
```

CLI entry points are available in `scripts/dmax_train.py` and
`scripts/dmax_generate.py`.

```bash
python scripts/dmax_train.py \
  --model Qwen/Qwen3-0.6B \
  --dataset Zigeng/DMax-LLaDA-2.0-Mini-Math-Trajectories \
  --max-steps 1000

python scripts/dmax_generate.py \
  --model Qwen/Qwen3-0.6B \
  --prompt "Solve 37 * 48." \
  --gen-length 256 --block-length 32 --steps 32
```

## Package Layout

```
dllm_jax/
├── models.py       # GenericDecoderLM, GenericEncoderLM, EditFlowModel
├── trainers.py     # MDLMTrainer, BD3LMTrainer, DreamTrainer, DMaxTrainer, EditFlowTrainer
├── configs.py      # ModelArguments, DataArguments, TrainingArguments, MDLMConfig, ...
├── dmax.py         # DMax SPD generation helpers
├── schedulers.py   # LinearAlpha, CosineAlpha, LinearKappa, CubicKappa, CosineKappa
├── data.py         # Collators (NoAttentionMaskWrapper, DreamSFTCollator, ...)
├── weights.py      # Torch-free safetensors -> NNX weight loader
└── utils.py        # resolve_with_base_env, parse_spec, get_default_logger, ...
```

## Sharding

| Model Size | Strategy    | Mesh Shape         | Notes                                     |
|------------|-------------|--------------------|-------------------------------------------|
| <= 3B      | 1D FSDP     | `(ndev,)` "fsdp"   | Direct TPU init, `P("fsdp", None)`        |
| 3B - 8B+   | 2D FSDP+TP  | `(8, 8)` fsdp,tp   | CPU init -> shard, `P("fsdp", "tp")`      |

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
