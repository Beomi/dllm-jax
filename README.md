# dllm-jax

<img max-width="100%" alt="Generated Image April 17, 2026 - 11_54PM" src="https://github.com/user-attachments/assets/cd62941c-20ad-44db-aa1a-98b77b8d498a" />

A pure-JAX stack for **diffusion language modeling** on TPU.

No PyTorch, no CUDA. Loads pretrained HuggingFace checkpoints (Qwen3, Llama,
…) directly into Flax NNX, then trains and decodes them under five
diffusion objectives — **MDLM**, **BD3LM**, **Dream**, **DMax/OPUT**,
**EditFlow** — with multi-host TPU sharding, GCS-resumable Orbax
checkpoints, and a KV-cached fast inference path.

> 📝 **Background reading:**
> [dLLM into TPU: An End-to-End Diffusion LM Stack in Pure JAX](https://medium.com/@JunbumLee/dllm-into-tpu-an-end-to-end-diffusion-lm-stack-in-pure-jax-5fc33c840ebb)


## Concepts

A **diffusion language model (dLLM)** trains a transformer to *denoise*
masked sequences in parallel, instead of predicting tokens one-at-a-time.
At inference it can refine many positions per step, generating text in a
fraction of the autoregressive forward passes.

Five training objectives ship in this repo:

| Objective | What it does | Trainer / Generator |
|---|---|---|
| **MDLM** (Sahoo et al.) | Mask-and-denoise with a learnable α schedule | `MDLMTrainer` |
| **BD3LM** | Block-diffusion variant of MDLM | `BD3LMTrainer` |
| **Dream** | SFT-friendly diffusion with prefix conditioning | `DreamTrainer` |
| **DMax / OPUT** | High-noise, on-policy block-diffusion + Soft Parallel Decoding | `DMaxTrainer`, `dmax_generate_spd_*` |
| **EditFlow** | Edit-style flow matching | `EditFlowTrainer` |

The polish in this repo concentrates on **DMax** — training, inference,
and checkpointing are all wired up for multi-host TPU. The other four
objectives share the same model loading and sharding infrastructure.


## Installation

```bash
pip install -e .
pip install -e '.[tpu]'   # TPU runtime
pip install -e '.[dev]'   # pytest, etc.
```

Requires Python ≥ 3.10. End-to-end validation on TPU v4-32 / v5e-64 / v6e-256.

For multi-host TPU runs you also need `gcloud` CLI and a regional GCS
bucket (default `gs://dllm-jax-${region}`). For TPU VM packaging quirks
and the verified pin set, see [`docs/install.md`](docs/install.md).


## Configuration

Every script reads its config from environment variables. The single source
of truth is [`.env.example`](.env.example) — 15 sections covering model,
data, training, optimizer, sharding, DMax, SFT, perf, checkpointing, resume,
HuggingFace, W&B, profiling, and inference.

```bash
cp .env.example .env
$EDITOR .env
```

Five variables to know on day one:

| Variable | Meaning | Default |
|---|---|---|
| `MODEL_NAME` | HF model ID for tokenizer + config + initial weights | `Qwen/Qwen3-8B` |
| `DATASET` | `tinystories`, `wikipedia`, `openthoughts`, `synthetic`, `parquet` | `tinystories` |
| `RUN_NAME` | name written under `${CHECKPOINT_DIR}/${RUN_NAME}` and to W&B | auto |
| `WANDB_LOG` | `1` to stream loss/MFU to W&B | `0` |
| `TP` | tensor-parallel axis size | `8` |

> 🔐 Never commit `WANDB_API_KEY` or `HF_TOKEN`. `.gitignore` excludes
> `.env`, `.env.local`, and `.env.deploy`. You can also `wandb login` and
> `huggingface-cli login` once per worker and skip the env vars.


## Train

The production training entry point is
[`scripts/tpu_train.py`](scripts/tpu_train.py). It runs on **v4-32, v5e-64,
v6e-256** and handles distributed init, 2D sharding, GCS DCP checkpoints,
W&B, MFU logging, and DMax masking.

A first run, end-to-end, from any TPU worker:

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all \
  --command="cd ~/dllm-jax && \
    PYTHONPATH=~/dllm-jax \
    RUN_NAME=hello-dmax \
    MODEL_NAME=Qwen/Qwen3-0.6B \
    DATASET=tinystories \
    DMAX_ENABLE=1 \
    MAX_LEN=4096 GLOBAL_BATCH=8 \
    NUM_STEPS=20 WARMUP_STEPS=5 \
    PEAK_LR=1e-4 OPTIMIZER=adamw \
    WANDB_LOG=1 \
    python3 scripts/tpu_train.py"
```

In the first ~3 minutes you'll see HF download → weight sharding →
optimizer init → W&B URL → step-by-step loss lines. Successful exit:

```
[Worker 0] step=20 loss=… mfu=…
[Worker 0] training complete in <N>s
```

For the full reference — datasets, DMax/OPUT knobs, optimizer choice (and
why **AdamW**, not Adafactor), LR schedule, mask-token warm start, memory
tuning, checkpointing, three resume modes, sharding tables per hardware,
and gotchas — see [`docs/training.md`](docs/training.md). Throughput
methodology is in [`docs/mfu-optimization.md`](docs/mfu-optimization.md);
runs land at ~38–48% MFU after splash + remat tuning.


## Infer

The multi-host inference entry point is
[`scripts/tpu_infer.py`](scripts/tpu_infer.py). It restores a sharded DCP
checkpoint, re-shards onto the inference mesh (which can differ from the
training mesh), and runs Soft Parallel Decoding (SPD).

```bash
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all \
  --command="cd ~/dllm-jax && \
    PYTHONPATH=~/dllm-jax \
    RESUME_DIR=gs://dllm-jax-us-east1/checkpoints/$RUN_NAME \
    MODEL_NAME=Qwen/Qwen3-8B \
    PROMPT='Once upon a time' \
    GEN_LENGTH=1024 BLOCK_LENGTH=32 STEPS=32 \
    INFER_IMPL=fast INFER_SPLASH=1 \
    TOP_K=3 THRESHOLD=0.5 CONFIDENCE_STOP=0.9 \
    TP=8 \
    python3 scripts/tpu_infer.py"
```

Three implementations are available — `fast` (default; fixed-shape,
splash-friendly), `kv_fast` (KV-cached, best for ≥1024-token gens),
`legacy` (debug-only Python loop). All three produce byte-identical output
at matching settings.

`INFER_SPLASH=1` swaps in a Pallas splash kernel matched to the
block-causal mask. On Qwen3-8B at `GEN_LENGTH=1024` it's **3.5× faster**
*and* fixes a latent dense-kernel quality bug at non-128-aligned sequence
lengths.

For the full reference — implementation comparison, splash setup, prompt
sweeps, env var glossary, throughput tables, and the single-host CLI —
see [`docs/inference.md`](docs/inference.md). The deep-dive writeup with
end-to-end timing is [`docs/inference-optimization.md`](docs/inference-optimization.md).


## Python API (single host)

For sanity checks on a laptop or single TPU host:

```python
from transformers import AutoTokenizer
from dllm_jax import (
    build_model_from_pretrained,
    DMaxConfig, DMaxTrainer, DMaxDataCollator,
)

model, config = build_model_from_pretrained("Qwen/Qwen3-0.6B", task="llada")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

trainer = DMaxTrainer(
    model=model,
    tokenizer=tokenizer,
    args=DMaxConfig(
        output_dir="./out-dmax",
        max_steps=1000,
        learning_rate=2e-6,
        block_size=32,
        on_policy_ratio=0.5,
        gradient_accumulation_steps=4,
    ),
    train_dataset=dataset,           # any HF datasets-style iterable
    data_collator=DMaxDataCollator(tokenizer=tokenizer, label_pad_token_id=-100),
)
trainer.train()
```

This is the same code path that
[`scripts/dmax_train.py`](scripts/dmax_train.py) drives. Replace
`DMaxConfig` / `DMaxTrainer` with `MDLMConfig` / `MDLMTrainer` (or the
other objectives) for the equivalent single-host MDLM run.


## Package layout

```
dllm_jax/
├── models.py       # GenericDecoderLM (+ call_cached for KV cache),
│                   # GenericEncoderLM, EditFlowModel
├── trainers.py     # MDLM / BD3LM / Dream / DMax / EditFlow trainers
├── configs.py      # ModelArguments, DataArguments, TrainingArguments,
│                   # MDLMConfig, DMaxConfig, …
├── dmax.py         # DMax SPD: dmax_generate_spd[_fast|_kv_fast]
├── schedulers.py   # LinearAlpha, CosineAlpha, LinearKappa, CubicKappa, CosineKappa
├── data.py         # DMaxDataCollator, DreamSFTCollator, EditFlowCollator, …
├── checkpoints.py  # restore_model_checkpoint (single-host)
├── weights.py      # Torch-free safetensors → NNX weight loader
└── utils.py

scripts/
├── tpu_train.py                   # multi-host training (v4 / v5e / v6e)
├── tpu_infer.py                   # multi-host inference from GCS DCP ckpt
├── dmax_train.py                  # single-host DMax CLI
├── dmax_generate.py               # single-host SPD generation (base model)
├── dmax_generate_checkpoint.py    # single-host SPD from saved ckpt
├── inspect_sft_data.py            # SFT pipeline data dump
└── examples/                      # tinystories sanity, legacy v4-32 wrappers

docs/
├── install.md, training.md, inference.md       # user-facing references
├── inference-optimization.md, mfu-optimization.md, kv-cache-design.md
                                                # deep-dive writeups
```


## License

Apache-2.0
