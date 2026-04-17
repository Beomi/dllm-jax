# TPU v4-32 Ondemand DMax Inference

This is the runbook for checkpoint inference on the ondemand TPU v4-32 VM.

## Target TPU

- TPU VM name: `tpu-v4-32-ondemand`
- Zone: `us-central2-b`
- Accelerator: `v4-32`
- Workers: `4`
- JAX devices: `16`
- Mesh used by the inference script: `TP=8`, giving `fsdp=2`, `tp=8`

Check TPU status from the local machine:

```bash
gcloud compute tpus tpu-vm list --zone=us-central2-b
```

Expected row:

```text
NAME                  ZONE           ACCELERATOR_TYPE  TYPE  TOPOLOGY  STATUS
tpu-v4-32-ondemand    us-central2-b  v4-32             V4    2x2x4     READY
```

## Checkpoint

Use this checkpoint parent directory:

```bash
gs://dllm-jax-us-central1/checkpoints/dmax-8b-tinystories-forward8k-3epoch-wandb-online-resume1200-ckpt100-v5e-us-gcsfix-20260416-070036
```

The inference script can auto-detect the latest committed checkpoint by scanning `commit_success.txt` files:

```bash
RESUME_DIR=gs://dllm-jax-us-central1/checkpoints/dmax-8b-tinystories-forward8k-3epoch-wandb-online-resume1200-ckpt100-v5e-us-gcsfix-20260416-070036
```

To force the latest verified checkpoint from our run, set:

```bash
RESUME_STEP=12400
```

Full checkpoint path for that step:

```bash
gs://dllm-jax-us-central1/checkpoints/dmax-8b-tinystories-forward8k-3epoch-wandb-online-resume1200-ckpt100-v5e-us-gcsfix-20260416-070036/checkpoint_12400
```

## Sync Current Code To TPU

From the repo root on the local machine (i.e. the directory that contains `dllm_jax/`, `scripts/`, and this `docs/` folder):

```bash
gcloud compute tpus tpu-vm scp \
  dllm_jax/__init__.py \
  dllm_jax/checkpoints.py \
  dllm_jax/configs.py \
  dllm_jax/data.py \
  dllm_jax/dmax.py \
  dllm_jax/models.py \
  dllm_jax/trainers.py \
  scripts/tpu_dmax_infer_checkpoint.py \
  tpu-v4-32-ondemand:~/dllm-jax/ \
  --zone=us-central2-b \
  --worker=all
```

Move files into the correct package paths and compile-check on every worker:

```bash
gcloud compute tpus tpu-vm ssh tpu-v4-32-ondemand \
  --zone=us-central2-b \
  --worker=all \
  --command='cd ~/dllm-jax && \
    mv -f __init__.py checkpoints.py configs.py data.py dmax.py models.py trainers.py dllm_jax/ && \
    mv -f tpu_dmax_infer_checkpoint.py scripts/ && \
    python3 -m py_compile \
      dllm_jax/__init__.py \
      dllm_jax/checkpoints.py \
      dllm_jax/configs.py \
      dllm_jax/data.py \
      dllm_jax/dmax.py \
      dllm_jax/models.py \
      dllm_jax/trainers.py \
      scripts/tpu_dmax_infer_checkpoint.py'
```

If only inference files changed:

```bash
gcloud compute tpus tpu-vm scp \
  dllm_jax/dmax.py \
  scripts/tpu_dmax_infer_checkpoint.py \
  tpu-v4-32-ondemand:~/dllm-jax/ \
  --zone=us-central2-b \
  --worker=all

gcloud compute tpus tpu-vm ssh tpu-v4-32-ondemand \
  --zone=us-central2-b \
  --worker=all \
  --command='cp ~/dllm-jax/dmax.py ~/dllm-jax/dllm_jax/dmax.py && \
    cp ~/dllm-jax/tpu_dmax_infer_checkpoint.py ~/dllm-jax/scripts/tpu_dmax_infer_checkpoint.py && \
    rm -f ~/dllm-jax/dmax.py ~/dllm-jax/tpu_dmax_infer_checkpoint.py && \
    cd ~/dllm-jax && \
    python3 -m py_compile dllm_jax/dmax.py scripts/tpu_dmax_infer_checkpoint.py'
```

## Run Inference

Reference-compatible DMax settings:

- `BLOCK_LENGTH=32`
- `STEPS=32`
- `INFER_IMPL=fast`
- `FAST_BUCKET_LENGTH=4096`
- `SUPPRESS_MASK_TOKEN=0`
- `CONFIDENCE_STOP=0.9`
- `TP=8`
- `TOP_K=1` for strict reference defaults; use `TOP_K=3` for noticeably more coherent output on this checkpoint

Run a 128-token inference test:

```bash
gcloud compute tpus tpu-vm ssh tpu-v4-32-ondemand \
  --zone=us-central2-b \
  --worker=all \
  --command='cd ~/dllm-jax && \
    RESUME_DIR=gs://dllm-jax-us-central1/checkpoints/dmax-8b-tinystories-forward8k-3epoch-wandb-online-resume1200-ckpt100-v5e-us-gcsfix-20260416-070036 \
    RESUME_STEP=12400 \
    GEN_LENGTH=128 \
    BLOCK_LENGTH=32 \
    STEPS=32 \
    INFER_IMPL=fast \
    FAST_BUCKET_LENGTH=4096 \
    THRESHOLD=0.5 \
    CONFIDENCE_STOP=0.9 \
    SUPPRESS_MASK_TOKEN=0 \
    TOP_K=3 \
    TP=8 \
    python3 scripts/tpu_dmax_infer_checkpoint.py'
```

Run a 1024-token inference test:

```bash
gcloud compute tpus tpu-vm ssh tpu-v4-32-ondemand \
  --zone=us-central2-b \
  --worker=all \
  --command='cd ~/dllm-jax && \
    RESUME_DIR=gs://dllm-jax-us-central1/checkpoints/dmax-8b-tinystories-forward8k-3epoch-wandb-online-resume1200-ckpt100-v5e-us-gcsfix-20260416-070036 \
    RESUME_STEP=12400 \
    GEN_LENGTH=1024 \
    BLOCK_LENGTH=32 \
    STEPS=32 \
    INFER_IMPL=fast \
    FAST_BUCKET_LENGTH=4096 \
    THRESHOLD=0.5 \
    CONFIDENCE_STOP=0.9 \
    SUPPRESS_MASK_TOKEN=0 \
    TOP_K=3 \
    TP=8 \
    python3 scripts/tpu_dmax_infer_checkpoint.py'
```

To use the latest committed checkpoint automatically, omit `RESUME_STEP`.

## Useful Environment Variables

- `PROMPT`: input prompt. Default is `Once upon a time`.
- `MODEL_NAME`: tokenizer/config source. Default is `Qwen/Qwen3-8B`.
- `GEN_LENGTH`: number of tokens to generate.
- `BLOCK_LENGTH`: DMax block size. Use `32` for this checkpoint.
- `STEPS`: max denoising steps per block. Use `32` to match block size.
- `THRESHOLD`: left-to-right confidence threshold. DMax math eval uses `0.5`; code eval uses `0.65`; some benchmark configs use `0.9` or `0.95`.
- `CONFIDENCE_STOP`: block early-stop confidence, default `0.9`.
- `SUPPRESS_MASK_TOKEN`: keep `0` for upstream DMax behavior.
- `INFER_IMPL`: `fast` for compiled JAX loop, `kv_fast` for the KV-cached variant (matches reference's `cache='prefix'` path; each step only projects K/V for the current block rather than the full sequence), `legacy` for the Python loop.
- `FAST_BUCKET_LENGTH`: fixed compiled window length for `fast`. Use `4096` to avoid repeated bucket recompiles for <=4096 total context. Ignored for `kv_fast` (which uses a single compiled while_loop with a pre-allocated KV cache of size `total_length`).
- `TOP_K`: soft-mix top-k (reference default `1`). Setting `3` produces substantially more coherent output on undertrained checkpoints because the previous step's top-k distribution is fed back as a weighted mix of embeddings.
- `TEMPERATURE`: gumbel-max sampling temperature. Default `0.0` = greedy argmax (reference default for eval). Values > 0 sample via gumbel noise.
- `SEED`: RNG seed for gumbel sampling when `TEMPERATURE > 0`. Optional; omit for no RNG.
- `MASK_TOKEN_ID`: optional override. Normally inferred as the model vocab mask token.
- `EOS_TOKEN_ID`: optional override. Normally uses tokenizer EOS.
- `RESTORE_OPTIMIZER`: keep unset or `0` for inference.

## Known Verified Output

All outputs below are from `tpu-v4-32-ondemand`, `checkpoint_12400`, `GEN_LENGTH=128`, `BLOCK_LENGTH=32`, `STEPS=32`, `THRESHOLD=0.5`, `CONFIDENCE_STOP=0.9`.

`TOP_K=1` (reference default) — collapses into repetition because the checkpoint is undertrained for single-token soft mixing:

```text
prompt='Once upon a time'
generated:
, there was a little girl named Lily. She loved to play outside her the and. One day, One day, her her her her her...................................................................................................
nfe=151 generated_tokens=128
```

`TOP_K=3` — same checkpoint, same settings, coherent prose:

```text
prompt='Once upon a time'
generated:
, there was a little girl named Lily. She loved to play with her toys and her favorite toy was a teddy bear. One day, Lily's mom asked her to clean up her toys. Lily didn't want to clean up her toys, but her mom said it was important to keep her room clean.

Lily started to clean up her toys, but she did not want to put her teddy bear away. She asked her mom if she could keep her teddy bear. Her mom said yes, but only if she promised to clean up her toys every.
...
nfe=114 generated_tokens=128
```

Both `INFER_IMPL=fast` and `INFER_IMPL=legacy` report identical NFE (step-level early-break is aligned) and identical generated text.

`INFER_IMPL=kv_fast` produces the same text as `fast` but reports `nfe = fast_nfe + num_active_blocks` (the extra `num_active_blocks` comes from the post-block hard-write forwards that overwrite the soft K/V in the cache with hard K/V — the JAX analogue of reference's cross-block update).

Measured on `tpu-v4-32-ondemand`, `checkpoint_12400`, `TOP_K=3`:

| Config | nfe | generate_seconds |
|-|-|-|
| fast, GEN=128 | 114 | 32.1 |
| kv_fast, GEN=128 | 119 | 65.9 |
| fast, GEN=1024 | 1010 | 124.7 |
| kv_fast, GEN=1024 | 1043 | 76.6 |

At short generations the KV-cache compile overhead dominates (larger PyTree carry). At `GEN=1024` the block-local forward wins, ~1.6× faster than `fast`.

## Troubleshooting

If import errors mention missing `DMaxConfig`, `DMaxTrainer`, or `dmax_generate_spd`, the TPU checkout is stale. Re-run the full sync section.

If Orbax restore hangs or errors around signal contracts, `scripts/tpu_dmax_infer_checkpoint.py` installs the fallback by default through `CHECKPOINT_ORBAX_SIGNAL_FALLBACK=1`.

If `gcs://` or `gs://` access warnings appear from TensorFlow, they are usually harmless for this script because checkpoint restore uses Orbax/GCS client paths.

If generation is slow on the first run, that includes model restore and XLA compilation. Repeated runs with the same shapes should mostly measure restore plus compiled generation.

If output quality collapses into repeated punctuation with `TOP_K=1` on an undertrained checkpoint, try `TOP_K=3` or higher. The SPD algorithm feeds the previous step's token distribution back as a soft mix; for `TOP_K=1` a low-confidence single token gives a bad signal, while `TOP_K=3` aggregates across the top candidates and stabilizes the decoding feedback.

If output quality collapses into repeated punctuation, that is a model/checkpoint quality issue seen with the verified checkpoint, not a failure to use TPU or the DMax block size.
