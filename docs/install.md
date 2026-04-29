# Installation

## Prerequisites for multi-host TPU runs

You need three things before any of the multi-host scripts run. Single-host
CPU/GPU development needs none of them.

**1. A Google Cloud project with TPU access.** Install the [`gcloud` CLI](https://cloud.google.com/sdk/docs/install), then:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

**2. A regional GCS bucket for checkpoints.** TPU and bucket should share a
region (cross-region writes are slow):

```bash
gcloud storage buckets create gs://YOUR_BUCKET_NAME --location=us-east1
```

By default this repo writes to `gs://${CHECKPOINT_BUCKET_PREFIX}-${region}`
(prefix `dllm-jax`), so the matching bucket would be `gs://dllm-jax-us-east1`.
Override via `CHECKPOINT_BUCKET=...` or `CHECKPOINT_DIR=gs://...` if you prefer
a different layout.

**3. Optional but recommended:** a [Weights & Biases](https://wandb.ai)
account for loss curves, and a [HuggingFace](https://huggingface.co) account
if you want gated models or to upload checkpoints. You can run `wandb login`
and `huggingface-cli login` once per TPU worker and skip the env vars
entirely.

## TPU VM packaging caveat

Some Python 3.10 TPU VM images ship an older packaging stack where
`pip install -e '.[tpu]'` can fail with a missing `build_editable` hook,
or `pip install '.[tpu]'` misreads metadata as `UNKNOWN-0.0.0` without
installing dependencies. If that happens, skip editable mode and install
deps explicitly from the synced checkout:

```bash
python3 -m pip install --user -U 'jax[tpu]' \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
  'flax>=0.10.0,<0.11' orbax-checkpoint 'gcsfs<=2026.2.0' 'fsspec<=2026.2.0' \
  'optax>=0.2.0' numpy 'transformers>=4.40.0' safetensors \
  huggingface_hub datasets wandb
```

Then run scripts with `PYTHONPATH=/path/to/dllm-jax python3 …`.

## Verified TPU versions

End-to-end validation on TPU v4-32 (`us-central2-b`) with this exact stack.
Pin to these if `pip install '.[tpu]'` shows version drift:

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

Newer flax (0.12+) on JAX 0.7+ should also work; the
`_nnx_list = getattr(nnx, "List", list)` shim in `models.py` handles the
cross-version difference.
