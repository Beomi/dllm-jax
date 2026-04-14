"""Qwen3-8B MDLM smoke test on TPU v6e-64 — clean dllm_jax imports.

Config: seq=16384, batch=8, 20 steps, AdamW, 2D FSDP+TP mesh (8x8).
This replaces the old module-stub training scripts with direct
`from dllm_jax import ...` usage now that the package is standalone.
"""
import os
import sys
import time
import gc
import json
import shutil
import urllib.request
from contextlib import contextmanager
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LIBTPU_INIT_ARGS"] = " ".join([
    "--xla_tpu_enable_async_collective_fusion=true",
    "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true",
    "--xla_tpu_overlap_compute_collective_tc=true",
    "--xla_enable_async_all_gather=true",
    "--xla_tpu_data_parallel_opt_different_sized_ops=true",
])

import jax
import jax.numpy as jnp
import numpy as np
import optax
import transformers
from flax import nnx
from flax.training import checkpoints
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.shard_map import shard_map
from jax.experimental.pallas.ops.tpu.flash_attention import (
    BlockSizes as _FlashBlockSizes,
    flash_attention as _pallas_flash,
)
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dllm_jax import (
    GenericDecoderLM,
    LinearAlphaScheduler,
    MDLMConfig,
    load_pretrained_weights,
    model_spec_from_config,
)
from dllm_jax import models as dllm_models

try:
    import orbax.checkpoint as ocp
    from orbax.checkpoint import options as ocp_options
except ImportError:
    ocp = None
    ocp_options = None

proc = jax.process_index()
nproc = jax.process_count()
ndev = jax.device_count()
nlocal = jax.local_device_count()
print(f"[Worker {proc}/{nproc}] devices={ndev} local={nlocal} backend={jax.default_backend()}", flush=True)

# ── 2D mesh: FSDP (8) × TP (8) ──────────────────────────────
TP = 8
DP = ndev // TP  # 8 on v6e-64
devices = mesh_utils.create_device_mesh((DP, TP))
mesh = Mesh(devices, axis_names=("fsdp", "tp"))
if proc == 0:
    print(f"[Worker {proc}] 2D Mesh: fsdp={DP} × tp={TP}", flush=True)

def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}

def sync_all(label: str):
    if nproc > 1:
        multihost_utils.sync_global_devices(label)

def path_safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "-" for c in value).strip("-")

def metadata_value(path: str, timeout: float = 0.25) -> str | None:
    req = urllib.request.Request(
        f"http://metadata.google.internal/computeMetadata/v1/{path}",
        headers={"Metadata-Flavor": "Google"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode().strip()
    except Exception:
        return None

def normalize_zone(value: str | None) -> str | None:
    if not value:
        return None
    return value.rsplit("/", 1)[-1]

def region_from_zone(zone: str | None) -> str | None:
    if not zone:
        return None
    parts = zone.split("-")
    if len(parts) < 3:
        return zone
    return "-".join(parts[:-1])

def detect_tpu_zone() -> str | None:
    for name in ("TPU_ZONE", "ZONE", "CLOUDSDK_COMPUTE_ZONE", "GOOGLE_CLOUD_ZONE"):
        zone = normalize_zone(os.environ.get(name))
        if zone:
            return zone
    return normalize_zone(metadata_value("instance/zone"))

def gs_join(base: str, *parts: str) -> str:
    return "/".join([base.rstrip("/"), *(part.strip("/") for part in parts if part and part.strip("/"))])

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
DATASET = os.environ.get("DATASET", "tinystories").lower()
SYNTHETIC_DATA = DATASET in {"synthetic", "random", "dummy"}
DATASET_PATH = os.environ.get("DATASET_PATH")
MODEL_SLUG = path_safe_name(MODEL_NAME.rstrip("/").rsplit("/", 1)[-1]) or "model"
RUN_NAME = os.environ.get("RUN_NAME") or os.environ.get("WANDB_RUN_NAME") or f"{MODEL_SLUG}-{DATASET}-{int(time.time())}"
MAX_LEN = int(os.environ.get("MAX_LEN", "16384"))
GLOBAL_BATCH = int(os.environ.get("GLOBAL_BATCH", "8"))
NUM_STEPS = int(os.environ.get("NUM_STEPS", "20"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "0"))
RUN_FULL_EPOCHS = NUM_EPOCHS > 0 and NUM_STEPS <= 0
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "5"))
PEAK_LR = float(os.environ.get("PEAK_LR", "1e-4"))
OPTIMIZER = os.environ.get("OPTIMIZER", "adamw").lower()
LOAD_PRETRAINED = env_flag("LOAD_PRETRAINED", True)
LOGGING_STEPS = int(os.environ.get("LOGGING_STEPS", "1"))
WANDB_LOG = os.environ.get("WANDB_LOG", os.environ.get("USE_WANDB", "0")).lower() in {"1", "true", "yes", "on"}
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "dllm-jax")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME")
WANDB_MODE = os.environ.get("WANDB_MODE", "online")
PEAK_TFLOPS_PER_CHIP = float(os.environ.get("PEAK_TFLOPS_PER_CHIP", "918"))
CHECKPOINT_STEPS = int(os.environ.get("CHECKPOINT_STEPS", os.environ.get("SAVE_STEPS", "500")))
TPU_ZONE = detect_tpu_zone()
TPU_REGION = os.environ.get("TPU_REGION") or os.environ.get("REGION") or region_from_zone(TPU_ZONE)
CHECKPOINT_BUCKET_PREFIX = os.environ.get("CHECKPOINT_BUCKET_PREFIX", "dllm-jax")
CHECKPOINT_BUCKET = os.environ.get("CHECKPOINT_BUCKET")
if not CHECKPOINT_BUCKET and TPU_REGION:
    CHECKPOINT_BUCKET = f"{CHECKPOINT_BUCKET_PREFIX}-{TPU_REGION}"
CHECKPOINT_SUBDIR = os.environ.get("CHECKPOINT_SUBDIR", "checkpoints").strip("/")
if os.environ.get("CHECKPOINT_DIR"):
    CHECKPOINT_DIR = os.environ["CHECKPOINT_DIR"]
elif env_flag("CHECKPOINT_USE_GCS", True) and CHECKPOINT_BUCKET:
    CHECKPOINT_DIR = gs_join(
        CHECKPOINT_BUCKET if CHECKPOINT_BUCKET.startswith("gs://") else f"gs://{CHECKPOINT_BUCKET}",
        CHECKPOINT_SUBDIR,
        RUN_NAME,
    )
else:
    CHECKPOINT_DIR = str(Path(os.environ.get("LOCAL_CHECKPOINT_DIR", "/tmp/dllm-jax-checkpoints")).expanduser() / RUN_NAME)
CHECKPOINT_DIR_IS_GCS = CHECKPOINT_DIR.startswith("gs://")
CHECKPOINT_LOCAL_DIR = None if CHECKPOINT_DIR_IS_GCS else Path(CHECKPOINT_DIR).expanduser()
CHECKPOINT_KEEP = int(os.environ.get("CHECKPOINT_KEEP", "2"))
CHECKPOINT_ON_FINISH = env_flag("CHECKPOINT_ON_FINISH", False)
CHECKPOINT_ORBAX_SYNC_DIRS = env_flag("CHECKPOINT_ORBAX_SYNC_DIRS", True)
CHECKPOINT_ORBAX_SIGNAL_FALLBACK = env_flag("CHECKPOINT_ORBAX_SIGNAL_FALLBACK", True)
HF_CHECKPOINT_REPO = os.environ.get("HF_CHECKPOINT_REPO") or os.environ.get("HF_REPO_ID")
HF_CHECKPOINT_REPO_TYPE = os.environ.get("HF_CHECKPOINT_REPO_TYPE", "model")
HF_CHECKPOINT_PATH = os.environ.get("HF_CHECKPOINT_PATH", "checkpoints").strip("/")
HF_CHECKPOINT_PRIVATE = env_flag("HF_CHECKPOINT_PRIVATE", True)
HF_CHECKPOINT_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
)
RESUME_DIR = os.environ.get("RESUME_DIR") or os.environ.get("RESUME_FROM")
RESUME_STEP = int(os.environ.get("RESUME_STEP", "0"))  # 0 = latest available

if WANDB_LOG and WANDB_MODE != "offline" and proc == 0:
    has_wandb_auth = bool(os.environ.get("WANDB_API_KEY")) or Path.home().joinpath(".netrc").exists()
    if not has_wandb_auth:
        raise RuntimeError(
            "WANDB_LOG=1 requested, but no WANDB_API_KEY or ~/.netrc exists on this TPU worker. "
            "Run `wandb login` or export WANDB_API_KEY on all workers first."
        )

wandb = None
wandb_run = None
if WANDB_LOG and proc == 0:
    try:
        import wandb as _wandb
    except ImportError as exc:
        raise RuntimeError("WANDB_LOG=1 requested, but wandb is not installed. Install `wandb` on all workers.") from exc
    wandb = _wandb
    wandb_run = wandb.init(
        project=WANDB_PROJECT,
        name=RUN_NAME,
        config={
            "run_name": RUN_NAME,
            "model_name": MODEL_NAME,
            "dataset": DATASET,
            "synthetic_data": SYNTHETIC_DATA,
            "dataset_path": DATASET_PATH,
            "max_len": MAX_LEN,
            "global_batch": GLOBAL_BATCH,
            "num_steps": NUM_STEPS,
            "num_epochs": NUM_EPOCHS,
            "run_full_epochs": RUN_FULL_EPOCHS,
            "warmup_steps": WARMUP_STEPS,
            "peak_lr": PEAK_LR,
            "optimizer": OPTIMIZER,
            "load_pretrained": LOAD_PRETRAINED,
            "peak_tflops_per_chip": PEAK_TFLOPS_PER_CHIP,
            "tpu_zone": TPU_ZONE,
            "tpu_region": TPU_REGION,
            "checkpoint_bucket": CHECKPOINT_BUCKET,
            "checkpoint_steps": CHECKPOINT_STEPS,
            "checkpoint_dir": CHECKPOINT_DIR,
            "checkpoint_keep": CHECKPOINT_KEEP,
            "checkpoint_on_finish": CHECKPOINT_ON_FINISH,
            "resume_dir": RESUME_DIR,
            "resume_step": RESUME_STEP,
            "hf_checkpoint_repo": HF_CHECKPOINT_REPO,
            "hf_checkpoint_path": HF_CHECKPOINT_PATH,
        },
    )

hf_api = None
if HF_CHECKPOINT_REPO and CHECKPOINT_DIR_IS_GCS and proc == 0:
    print(
        "[Worker 0] HF_CHECKPOINT_REPO ignored because CHECKPOINT_DIR is gs://. "
        "Use GCS as the durable checkpoint target.",
        flush=True,
    )
if HF_CHECKPOINT_REPO and not CHECKPOINT_DIR_IS_GCS and (CHECKPOINT_STEPS > 0 or CHECKPOINT_ON_FINISH):
    has_hf_auth = bool(HF_CHECKPOINT_TOKEN) or Path.home().joinpath(".cache", "huggingface", "token").exists()
    if not has_hf_auth:
        raise RuntimeError(
            "HF_CHECKPOINT_REPO is set, but no HF_TOKEN/HUGGING_FACE_HUB_TOKEN or cached Hugging Face token exists."
        )
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "HF_CHECKPOINT_REPO is set, but huggingface_hub is not installed."
        ) from exc
    hf_api = HfApi(token=HF_CHECKPOINT_TOKEN)
    if proc == 0:
        hf_api.create_repo(
            repo_id=HF_CHECKPOINT_REPO,
            repo_type=HF_CHECKPOINT_REPO_TYPE,
            private=HF_CHECKPOINT_PRIVATE,
            exist_ok=True,
        )
        print(
            f"[Worker {proc}] HF checkpoints enabled: repo={HF_CHECKPOINT_REPO} path={HF_CHECKPOINT_PATH}",
            flush=True,
        )
    sync_all("hf-checkpoint-repo-ready")

config = transformers.AutoConfig.from_pretrained(MODEL_NAME)

def estimate_model_params(cfg) -> int:
    hidden = int(getattr(cfg, "hidden_size"))
    vocab = int(getattr(cfg, "vocab_size"))
    layers = int(getattr(cfg, "num_hidden_layers"))
    heads = int(getattr(cfg, "num_attention_heads"))
    kv_heads = int(getattr(cfg, "num_key_value_heads", heads))
    head_dim = int(getattr(cfg, "head_dim", hidden // heads))
    intermediate = int(getattr(cfg, "intermediate_size"))
    attn_params = hidden * (heads * head_dim + 2 * kv_heads * head_dim) + hidden * hidden
    mlp_params = 3 * hidden * intermediate
    norm_params = 2 * hidden + 2 * kv_heads * head_dim
    embedding_params = vocab * hidden
    lm_head_params = 0 if bool(getattr(cfg, "tie_word_embeddings", False)) else vocab * hidden
    return int(embedding_params + lm_head_params + layers * (attn_params + mlp_params + norm_params))

EST_PARAMS = estimate_model_params(config)
TRAIN_FLOPS_PER_TOKEN_DENSE = 6 * EST_PARAMS
TRAIN_FLOPS_PER_TOKEN_ATTN = 12 * int(config.num_hidden_layers) * MAX_LEN * int(config.hidden_size)
TRAIN_FLOPS_PER_TOKEN_TOTAL = TRAIN_FLOPS_PER_TOKEN_DENSE + TRAIN_FLOPS_PER_TOKEN_ATTN
PEAK_FLOPS = PEAK_TFLOPS_PER_CHIP * 1e12 * ndev

if wandb_run is not None:
    wandb.config.update(
        {
            "estimated_params": EST_PARAMS,
            "train_flops_per_token_dense": TRAIN_FLOPS_PER_TOKEN_DENSE,
            "train_flops_per_token_attention": TRAIN_FLOPS_PER_TOKEN_ATTN,
            "train_flops_per_token_total": TRAIN_FLOPS_PER_TOKEN_TOTAL,
            "peak_flops": PEAK_FLOPS,
        },
        allow_val_change=True,
    )

if proc == 0:
    print(f"[Worker {proc}] {MODEL_NAME}: {config.num_hidden_layers}L h={config.hidden_size} "
          f"V={config.vocab_size}", flush=True)
    print(
        f"[Worker {proc}] estimated_params={EST_PARAMS/1e9:.2f}B "
        f"flops/token dense={TRAIN_FLOPS_PER_TOKEN_DENSE/1e9:.1f}G "
        f"attn={TRAIN_FLOPS_PER_TOKEN_ATTN/1e9:.1f}G "
        f"peak={PEAK_FLOPS/1e15:.2f} PFLOP/s",
        flush=True,
    )
    if RUN_FULL_EPOCHS:
        print(f"[Worker {proc}] seq={MAX_LEN} batch={GLOBAL_BATCH} epochs={NUM_EPOCHS}", flush=True)
    else:
        print(f"[Worker {proc}] seq={MAX_LEN} batch={GLOBAL_BATCH} steps={NUM_STEPS}", flush=True)

# ── Pallas flash attention under shard_map ─────────────────
_FLASH_BLOCKS = _FlashBlockSizes(
    block_q=512, block_k_major=512, block_k=512, block_b=1,
    block_q_major_dkv=512, block_k_major_dkv=512, block_k_dkv=512, block_q_dkv=512,
    block_k_major_dq=512, block_k_dq=512, block_q_dq=512,
)

def _sharded_flash_attn(q, k, v, sm_scale):
    return shard_map(
        lambda q, k, v: _pallas_flash(q, k, v, sm_scale=sm_scale, block_sizes=_FLASH_BLOCKS),
        mesh=mesh,
        in_specs=(P("fsdp", "tp", None, None),) * 3,
        out_specs=P("fsdp", "tp", None, None),
        check_rep=False,
    )(q, k, v)

dllm_models._FLASH_ATTN_FN = _sharded_flash_attn
if proc == 0:
    print(f"[Worker {proc}] Pallas flash installed (blocks=512)", flush=True)

# ── Per-layer remat ────────────────────────────────────────
remat_policy = jax.checkpoint_policies.nothing_saveable

def _remat_hidden_for_heads(self, input_ids, *, attention_mask=None, position_ids=None):
    hidden_states = self.embed_tokens(input_ids)
    if position_ids is None:
        position_ids = jnp.broadcast_to(
            jnp.arange(hidden_states.shape[1])[None, :],
            (hidden_states.shape[0], hidden_states.shape[1]),
        )
    for layer in self.layers:
        hidden_states = jax.remat(
            lambda hs, l=layer: l(hs, attention_mask=attention_mask, position_ids=position_ids),
            policy=remat_policy,
        )(hidden_states)
    return self.norm(hidden_states)

GenericDecoderLM.hidden_for_heads = _remat_hidden_for_heads

# ── CPU init, then shard to TPU ────────────────────────────
print(f"[Worker {proc}] Building model on CPU...", flush=True)
t0 = time.time()

spec = model_spec_from_config(config, task="llada")
if getattr(config, "model_type", "") == "qwen3":
    spec.use_qk_norm = True  # Qwen3 q_norm/k_norm not in config.json but present in weights

cpu_device = jax.devices("cpu")[0]
with jax.default_device(cpu_device):
    model = GenericDecoderLM(spec, dtype_name="bfloat16", rngs=nnx.Rngs(params=42, dropout=43))

print(f"[Worker {proc}] CPU init done in {time.time()-t0:.1f}s", flush=True)

# ── Pretrained weight loading (torch-free) ──────────────────
if LOAD_PRETRAINED:
    tload = time.time()
    n_loaded, missing, shape_mismatch = load_pretrained_weights(model, MODEL_NAME, verbose=(proc == 0))
    print(f"[Worker {proc}] Loaded {n_loaded} tensors in {time.time()-tload:.1f}s", flush=True)
    if missing or shape_mismatch:
        raise RuntimeError(
            f"Weight loading incomplete: missing={len(missing)} shape_mismatch={len(shape_mismatch)}"
        )
else:
    n_loaded = 0
    print(f"[Worker {proc}] LOAD_PRETRAINED=0; using random initialization", flush=True)

# ── Shard to 2D mesh ───────────────────────────────────────
print(f"[Worker {proc}] Sharding to 2D TPU mesh...", flush=True)
t1 = time.time()

gdef, state = nnx.split(model)

def _host_value(x):
    if isinstance(x, jax.Array):
        return np.asarray(x)
    return x

def fsdp_tp_sharding(x):
    if isinstance(x, (jax.Array, jnp.ndarray, np.ndarray)):
        if isinstance(x, jax.Array) and any(d.platform != "cpu" for d in x.devices()):
            return x
        x = _host_value(x)
        if x.ndim >= 2 and x.shape[0] >= 8 and x.shape[1] >= 8:
            sharding = NamedSharding(mesh, P("fsdp", "tp"))
        elif x.ndim == 1 and x.shape[0] >= 8:
            sharding = NamedSharding(mesh, P("tp"))
        else:
            sharding = NamedSharding(mesh, P())
        return jax.make_array_from_callback(x.shape, sharding, lambda idx, arr=x: arr[idx])
    return x

def shard_tree(tree, label: str):
    print(f"[Worker {proc}] Flattening {label} state...", flush=True)
    leaves, treedef = jax.tree_util.tree_flatten(
        tree,
        is_leaf=lambda x: isinstance(x, (jax.Array, jnp.ndarray, np.ndarray)),
    )
    print(f"[Worker {proc}] Flattened {label} state: {len(leaves)} leaves", flush=True)
    sharded = []
    for i, leaf in enumerate(leaves, start=1):
        should_log = i == 1 or i % 400 == 0 or i == len(leaves)
        if should_log:
            shape = getattr(leaf, "shape", None)
            dtype = getattr(leaf, "dtype", None)
            print(f"[Worker {proc}] Sharding {label}: leaf {i}/{len(leaves)} shape={shape} dtype={dtype}", flush=True)
        sharded.append(fsdp_tp_sharding(leaf))
        if should_log:
            print(f"[Worker {proc}] Sharded {label}: leaf {i}/{len(leaves)}", flush=True)
    return jax.tree_util.tree_unflatten(treedef, sharded)

state = shard_tree(state, "model")
model = nnx.merge(gdef, state)
del gdef, state
gc.collect()
print(f"[Worker {proc}] Sharding done in {time.time()-t1:.1f}s (total: {time.time()-t0:.1f}s)", flush=True)

# ── Optimizer: AdamW (safer than Adafactor for pretrained MDLM) ──
if WARMUP_STEPS > 0:
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, PEAK_LR, WARMUP_STEPS),
            optax.constant_schedule(PEAK_LR),
        ],
        boundaries=[WARMUP_STEPS],
    )
else:
    lr_schedule = optax.constant_schedule(PEAK_LR)
if OPTIMIZER == "adafactor":
    # Factored optimizer state (~4x less HBM than AdamW) — required to fit
    # 128k context on Qwen3-8B per chip. Loss may climb back after ~step 60
    # per CLAUDE.md, so only use for short-horizon smoke tests.
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adafactor(learning_rate=lr_schedule, weight_decay_rate=0.01, dtype_momentum=jnp.bfloat16),
    )
else:
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.01),
    )
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

opt_gdef, opt_state = nnx.split(optimizer)
opt_state = shard_tree(opt_state, "optimizer")
optimizer = nnx.merge(opt_gdef, opt_state)
del opt_gdef, opt_state
gc.collect()
# CRITICAL per CLAUDE.md: rebind after nnx.merge or training becomes silent no-op.
model = optimizer.model
print(f"[Worker {proc}] Optimizer ready ({OPTIMIZER}, model rebound)", flush=True)

# ── MDLM config + tokenizer + data ─────────────────────────
mdlm_cfg = MDLMConfig(output_dir="/tmp/q3-8b-smoke", max_steps=NUM_STEPS, learning_rate=PEAK_LR)
alpha_sched = LinearAlphaScheduler()
mask_id = config.vocab_size - 1

print(f"[Worker {proc}] Loading tokenizer + dataset stream...", flush=True)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

if not SYNTHETIC_DATA:
    from datasets import load_dataset

    def make_dataset_iter():
        if DATASET_PATH:
            parquet_files = sorted(str(p) for p in Path(DATASET_PATH).glob("**/*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found under DATASET_PATH={DATASET_PATH}")
            ds = load_dataset("parquet", data_files=parquet_files, split="train", streaming=True)
        elif DATASET == "wikipedia":
            ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        else:
            ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        return iter(ds)

synthetic_rng = np.random.default_rng(1000 + proc)
print(f"[Worker {proc}] Dataset: {DATASET}", flush=True)
ds_iter = None if SYNTHETIC_DATA else make_dataset_iter()
token_buffer: list[int] = []
eos_id = tokenizer.eos_token_id

def refill_buffer(needed):
    global token_buffer
    exhausted = False
    while len(token_buffer) < needed:
        try:
            text = next(ds_iter)["text"]
        except StopIteration:
            exhausted = True
            break
        # Append EOS between docs so the model sees document boundaries.
        token_buffer.extend(tokenizer.encode(text, add_special_tokens=False))
        token_buffer.append(eos_id)
    return not exhausted

def get_batch():
    global token_buffer
    if SYNTHETIC_DATA:
        ids = synthetic_rng.integers(0, max(2, mask_id), size=(GLOBAL_BATCH, MAX_LEN), dtype=np.int64)
        return {"input_ids": ids, "labels": ids.copy()}
    refill_buffer(GLOBAL_BATCH * MAX_LEN)
    if not token_buffer:
        return None
    ids = np.full((GLOBAL_BATCH, MAX_LEN), tokenizer.pad_token_id, dtype=np.int64)
    for i in range(GLOBAL_BATCH):
        length = min(MAX_LEN, len(token_buffer))
        if length > 0:
            ids[i, :length] = token_buffer[:length]
            token_buffer = token_buffer[length:]
    return {"input_ids": ids, "labels": ids.copy()}

# ── Loss + step ─────────────────────────────────────────────
def loss_fn(mdl, batch):
    out = mdl(batch["model_input_ids"], attention_mask=batch.get("attention_mask"))
    nll = optax.softmax_cross_entropy_with_integer_labels(out["logits"], batch["input_ids"])
    nll = nll * batch["loss_weights"] * batch["masked_mask"].astype(nll.dtype)
    nll = nll / jnp.clip(batch["maskable_mask"].sum(), min=1)
    loss = nll.sum()
    return loss, {"loss": loss}

@nnx.jit
def train_step(mdl, opt, batch):
    (_, m), g = nnx.value_and_grad(lambda m: loss_fn(m, batch), has_aux=True)(mdl)
    opt.update(g)
    return m

data_sharding = NamedSharding(mesh, P("fsdp", None))

def shard_batch_array(arr):
    return jax.make_array_from_callback(arr.shape, data_sharding, lambda idx, x=arr: x[idx])

if proc == 0:
    print(f"\n{'='*70}")
    print(f"MDLM {'full epoch training' if RUN_FULL_EPOCHS else 'smoke test'}: {MODEL_NAME}")
    print(f"  {ndev} chips | mesh=({DP},{TP}) fsdp×tp")
    if RUN_FULL_EPOCHS:
        print(f"  seq={MAX_LEN} batch={GLOBAL_BATCH} epochs={NUM_EPOCHS}")
    else:
        print(f"  seq={MAX_LEN} batch={GLOBAL_BATCH} steps={NUM_STEPS}")
    print(f"  tokens/step={GLOBAL_BATCH * MAX_LEN:,}")
    print(f"  PRETRAINED: {n_loaded} tensors")
    if WANDB_LOG:
        print(f"  wandb={WANDB_PROJECT} mode={WANDB_MODE}")
    if CHECKPOINT_STEPS > 0:
        print(f"  checkpoints every {CHECKPOINT_STEPS} steps -> {CHECKPOINT_DIR}")
        if TPU_REGION:
            print(f"  checkpoint region={TPU_REGION} zone={TPU_ZONE}")
        if HF_CHECKPOINT_REPO and not CHECKPOINT_DIR_IS_GCS:
            print(f"  checkpoint hub repo={HF_CHECKPOINT_REPO} path={HF_CHECKPOINT_PATH}")
    if RESUME_DIR:
        print(f"  RESUME from {RESUME_DIR}" + (f" step={RESUME_STEP}" if RESUME_STEP > 0 else " (latest)"))
    print(f"{'='*70}", flush=True)

rng = jax.random.key(42)
total_tokens = 0
last_epoch = 0
last_epoch_step = 0
last_checkpoint_step = 0
t_start = time.time()

def repo_path(*parts: str) -> str:
    return "/".join(part.strip("/") for part in (HF_CHECKPOINT_PATH, *parts) if part and part.strip("/"))

def prune_local_checkpoints():
    if CHECKPOINT_LOCAL_DIR is None or CHECKPOINT_KEEP <= 0:
        return
    candidates = sorted(path for path in CHECKPOINT_LOCAL_DIR.glob("checkpoint-*") if path.is_dir())
    for path in candidates[:-CHECKPOINT_KEEP]:
        shutil.rmtree(path, ignore_errors=True)

def upload_checkpoint_to_hub(step_dir: str, checkpoint_name: str, global_step: int):
    if hf_api is None:
        return
    if CHECKPOINT_DIR_IS_GCS:
        return
    if nproc == 1:
        if proc == 0:
            path_in_repo = repo_path(checkpoint_name)
            print(f"[Worker {proc}] Uploading {checkpoint_name} to {HF_CHECKPOINT_REPO}/{path_in_repo}", flush=True)
            hf_api.upload_folder(
                repo_id=HF_CHECKPOINT_REPO,
                repo_type=HF_CHECKPOINT_REPO_TYPE,
                folder_path=str(step_dir),
                path_in_repo=path_in_repo,
                commit_message=f"Add checkpoint step {global_step}",
            )
        return

    for owner in range(nproc):
        sync_all(f"hf-upload-{global_step}-{owner}-start")
        if proc == owner:
            process_name = f"process-{proc:05d}-of-{nproc:05d}"
            path_in_repo = repo_path(checkpoint_name, process_name)
            print(f"[Worker {proc}] Uploading {checkpoint_name}/{process_name} to HF Hub", flush=True)
            hf_api.upload_folder(
                repo_id=HF_CHECKPOINT_REPO,
                repo_type=HF_CHECKPOINT_REPO_TYPE,
                folder_path=str(step_dir),
                path_in_repo=path_in_repo,
                commit_message=f"Add checkpoint step {global_step} {process_name}",
            )
        sync_all(f"hf-upload-{global_step}-{owner}-done")

_orbax_signal_fallback_installed = False

def jax_distributed_client_initialized() -> bool:
    try:
        import jax._src.distributed as jax_distributed  # pylint: disable=import-outside-toplevel
        return jax_distributed.global_state.client is not None
    except Exception:
        return False

def maybe_install_orbax_signal_fallback():
    global _orbax_signal_fallback_installed
    if (
        _orbax_signal_fallback_installed
        or not CHECKPOINT_ORBAX_SIGNAL_FALLBACK
        or nproc <= 1
        or jax_distributed_client_initialized()
    ):
        return

    try:
        from orbax.checkpoint._src.futures import future as orbax_future  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        if proc == 0:
            print(f"[Worker {proc}] Orbax signal fallback unavailable: {exc}", flush=True)
        return

    # This TPU runtime uses JAX multihost process sync but does not initialize
    # the distributed key-value client Orbax uses for async signal contracts.
    # With synchronous directory creation, no contract signals are needed.
    orbax_future.AwaitableSignalsContract.get_awaitable_signals_from_contract = classmethod(
        lambda cls, operation_id=None: []
    )
    _orbax_signal_fallback_installed = True
    if proc == 0:
        print(
            "[Worker 0] Orbax signal-contract fallback enabled "
            "(JAX distributed client is not initialized)",
            flush=True,
        )

def make_orbax_checkpointer():
    if ocp is None:
        return None

    if CHECKPOINT_ORBAX_SYNC_DIRS:
        maybe_install_orbax_signal_fallback()

    if ocp_options is None:
        return ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

    def barrier_sync_fn(key: str, timeout_ms: int):
        del timeout_ms
        multihost_utils.sync_global_devices(key)

    multiprocessing_options = ocp_options.MultiprocessingOptions(primary_host=0)
    async_options = ocp_options.AsyncOptions(
        barrier_sync_fn=barrier_sync_fn,
        create_directories_asynchronously=not CHECKPOINT_ORBAX_SYNC_DIRS,
    )
    try:
        if hasattr(ocp, "StandardCheckpointer"):
            return ocp.StandardCheckpointer(
                async_options=async_options,
                multiprocessing_options=multiprocessing_options,
            )
        return ocp.AsyncCheckpointer(
            ocp.PyTreeCheckpointHandler(multiprocessing_options=multiprocessing_options),
            async_options=async_options,
            multiprocessing_options=multiprocessing_options,
        )
    except TypeError:
        return ocp.Checkpointer(
            ocp.PyTreeCheckpointHandler(multiprocessing_options=multiprocessing_options)
        )

def wait_for_checkpoint(checkpointer):
    if hasattr(checkpointer, "wait_until_finished"):
        checkpointer.wait_until_finished()

@contextmanager
def orbax_set_mesh_context_patch():
    original_set_mesh = jax.sharding.set_mesh
    probe = original_set_mesh(None)
    if hasattr(probe, "__enter__"):
        yield
        return
    original_set_mesh(probe)

    def patched_set_mesh(mesh_arg):
        @contextmanager
        def mesh_context():
            previous_mesh = original_set_mesh(mesh_arg)
            try:
                yield previous_mesh
            finally:
                original_set_mesh(previous_mesh)

        return mesh_context()

    jax.sharding.set_mesh = patched_set_mesh
    try:
        yield
    finally:
        jax.sharding.set_mesh = original_set_mesh

def save_training_checkpoint(global_step: int, epoch: int, epoch_step: int, *, force: bool = False):
    global last_checkpoint_step
    if global_step <= 0:
        return
    if not force and (CHECKPOINT_STEPS <= 0 or global_step % CHECKPOINT_STEPS != 0):
        return
    if last_checkpoint_step == global_step:
        return

    checkpoint_name = f"checkpoint-{global_step:06d}"
    step_dir = CHECKPOINT_DIR if CHECKPOINT_DIR_IS_GCS else str(CHECKPOINT_LOCAL_DIR / checkpoint_name)
    checkpoint_keep = CHECKPOINT_KEEP if CHECKPOINT_DIR_IS_GCS else 1
    if not CHECKPOINT_DIR_IS_GCS:
        Path(step_dir).mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_name": MODEL_NAME,
        "dataset": DATASET,
        "dataset_path": DATASET_PATH,
        "global_step": global_step,
        "epoch": epoch,
        "epoch_step": epoch_step,
        "total_tokens": total_tokens,
        "max_len": MAX_LEN,
        "global_batch": GLOBAL_BATCH,
        "optimizer": OPTIMIZER,
        "nproc": nproc,
        "process_index": proc,
        "tpu_zone": TPU_ZONE,
        "tpu_region": TPU_REGION,
        "checkpoint_dir": CHECKPOINT_DIR,
    }
    if not CHECKPOINT_DIR_IS_GCS:
        local_step_dir = Path(step_dir)
        config.save_pretrained(local_step_dir)
        tokenizer.save_pretrained(local_step_dir)
        with local_step_dir.joinpath("training_state.json").open("w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    target = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "global_step": np.asarray(global_step, dtype=np.int64),
        "epoch": np.asarray(epoch, dtype=np.int32),
        "epoch_step": np.asarray(epoch_step, dtype=np.int64),
        "total_tokens": np.asarray(total_tokens, dtype=np.int64),
        "rng": np.asarray(jax.random.key_data(rng)),
    }
    if proc == 0:
        print(f"[Worker {proc}] Saving checkpoint {checkpoint_name} -> {step_dir}", flush=True)

    if nproc > 1:
        if not hasattr(checkpoints, "save_checkpoint_multiprocess"):
            raise RuntimeError("Multi-process checkpointing requires flax.training.checkpoints.save_checkpoint_multiprocess.")
        if ocp is None:
            raise RuntimeError("Multi-process checkpointing requires orbax-checkpoint.")
        orbax_checkpointer = make_orbax_checkpointer()
        with orbax_set_mesh_context_patch():
            checkpoints.save_checkpoint_multiprocess(
                ckpt_dir=str(step_dir),
                target=target,
                step=global_step,
                keep=checkpoint_keep,
                overwrite=True,
                orbax_checkpointer=orbax_checkpointer,
            )
            wait_for_checkpoint(orbax_checkpointer)
    else:
        if CHECKPOINT_DIR_IS_GCS and ocp is None:
            raise RuntimeError("GCS checkpointing requires orbax-checkpoint.")
        orbax_checkpointer = make_orbax_checkpointer() if CHECKPOINT_DIR_IS_GCS else None
        if orbax_checkpointer is None:
            checkpoints.save_checkpoint(
                ckpt_dir=str(step_dir),
                target=target,
                step=global_step,
                keep=checkpoint_keep,
                overwrite=True,
                orbax_checkpointer=None,
            )
        else:
            with orbax_set_mesh_context_patch():
                checkpoints.save_checkpoint(
                    ckpt_dir=str(step_dir),
                    target=target,
                    step=global_step,
                    keep=checkpoint_keep,
                    overwrite=True,
                    orbax_checkpointer=orbax_checkpointer,
                )
                wait_for_checkpoint(orbax_checkpointer)

    sync_all(f"checkpoint-saved-{global_step}")
    upload_checkpoint_to_hub(step_dir, checkpoint_name, global_step)
    sync_all(f"checkpoint-uploaded-{global_step}")
    prune_local_checkpoints()
    last_checkpoint_step = global_step
    if proc == 0:
        print(f"[Worker {proc}] Checkpoint {checkpoint_name} complete", flush=True)
        if wandb_run is not None:
            wandb.log({"checkpoint/global_step": global_step}, step=global_step)

def run_training_step(global_step: int, epoch: int, epoch_step: int, total_steps: int | None):
    global rng, total_tokens, last_epoch, last_epoch_step
    ts = time.time()
    data_t0 = time.time()
    debug_step = global_step == 1
    if debug_step:
        print(f"[Worker {proc}] step {global_step}: get_batch start", flush=True)
    raw = get_batch()
    data_dt = time.time() - data_t0
    if raw is None:
        return False
    if debug_step:
        print(f"[Worker {proc}] step {global_step}: get_batch done in {data_dt:.2f}s", flush=True)
    put_t0 = time.time()
    if debug_step:
        print(f"[Worker {proc}] step {global_step}: device_put start", flush=True)
    ids = shard_batch_array(raw["input_ids"])
    labels = shard_batch_array(raw["labels"])
    put_dt = time.time() - put_t0
    if debug_step:
        print(f"[Worker {proc}] step {global_step}: device_put done in {put_dt:.2f}s", flush=True)

    if debug_step:
        print(f"[Worker {proc}] step {global_step}: masking start", flush=True)
    rng, sk = jax.random.split(rng)
    rng_t, rng_m = jax.random.split(sk)
    maskable = jnp.ones_like(ids, dtype=bool)
    t = mdlm_cfg.time_epsilon + (1.0 - mdlm_cfg.time_epsilon) * jax.random.uniform(rng_t, (ids.shape[0],))
    p_mask = 1.0 - alpha_sched(t)[:, None]
    masked = jax.random.bernoulli(rng_m, p_mask, shape=ids.shape) & maskable
    noised = jnp.where(masked, mask_id, ids)
    w = alpha_sched.weight(t)[:, None] * jnp.ones_like(ids, dtype=jnp.float32)

    batch = {
        "input_ids": ids, "attention_mask": None, "labels": labels,
        "maskable_mask": maskable, "masked_mask": masked,
        "loss_weights": w, "model_input_ids": noised,
    }
    if debug_step:
        print(f"[Worker {proc}] step {global_step}: train_step start", flush=True)

    train_t0 = time.time()
    metrics = train_step(model, optimizer, batch)
    loss_val = float(metrics["loss"])
    train_dt = time.time() - train_t0
    ntok = GLOBAL_BATCH * MAX_LEN
    total_tokens += ntok
    last_epoch = epoch
    last_epoch_step = epoch_step
    dt = time.time() - ts
    lr_val = float(lr_schedule(global_step))
    tokens_per_second = ntok / dt
    dense_mfu = tokens_per_second * TRAIN_FLOPS_PER_TOKEN_DENSE / PEAK_FLOPS
    attention_adjusted_mfu = tokens_per_second * TRAIN_FLOPS_PER_TOKEN_TOTAL / PEAK_FLOPS

    if proc == 0:
        if total_steps is None:
            progress = f"epoch {epoch:2d}/{NUM_EPOCHS} step {epoch_step:5d} global {global_step:6d}"
        else:
            progress = f"step {global_step:3d}/{total_steps}"
        print(f"  {progress} | loss={loss_val:8.4f} | "
              f"{ntok:,} tok | {dt:5.1f}s | {tokens_per_second:,.0f} tok/s | "
              f"mfu={attention_adjusted_mfu*100:4.1f}% "
              f"(dense={dense_mfu*100:4.1f}%) | "
              f"data={data_dt:.2f}s put={put_dt:.2f}s train={train_dt:.2f}s",
              flush=True)
        if wandb_run is not None and (global_step == 1 or global_step % LOGGING_STEPS == 0):
            wandb.log(
                {
                    "train/loss": loss_val,
                    "train/lr": lr_val,
                    "train/tokens_per_second": tokens_per_second,
                    "train/tokens": total_tokens,
                    "train/step_time_seconds": dt,
                    "train/data_time_seconds": data_dt,
                    "train/device_put_seconds": put_dt,
                    "train/compiled_train_time_seconds": train_dt,
                    "train/mfu_attention_adjusted": attention_adjusted_mfu,
                    "train/mfu_dense": dense_mfu,
                    "epoch": epoch,
                    "epoch_step": epoch_step,
                },
                step=global_step,
            )
    save_training_checkpoint(global_step, epoch, epoch_step)
    return True

# ── Resume from checkpoint ────────────────────────────────
resumed_step = 0
resumed_epoch = 0
resumed_epoch_step = 0
if RESUME_DIR:
    resume_is_gcs = RESUME_DIR.startswith("gs://")

    # Find the checkpoint step to restore
    resume_step = RESUME_STEP
    if resume_step <= 0 and resume_is_gcs:
        # List checkpoint_N subdirs via google.cloud.storage (gfile unavailable on TPU VMs)
        import re
        from google.cloud import storage as gcs_storage
        bucket_name = RESUME_DIR.replace("gs://", "").split("/", 1)[0]
        prefix = RESUME_DIR.replace("gs://", "").split("/", 1)[1].rstrip("/") + "/"
        client = gcs_storage.Client()
        blobs = client.list_blobs(bucket_name, prefix=prefix, delimiter="/")
        _ = list(blobs)  # consume iterator to populate prefixes
        steps = []
        for p in blobs.prefixes:
            m = re.search(r"checkpoint_(\d+)/?$", p)
            if m:
                steps.append(int(m.group(1)))
        if steps:
            resume_step = max(steps)
        if proc == 0:
            print(f"[Worker {proc}] Found checkpoint steps: {sorted(steps)}, picking {resume_step}", flush=True)
    elif resume_step <= 0:
        # Local: find latest checkpoint_N dir
        import re
        ckpt_base = Path(RESUME_DIR)
        steps = []
        for d in ckpt_base.iterdir():
            m = re.match(r"checkpoint_(\d+)$", d.name)
            if m and d.is_dir():
                steps.append(int(m.group(1)))
        if steps:
            resume_step = max(steps)

    if resume_step <= 0:
        raise RuntimeError(f"RESUME_DIR={RESUME_DIR} specified but no checkpoints found")

    ckpt_path = RESUME_DIR.rstrip("/") + f"/checkpoint_{resume_step}"
    if proc == 0:
        print(f"[Worker {proc}] Restoring checkpoint: {ckpt_path}", flush=True)

    _resume_rng_placeholder = np.asarray(jax.random.key_data(jax.random.key(0)))
    restore_target = {
        "model": nnx.state(model),
        "optimizer": nnx.state(optimizer),
        "global_step": np.asarray(0, dtype=np.int64),
        "epoch": np.asarray(0, dtype=np.int32),
        "epoch_step": np.asarray(0, dtype=np.int64),
        "total_tokens": np.asarray(0, dtype=np.int64),
        "rng": _resume_rng_placeholder,
    }

    # Use orbax directly for GCS restore (flax's restore_checkpoint can't list GCS dirs)
    resume_ckpt = make_orbax_checkpointer()
    if proc == 0:
        print(f"[Worker {proc}] Starting orbax restore...", flush=True)
    with orbax_set_mesh_context_patch():
        restored = resume_ckpt.restore(ckpt_path, target=restore_target)
    if proc == 0:
        print(f"[Worker {proc}] Orbax restore done, re-sharding...", flush=True)

    # Re-shard restored state to the 2D mesh. Orbax restores to multi-host
    # arrays that may span non-addressable devices (so jax.device_get fails),
    # or it places scalars on a single device. fsdp_tp_sharding() uses
    # make_array_from_callback which requires a local numpy input and won't
    # work for either case. Use jax.device_put with a NamedSharding instead —
    # it handles already-placed multi-host arrays via resharding on device.
    def _target_sharding(shape):
        if len(shape) >= 2 and shape[0] >= 8 and shape[1] >= 8:
            return NamedSharding(mesh, P("fsdp", "tp"))
        if len(shape) == 1 and shape[0] >= 8:
            return NamedSharding(mesh, P("tp"))
        return NamedSharding(mesh, P())

    # Cache identity-jits by sharding so we don't re-compile per leaf.
    _reshard_jit_cache: dict = {}

    def _reshard_via_jit(x: jax.Array, sharding):
        fn = _reshard_jit_cache.get(sharding)
        if fn is None:
            fn = jax.jit(lambda a: a, out_shardings=sharding)
            _reshard_jit_cache[sharding] = fn
        return fn(x)

    from jax.experimental import multihost_utils as _mhu

    def _reshard(x):
        if isinstance(x, jax.Array):
            target = _target_sharding(x.shape)
            # If the array spans all global devices, an identity-jit reshard works.
            # Orbax often places scalars on a single device (non-addressable on
            # most hosts), which identity-jit rejects. Fall back to a multi-host
            # allgather: gather to every host as numpy, then device_put replicated.
            x_devs = set(x.devices())
            global_devs = set(mesh.devices.flat)
            if x_devs >= global_devs:
                return _reshard_via_jit(x, target)
            # Single-device or partial: allgather to numpy on all hosts
            host_val = np.asarray(_mhu.process_allgather(x, tiled=False))
            # process_allgather stacks along a new leading axis — pick the first
            # copy since the value is identical across the single-device placement.
            if host_val.ndim > x.ndim:
                host_val = host_val[0]
            return jax.device_put(host_val, target)
        if isinstance(x, (np.ndarray, jnp.ndarray)):
            return jax.device_put(np.asarray(x), _target_sharding(x.shape))
        return x

    def reshard_tree(tree, label: str):
        print(f"[Worker {proc}] Flattening {label} state...", flush=True)
        leaves, treedef = jax.tree_util.tree_flatten(
            tree,
            is_leaf=lambda x: isinstance(x, (jax.Array, jnp.ndarray, np.ndarray)),
        )
        print(f"[Worker {proc}] Flattened {label} state: {len(leaves)} leaves", flush=True)
        out = []
        for i, leaf in enumerate(leaves, start=1):
            should_log = i == 1 or i % 400 == 0 or i == len(leaves)
            if should_log:
                shape = getattr(leaf, "shape", None)
                dtype = getattr(leaf, "dtype", None)
                print(f"[Worker {proc}] Resharding {label}: leaf {i}/{len(leaves)} shape={shape} dtype={dtype}", flush=True)
            out.append(_reshard(leaf))
            if should_log:
                print(f"[Worker {proc}] Resharded {label}: leaf {i}/{len(leaves)}", flush=True)
        return jax.tree_util.tree_unflatten(treedef, out)

    gdef, _ = nnx.split(model)
    restored_model_state = reshard_tree(restored["model"], "resume-model")
    model = nnx.merge(gdef, restored_model_state)

    opt_gdef, _ = nnx.split(optimizer)
    restored_opt_state = reshard_tree(restored["optimizer"], "resume-optimizer")
    optimizer = nnx.merge(opt_gdef, restored_opt_state)
    model = optimizer.model  # rebind after merge

    resumed_step = int(restored["global_step"])
    resumed_epoch = int(restored["epoch"])
    resumed_epoch_step = int(restored["epoch_step"])
    total_tokens = int(restored["total_tokens"])
    rng = jax.random.wrap_key_data(restored["rng"])

    del restored, restore_target, _resume_rng_placeholder
    gc.collect()

    if proc == 0:
        print(f"[Worker {proc}] Resumed: step={resumed_step} epoch={resumed_epoch} "
              f"epoch_step={resumed_epoch_step} tokens={total_tokens:,}", flush=True)
    if resumed_step <= 0:
        raise RuntimeError(f"Checkpoint restore returned step=0, expected step={resume_step}")
    sync_all("resume-complete")

global_step = resumed_step
if RUN_FULL_EPOCHS:
    start_epoch = max(1, resumed_epoch) if resumed_step > 0 else 1
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        ds_iter = None if SYNTHETIC_DATA else make_dataset_iter()
        token_buffer = []
        epoch_step = 0
        # On the resumed epoch, fast-forward the data pipeline past batches
        # already consumed before the checkpoint so we don't retrain on them.
        if epoch == resumed_epoch and resumed_epoch_step > 0 and not SYNTHETIC_DATA:
            if proc == 0:
                print(f"[Worker {proc}] Fast-forwarding dataset by {resumed_epoch_step} batches...", flush=True)
            t_skip = time.time()
            skipped = 0
            for _ in range(resumed_epoch_step):
                if get_batch() is None:
                    break
                skipped += 1
                if proc == 0 and (skipped % 200 == 0 or skipped == resumed_epoch_step):
                    print(f"[Worker {proc}] Fast-forward {skipped}/{resumed_epoch_step} ({time.time()-t_skip:.1f}s)", flush=True)
            epoch_step = skipped
            if proc == 0:
                print(f"[Worker {proc}] Fast-forward done: skipped {skipped} batches in {time.time()-t_skip:.1f}s", flush=True)
            sync_all("fast-forward-done")
        while True:
            global_step += 1
            epoch_step += 1
            had_batch = run_training_step(global_step, epoch, epoch_step, total_steps=None)
            if not had_batch:
                global_step -= 1
                break
        if proc == 0:
            print(f"[Worker {proc}] Finished epoch {epoch}/{NUM_EPOCHS} after {epoch_step - 1} steps", flush=True)
else:
    for step in range(resumed_step + 1, NUM_STEPS + 1):
        global_step = step
        if not run_training_step(global_step, epoch=0, epoch_step=step, total_steps=NUM_STEPS):
            break

tt = time.time() - t_start
if CHECKPOINT_ON_FINISH:
    save_training_checkpoint(global_step, epoch=last_epoch, epoch_step=last_epoch_step, force=True)
if proc == 0:
    print(f"{'='*70}")
    print(f"Done: {global_step} steps in {tt:.1f}s | {total_tokens:,} total tokens "
          f"| avg {total_tokens/tt:,.0f} tok/s")
    if wandb_run is not None:
        wandb.finish()
print(f"[Worker {proc}] Done.", flush=True)
