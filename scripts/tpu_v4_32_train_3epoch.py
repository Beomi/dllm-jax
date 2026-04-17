"""Qwen3-8B MDLM full training on TPU v4-32 — 3 epochs TinyStories."""
import os
import sys
import time
import gc

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
from jax.experimental import mesh_utils
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

proc = jax.process_index()
nproc = jax.process_count()
ndev = jax.device_count()
nlocal = jax.local_device_count()
print(f"[Worker {proc}/{nproc}] devices={ndev} local={nlocal} backend={jax.default_backend()}", flush=True)

TP = 8
DP = ndev // TP
devices = mesh_utils.create_device_mesh((DP, TP))
mesh = Mesh(devices, axis_names=("fsdp", "tp"))
if proc == 0:
    print(f"[Worker {proc}] 2D Mesh: fsdp={DP} × tp={TP}", flush=True)

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
MAX_LEN = int(os.environ.get("MAX_LEN", "1024"))
GLOBAL_BATCH = int(os.environ.get("GLOBAL_BATCH", "64"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "3"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "500"))
PEAK_LR = float(os.environ.get("PEAK_LR", "1e-4"))
LOG_EVERY = int(os.environ.get("LOG_EVERY", "50"))

config = transformers.AutoConfig.from_pretrained(MODEL_NAME)
if proc == 0:
    print(f"[Worker {proc}] {MODEL_NAME}: {config.num_hidden_layers}L h={config.hidden_size} "
          f"V={config.vocab_size}", flush=True)

# ── Pallas flash attention ─────────────────────────────────
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

# ── CPU init ──────────────────────────────────────────────
if proc == 0:
    print(f"[Worker {proc}] Building model on CPU...", flush=True)
t0 = time.time()

spec = model_spec_from_config(config, task="llada")
if getattr(config, "model_type", "") == "qwen3":
    spec.use_qk_norm = True

cpu_device = jax.devices("cpu")[0]
with jax.default_device(cpu_device):
    model = GenericDecoderLM(spec, dtype_name="bfloat16", rngs=nnx.Rngs(params=42, dropout=43))

if proc == 0:
    print(f"[Worker {proc}] CPU init done in {time.time()-t0:.1f}s", flush=True)

# ── Weight loading ─────────────────────────────────────────
tload = time.time()
n_loaded, missing, shape_mismatch = load_pretrained_weights(model, MODEL_NAME, verbose=(proc == 0))
if proc == 0:
    print(f"[Worker {proc}] Loaded {n_loaded} tensors in {time.time()-tload:.1f}s", flush=True)
if missing or shape_mismatch:
    raise RuntimeError(
        f"Weight loading incomplete: missing={len(missing)} shape_mismatch={len(shape_mismatch)}"
    )

# ── Shard to 2D mesh ──────────────────────────────────────
if proc == 0:
    print(f"[Worker {proc}] Sharding to 2D TPU mesh...", flush=True)
t1 = time.time()
gdef, state = nnx.split(model)

def fsdp_tp_sharding(x):
    if isinstance(x, (jax.Array, jnp.ndarray, np.ndarray)):
        if x.ndim >= 2 and x.shape[0] >= 8 and x.shape[1] >= 8:
            return jax.device_put(jnp.asarray(x), NamedSharding(mesh, P("fsdp", "tp")))
        elif x.ndim == 1 and x.shape[0] >= 8:
            return jax.device_put(jnp.asarray(x), NamedSharding(mesh, P("tp")))
        return jax.device_put(jnp.asarray(x), NamedSharding(mesh, P()))
    return x

state = jax.tree.map(fsdp_tp_sharding, state)
model = nnx.merge(gdef, state)
del gdef, state
gc.collect()
if proc == 0:
    print(f"[Worker {proc}] Sharding done in {time.time()-t1:.1f}s", flush=True)

# ── Tokenizer + dataset ───────────────────────────────────
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

from datasets import load_dataset
import glob as globmod

LOCAL_DS = os.path.expanduser("~/tinystories_data")
parquet_files = sorted(globmod.glob(os.path.join(LOCAL_DS, "*.parquet")))
if parquet_files:
    _ds = load_dataset("parquet", data_files=parquet_files, split="train")
    if proc == 0:
        print(f"[Worker {proc}] Loaded TinyStories from local parquet: {len(_ds)} examples", flush=True)
else:
    _ds = load_dataset("roneneldan/TinyStories", split="train")
    if proc == 0:
        print(f"[Worker {proc}] Loaded TinyStories from HF: {len(_ds)} examples", flush=True)

def make_ds_iter():
    return iter(_ds)

ds_iter = make_ds_iter()
token_buffer: list[int] = []
eos_id = tokenizer.eos_token_id
current_epoch = 1
total_epoch_tokens = 0

def refill_buffer(needed):
    global token_buffer, ds_iter, current_epoch
    while len(token_buffer) < needed:
        try:
            text = next(ds_iter)["text"]
        except StopIteration:
            current_epoch += 1
            if current_epoch > NUM_EPOCHS:
                return False
            if proc == 0:
                print(f"[Worker {proc}] Starting epoch {current_epoch}/{NUM_EPOCHS}", flush=True)
            ds_iter = make_ds_iter()
            continue
        token_buffer.extend(tokenizer.encode(text, add_special_tokens=False))
        token_buffer.append(eos_id)
    return True

def get_batch():
    global token_buffer
    has_data = refill_buffer(GLOBAL_BATCH * MAX_LEN)
    if not has_data and len(token_buffer) < GLOBAL_BATCH * MAX_LEN:
        return None
    ids = np.full((GLOBAL_BATCH, MAX_LEN), tokenizer.pad_token_id, dtype=np.int64)
    for i in range(GLOBAL_BATCH):
        length = min(MAX_LEN, len(token_buffer))
        if length > 0:
            ids[i, :length] = token_buffer[:length]
            token_buffer = token_buffer[length:]
    return {"input_ids": ids, "labels": ids.copy()}

# ── Count approximate steps (first pass estimate) ─────────
# TinyStories ~2.12M examples, ~224 tok mean → ~475M tok/epoch
EST_TOKENS_PER_EPOCH = 475_000_000
TOKENS_PER_STEP = GLOBAL_BATCH * MAX_LEN
EST_TOTAL_STEPS = (EST_TOKENS_PER_EPOCH * NUM_EPOCHS) // TOKENS_PER_STEP
if proc == 0:
    print(f"[Worker {proc}] Estimated ~{EST_TOTAL_STEPS:,} steps for {NUM_EPOCHS} epochs", flush=True)

# ── Optimizer: AdamW + cosine schedule ─────────────────────
lr_schedule = optax.join_schedules(
    schedules=[
        optax.linear_schedule(0.0, PEAK_LR, WARMUP_STEPS),
        optax.cosine_decay_schedule(PEAK_LR, EST_TOTAL_STEPS - WARMUP_STEPS, alpha=0.01),
    ],
    boundaries=[WARMUP_STEPS],
)
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=lr_schedule, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.01),
)
optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

opt_gdef, opt_state = nnx.split(optimizer)
opt_state = jax.tree.map(fsdp_tp_sharding, opt_state)
optimizer = nnx.merge(opt_gdef, opt_state)
del opt_gdef, opt_state
gc.collect()
model = optimizer.model
if proc == 0:
    print(f"[Worker {proc}] Optimizer ready (AdamW cosine, model rebound)", flush=True)

# ── MDLM config ───────────────────────────────────────────
mdlm_cfg = MDLMConfig(output_dir="/tmp/q3-8b-full", max_steps=EST_TOTAL_STEPS, learning_rate=PEAK_LR)
alpha_sched = LinearAlphaScheduler()
mask_id = config.vocab_size - 1

# ── Loss + step ────────────────────────────────────────────
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

if proc == 0:
    print(f"\n{'='*70}")
    print(f"MDLM full training: Qwen3-8B × TinyStories × {NUM_EPOCHS} epochs")
    print(f"  {ndev} chips | mesh=({DP},{TP}) fsdp×tp")
    print(f"  seq={MAX_LEN} batch={GLOBAL_BATCH} est_steps=~{EST_TOTAL_STEPS:,}")
    print(f"  tokens/step={TOKENS_PER_STEP:,}")
    print(f"  lr={PEAK_LR} warmup={WARMUP_STEPS} schedule=cosine")
    print(f"  PRETRAINED: {n_loaded} tensors")
    print(f"{'='*70}", flush=True)

rng = jax.random.key(42)
total_tokens = 0
step = 0
t_start = time.time()
loss_accum = 0.0
loss_count = 0

while True:
    ts = time.time()
    raw = get_batch()
    if raw is None:
        break
    step += 1

    ids = jax.device_put(jnp.asarray(raw["input_ids"]), data_sharding)
    labels = jax.device_put(jnp.asarray(raw["labels"]), data_sharding)

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

    metrics = train_step(model, optimizer, batch)
    loss_val = float(metrics["loss"])
    ntok = TOKENS_PER_STEP
    total_tokens += ntok
    dt = time.time() - ts
    loss_accum += loss_val
    loss_count += 1

    if proc == 0 and (step % LOG_EVERY == 0 or step == 1):
        avg_loss = loss_accum / loss_count
        elapsed = time.time() - t_start
        print(f"  step {step:6d} | epoch {current_epoch}/{NUM_EPOCHS} | "
              f"loss={avg_loss:8.4f} (cur={loss_val:8.4f}) | "
              f"{ntok:,} tok | {dt:5.1f}s | {ntok/dt:,.0f} tok/s | "
              f"elapsed={elapsed/3600:.1f}h | total={total_tokens:,} tok",
              flush=True)
        loss_accum = 0.0
        loss_count = 0

tt = time.time() - t_start
if proc == 0:
    print(f"{'='*70}")
    print(f"Done: {step} steps, {NUM_EPOCHS} epochs in {tt/3600:.1f}h | "
          f"{total_tokens:,} total tokens | avg {total_tokens/tt:,.0f} tok/s")
print(f"[Worker {proc}] Done.", flush=True)
