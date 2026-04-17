"""Run a short DMax inference smoke test from a distributed TPU checkpoint."""

from __future__ import annotations

import os
import re
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import transformers
from flax import nnx
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from dllm_jax import (
    GenericDecoderLM,
    dmax_generate_spd,
    dmax_generate_spd_fast,
    dmax_generate_spd_kv_fast,
    model_spec_from_config,
)

try:
    import optax
except ImportError:
    optax = None

try:
    import orbax.checkpoint as ocp
    from orbax.checkpoint import args as ocp_args
    from orbax.checkpoint import options as ocp_options
    from orbax.checkpoint import type_handlers as ocp_type_handlers
except ImportError as exc:
    raise RuntimeError("orbax-checkpoint is required for TPU checkpoint restore.") from exc


def load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if os.environ.get("DLLM_DISABLE_LIBTPU_TUNING", "").lower() not in {"1", "true", "yes", "on"}:
    os.environ["LIBTPU_INIT_ARGS"] = " ".join(
        [
            "--xla_tpu_enable_async_collective_fusion=true",
            "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true",
            "--xla_tpu_overlap_compute_collective_tc=true",
            "--xla_enable_async_all_gather=true",
            "--xla_tpu_data_parallel_opt_different_sized_ops=true",
        ]
    )


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


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


def detect_tpu_zone() -> str | None:
    for name in ("TPU_ZONE", "ZONE", "CLOUDSDK_COMPUTE_ZONE", "GOOGLE_CLOUD_ZONE"):
        zone = normalize_zone(os.environ.get(name))
        if zone:
            return zone
    return normalize_zone(metadata_value("instance/zone"))


def sync_all(label: str) -> None:
    if jax.process_count() > 1:
        multihost_utils.sync_global_devices(label)


def latest_committed_gcs_step(resume_dir: str) -> int:
    from google.cloud import storage as gcs_storage

    bucket_name, prefix = resume_dir.replace("gs://", "").split("/", 1)
    prefix = prefix.rstrip("/") + "/"
    client = gcs_storage.Client()
    steps: list[int] = []
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        if not blob.name.endswith("/commit_success.txt"):
            continue
        match = re.search(r"checkpoint_(\d+)/commit_success\.txt$", blob.name)
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        raise RuntimeError(f"No committed checkpoints found under {resume_dir}")
    return max(steps)


def jax_distributed_client_initialized() -> bool:
    try:
        import jax._src.distributed as jax_distributed  # pylint: disable=import-outside-toplevel

        return jax_distributed.global_state.client is not None
    except Exception:
        return False


_orbax_signal_fallback_installed = False


def maybe_install_orbax_signal_fallback() -> None:
    global _orbax_signal_fallback_installed
    if (
        _orbax_signal_fallback_installed
        or not env_flag("CHECKPOINT_ORBAX_SIGNAL_FALLBACK", True)
        or jax.process_count() <= 1
        or jax_distributed_client_initialized()
    ):
        return
    try:
        from orbax.checkpoint._src.futures import future as orbax_future  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        if jax.process_index() == 0:
            print(f"[Worker 0] Orbax signal fallback unavailable: {exc}", flush=True)
        return
    orbax_future.AwaitableSignalsContract.get_awaitable_signals_from_contract = classmethod(
        lambda cls, operation_id=None: []
    )
    _orbax_signal_fallback_installed = True
    if jax.process_index() == 0:
        print("[Worker 0] Orbax signal-contract fallback enabled", flush=True)


def make_orbax_checkpointer():
    maybe_install_orbax_signal_fallback()
    if ocp_options is None:
        return ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

    def barrier_sync_fn(key: str, timeout_ms: int) -> None:
        del timeout_ms
        multihost_utils.sync_global_devices(key)

    multiprocessing_options = ocp_options.MultiprocessingOptions(primary_host=0)
    async_options = ocp_options.AsyncOptions(
        barrier_sync_fn=barrier_sync_fn,
        create_directories_asynchronously=False,
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


def make_pytree_checkpointer():
    maybe_install_orbax_signal_fallback()
    if ocp_options is None:
        return ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    multiprocessing_options = ocp_options.MultiprocessingOptions(primary_host=0)
    return ocp.Checkpointer(
        ocp.PyTreeCheckpointHandler(multiprocessing_options=multiprocessing_options)
    )


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


def build_mesh() -> Mesh:
    tp = int(os.environ.get("TP", "8"))
    ndev = jax.device_count()
    if ndev % tp != 0:
        raise ValueError(f"device_count={ndev} must be divisible by TP={tp}")
    devices = mesh_utils.create_device_mesh((ndev // tp, tp))
    return Mesh(devices, axis_names=("fsdp", "tp"))


def sharding_for_shape(mesh: Mesh, shape: tuple[int, ...]) -> NamedSharding:
    if len(shape) >= 2 and shape[0] >= 8 and shape[1] >= 8:
        return NamedSharding(mesh, P("fsdp", "tp"))
    if len(shape) == 1 and shape[0] >= 8:
        return NamedSharding(mesh, P("tp"))
    return NamedSharding(mesh, P())


def shard_tree_to_mesh(tree: Any, mesh: Mesh, label: str) -> Any:
    proc = jax.process_index()
    leaves, treedef = jax.tree_util.tree_flatten(
        tree,
        is_leaf=lambda x: isinstance(x, (jax.Array, jnp.ndarray, np.ndarray)),
    )
    print(f"[Worker {proc}] Flattened {label}: {len(leaves)} leaves", flush=True)
    out = []
    for i, leaf in enumerate(leaves, start=1):
        should_log = i == 1 or i % 400 == 0 or i == len(leaves)
        if should_log:
            print(
                f"[Worker {proc}] Sharding {label}: leaf {i}/{len(leaves)} "
                f"shape={getattr(leaf, 'shape', None)} dtype={getattr(leaf, 'dtype', None)}",
                flush=True,
            )
        if isinstance(leaf, jax.Array) and any(d.platform != "cpu" for d in leaf.devices()):
            out.append(leaf)
            continue
        arr = np.asarray(leaf)
        sharding = sharding_for_shape(mesh, arr.shape)
        out.append(jax.make_array_from_callback(arr.shape, sharding, lambda idx, x=arr: x[idx]))
    return jax.tree_util.tree_unflatten(treedef, out)


def reshard_restored_tree(tree: Any, mesh: Mesh, label: str) -> Any:
    proc = jax.process_index()
    leaves, treedef = jax.tree_util.tree_flatten(
        tree,
        is_leaf=lambda x: isinstance(x, (jax.Array, jnp.ndarray, np.ndarray)),
    )
    print(f"[Worker {proc}] Flattened restored {label}: {len(leaves)} leaves", flush=True)
    cache: dict[Any, Any] = {}
    global_devs = set(mesh.devices.flat)

    def reshard_leaf(x):
        if isinstance(x, jax.Array):
            sharding = sharding_for_shape(mesh, x.shape)
            if set(x.devices()) >= global_devs:
                fn = cache.get(sharding)
                if fn is None:
                    fn = jax.jit(lambda a: a, out_shardings=sharding)
                    cache[sharding] = fn
                return fn(x)
            host_val = np.asarray(multihost_utils.process_allgather(x, tiled=False))
            if host_val.ndim > x.ndim:
                host_val = host_val[0]
            return jax.device_put(host_val, sharding)
        if isinstance(x, (np.ndarray, jnp.ndarray)):
            arr = np.asarray(x)
            return jax.device_put(arr, sharding_for_shape(mesh, arr.shape))
        return x

    out = []
    for i, leaf in enumerate(leaves, start=1):
        should_log = i == 1 or i % 400 == 0 or i == len(leaves)
        if should_log:
            print(
                f"[Worker {proc}] Resharding {label}: leaf {i}/{len(leaves)} "
                f"shape={getattr(leaf, 'shape', None)} dtype={getattr(leaf, 'dtype', None)}",
                flush=True,
            )
        out.append(reshard_leaf(leaf))
    return jax.tree_util.tree_unflatten(treedef, out)


def restore_args_for_tree(tree: Any) -> Any:
    def is_array_leaf(x: Any) -> bool:
        return isinstance(x, (jax.Array, jnp.ndarray, np.ndarray))

    def to_restore_args(x: Any) -> Any:
        if isinstance(x, jax.Array):
            return ocp_type_handlers.ArrayRestoreArgs(
                restore_type=jax.Array,
                dtype=x.dtype,
                sharding=x.sharding,
                global_shape=x.shape,
            )
        if isinstance(x, (jnp.ndarray, np.ndarray)):
            arr = np.asarray(x)
            return ocp_type_handlers.RestoreArgs(restore_type=np.ndarray, dtype=arr.dtype)
        return ocp_type_handlers.RestoreArgs()

    return jax.tree_util.tree_map(to_restore_args, tree, is_leaf=is_array_leaf)


def build_optimizer_if_requested(model: Any, mesh: Mesh):
    if not env_flag("RESTORE_OPTIMIZER", False):
        return None
    if optax is None:
        raise RuntimeError("RESTORE_OPTIMIZER=1 requires optax.")
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=1e-4, b1=0.9, b2=0.95, eps=1e-8, weight_decay=0.01),
    )
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)
    opt_gdef, opt_state = nnx.split(optimizer)
    opt_state = shard_tree_to_mesh(opt_state, mesh, "optimizer")
    return nnx.merge(opt_gdef, opt_state)


def main() -> None:
    proc = jax.process_index()
    nproc = jax.process_count()
    ndev = jax.device_count()
    nlocal = jax.local_device_count()
    print(
        f"[Worker {proc}/{nproc}] devices={ndev} local={nlocal} backend={jax.default_backend()}",
        flush=True,
    )

    resume_dir = os.environ.get("RESUME_DIR") or os.environ.get("RESUME_FROM")
    if not resume_dir:
        raise ValueError("Set RESUME_DIR to the checkpoint parent directory.")

    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
    prompt = os.environ.get("PROMPT", "Once upon a time")
    gen_length = int(os.environ.get("GEN_LENGTH", "32"))
    block_length = int(os.environ.get("BLOCK_LENGTH", "32"))
    steps = int(os.environ.get("STEPS", "8"))
    threshold = float(os.environ.get("THRESHOLD", "0.95"))
    confidence_stop = float(os.environ.get("CONFIDENCE_STOP", "0.9"))
    suppress_mask_token = env_flag("SUPPRESS_MASK_TOKEN", False)
    infer_impl = os.environ.get("INFER_IMPL", "fast").strip().lower()
    fast_bucket_length = int(os.environ.get("FAST_BUCKET_LENGTH", "4096"))
    temperature = float(os.environ.get("TEMPERATURE", "0.0"))
    top_k = int(os.environ.get("TOP_K", "1"))
    seed_env = os.environ.get("SEED")
    seed = int(seed_env) if seed_env else None
    dtype_name = os.environ.get("DTYPE", "bfloat16")
    mask_token_id = int(os.environ["MASK_TOKEN_ID"]) if os.environ.get("MASK_TOKEN_ID") else None
    eos_token_id = int(os.environ["EOS_TOKEN_ID"]) if os.environ.get("EOS_TOKEN_ID") else None

    mesh = build_mesh()
    if proc == 0:
        zone = detect_tpu_zone()
        print(
            f"[Worker 0] mesh fsdp={mesh.shape['fsdp']} tp={mesh.shape['tp']} zone={zone}",
            flush=True,
        )

    resume_step = int(os.environ.get("RESUME_STEP", "0"))
    if resume_step <= 0:
        if resume_dir.startswith("gs://"):
            resume_step = latest_committed_gcs_step(resume_dir)
        else:
            steps_found = [
                int(m.group(1))
                for p in Path(resume_dir).iterdir()
                if (m := re.match(r"checkpoint_(\d+)$", p.name)) and p.is_dir()
            ]
            if not steps_found:
                raise RuntimeError(f"No checkpoints found under {resume_dir}")
            resume_step = max(steps_found)
    ckpt_path = resume_dir.rstrip("/") + f"/checkpoint_{resume_step}"

    t0 = time.time()
    if proc == 0:
        print(f"[Worker 0] Loading config/tokenizer: {model_name}", flush=True)
    config = transformers.AutoConfig.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    spec = model_spec_from_config(config, task="llada")
    if getattr(config, "model_type", "") == "qwen3":
        spec.use_qk_norm = True

    if proc == 0:
        print(f"[Worker 0] Building model on CPU: {config.num_hidden_layers} layers", flush=True)
    with jax.default_device(jax.devices("cpu")[0]):
        model = GenericDecoderLM(spec, dtype_name=dtype_name, rngs=nnx.Rngs(params=42, dropout=43))

    gdef, state = nnx.split(model)
    state = shard_tree_to_mesh(state, mesh, "model")
    model = nnx.merge(gdef, state)
    optimizer = build_optimizer_if_requested(model, mesh)
    if optimizer is not None:
        model = optimizer.model

    target = {
        "model": nnx.state(model),
        "global_step": np.asarray(0, dtype=np.int64),
        "epoch": np.asarray(0, dtype=np.int32),
        "epoch_step": np.asarray(0, dtype=np.int64),
        "total_tokens": np.asarray(0, dtype=np.int64),
        "rng": np.asarray(jax.random.key_data(jax.random.key(0))),
    }
    if optimizer is not None:
        target["optimizer"] = nnx.state(optimizer)

    sync_all("infer-before-restore")
    if proc == 0:
        print(f"[Worker 0] Restoring checkpoint: {ckpt_path}", flush=True)
    checkpointer = make_orbax_checkpointer() if optimizer is not None else make_pytree_checkpointer()
    with orbax_set_mesh_context_patch():
        if optimizer is None:
            restore_args = restore_args_for_tree(target)
            restored = checkpointer.restore(
                ckpt_path,
                args=ocp_args.PyTreeRestore(
                    item=target,
                    restore_args=restore_args,
                    partial_restore=True,
                ),
            )
        else:
            try:
                restored = checkpointer.restore(ckpt_path, target=target, strict=False)
            except TypeError:
                restored = checkpointer.restore(ckpt_path, target=target)

    if proc == 0:
        print("[Worker 0] Restore done; resharding model", flush=True)
    gdef, _ = nnx.split(model)
    restored_model_state = reshard_restored_tree(restored["model"], mesh, "model")
    model = nnx.merge(gdef, restored_model_state)

    restored_step = int(np.asarray(multihost_utils.process_allgather(restored["global_step"], tiled=False))[0])
    restored_epoch = int(np.asarray(multihost_utils.process_allgather(restored["epoch"], tiled=False))[0])
    restored_epoch_step = int(np.asarray(multihost_utils.process_allgather(restored["epoch_step"], tiled=False))[0])
    sync_all("infer-restore-complete")

    input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_arr = np.asarray([input_ids], dtype=np.int32)
    input_arr = jax.device_put(input_arr, NamedSharding(mesh, P()))

    if proc == 0:
        print(
            f"[Worker 0] Generating: prompt_tokens={len(input_ids)} gen_length={gen_length} "
            f"block_length={block_length} steps={steps} threshold={threshold} "
            f"confidence_stop={confidence_stop} suppress_mask_token={suppress_mask_token} "
            f"temperature={temperature} top_k={top_k} seed={seed} "
            f"impl={infer_impl} "
            f"fast_bucket_length={fast_bucket_length if infer_impl in {'fast', 'compiled', 'jit', 'jitted'} else 'n/a'}",
            flush=True,
        )
    tg = time.time()
    if infer_impl in {"fast", "compiled", "jit", "jitted"}:
        generate_fn = dmax_generate_spd_fast
    elif infer_impl in {"legacy", "python", "slow"}:
        generate_fn = dmax_generate_spd
    elif infer_impl in {"kv_fast", "kv", "cache", "prefix"}:
        generate_fn = dmax_generate_spd_kv_fast
    else:
        raise ValueError(
            f"Unknown INFER_IMPL={infer_impl!r}; use fast, legacy, or kv_fast."
        )
    generate_kwargs = {
        "tokenizer": tokenizer,
        "gen_length": gen_length,
        "block_length": block_length,
        "steps": steps,
        "threshold": threshold,
        "confidence_stop": confidence_stop,
        "mask_token_id": mask_token_id,
        "eos_token_id": eos_token_id,
        "suppress_mask_token": suppress_mask_token,
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
    }
    if generate_fn is dmax_generate_spd_fast:
        generate_kwargs["bucket_length"] = fast_bucket_length
    # kv_fast does not use bucket_length (it runs a single compiled while_loop
    # over blocks with a pre-allocated KV cache of size ``total_length``).
    output = generate_fn(model, input_arr, **generate_kwargs)
    output.generated_tokens.block_until_ready()
    nfe = int(np.asarray(output.nfe))
    generated_all = np.asarray(multihost_utils.process_allgather(output.generated_tokens, tiled=False))
    sync_all("infer-generation-complete")

    if proc == 0:
        generated = generated_all[0, 0].tolist() if generated_all.ndim == 3 else generated_all[0].tolist()
        print("=" * 70)
        print(f"checkpoint_step={restored_step} epoch={restored_epoch} epoch_step={restored_epoch_step}")
        print(f"prompt={prompt!r}")
        print("generated:")
        print(tokenizer.decode(generated, skip_special_tokens=True))
        print(f"nfe={nfe} generated_tokens={len(generated)}")
        print(f"restore_plus_generate_seconds={time.time() - t0:.1f}")
        print(f"generate_seconds={time.time() - tg:.1f}")
        print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
