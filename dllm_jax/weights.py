"""Torch-free pretrained weight loading via safetensors + numpy.

Downloads HuggingFace checkpoints and loads them directly into Flax NNX
models using safetensors' numpy framework. No PyTorch or CUDA required.
"""

from __future__ import annotations

import json
import os
import subprocess
import time

import jax
import jax.numpy as jnp


def _hf_download(model_name: str, verbose: bool = True) -> str:
    """Download model using hf CLI with hf_transfer for speed."""
    proc = jax.process_index()
    env = {**os.environ, "HF_HUB_ENABLE_HF_TRANSFER": "1"}
    hf_cli = os.path.expanduser("~/.local/bin/hf")
    if not os.path.exists(hf_cli):
        hf_cli = "hf"
    cmd = [
        hf_cli, "download", model_name,
        "--include", "*.safetensors",
        "--include", "*.safetensors.index.json",
        "--include", "config.json",
    ]
    token = os.environ.get("HF_TOKEN")
    if token:
        cmd.extend(["--token", token])

    if verbose and proc == 0:
        print(f"[Worker {proc}] Running: {' '.join(cmd)}", flush=True)

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    # Modern ``hf download`` (huggingface_hub>=0.25) decorates the final path
    # line with a ``path:`` label and can emit progress lines; prefer the
    # library call to get a clean local_dir path back.
    from huggingface_hub import snapshot_download
    if result.returncode != 0 and verbose and proc == 0:
        print(f"[Worker {proc}] hf CLI failed (rc={result.returncode})", flush=True)
        print(f"[Worker {proc}] stderr: {result.stderr[:200]}", flush=True)
    return snapshot_download(
        model_name,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
    )


def load_pretrained_weights(
    model,
    model_name_or_path: str,
    *,
    verbose: bool = True,
) -> tuple[int, list[str], list[tuple[str, tuple, tuple]]]:
    """Load HuggingFace safetensors weights into a Flax NNX model.

    Args:
        model: A GenericDecoderLM, GenericEncoderLM, or EditFlowModel instance.
        model_name_or_path: HuggingFace model name or local path.
        verbose: Print progress messages.

    Returns:
        (n_loaded, missing_keys, shape_mismatches)
    """
    import safetensors

    proc = jax.process_index()
    cpu_device = jax.devices("cpu")[0]

    if verbose and proc == 0:
        print(f"[Worker {proc}] Downloading {model_name_or_path}...", flush=True)
    t0 = time.time()

    if os.path.isdir(model_name_or_path):
        local_dir = model_name_or_path
    else:
        local_dir = _hf_download(model_name_or_path, verbose=verbose)

    if verbose and proc == 0:
        print(f"[Worker {proc}] Downloaded in {time.time() - t0:.1f}s -> {local_dir}", flush=True)

    # Determine weight file layout
    index_path = os.path.join(local_dir, "model.safetensors.index.json")
    single_path = os.path.join(local_dir, "model.safetensors")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    elif os.path.exists(single_path):
        weight_map = None
    else:
        raise FileNotFoundError(f"No safetensors files found at {local_dir}")

    # Lazy shard opener cache
    open_cache: dict = {}

    def _open(shard_file):
        full = os.path.join(local_dir, shard_file)
        if full not in open_cache:
            open_cache[full] = safetensors.safe_open(full, framework="numpy")
        return open_cache[full]

    def _get_tensor(key):
        if weight_map is None:
            f = _open("model.safetensors")
            return f.get_tensor(key) if key in f.keys() else None
        shard = weight_map.get(key)
        return _open(shard).get_tensor(key) if shard is not None else None

    missing: list[str] = []
    loaded = 0
    skipped_shape: list[tuple[str, tuple, tuple]] = []

    def _assign(nnx_param, hf_key, *, transpose: bool):
        nonlocal loaded
        arr = _get_tensor(hf_key)
        if arr is None:
            missing.append(hf_key)
            return
        if transpose:
            arr = arr.T
        target = nnx_param.value
        if tuple(arr.shape) != tuple(target.shape):
            skipped_shape.append((hf_key, tuple(arr.shape), tuple(target.shape)))
            return
        nnx_param.value = jnp.asarray(arr, dtype=target.dtype)
        loaded += 1

    # Resolve the backbone model (might be wrapped in EditFlowModel)
    mdl = model
    if hasattr(mdl, "backbone"):
        mdl = mdl.backbone

    with jax.default_device(cpu_device):
        _assign(mdl.embed_tokens.embedding, "model.embed_tokens.weight", transpose=False)

        for i, layer in enumerate(mdl.layers):
            p = f"model.layers.{i}"
            _assign(layer.attn_norm.scale, f"{p}.input_layernorm.weight", transpose=False)
            _assign(layer.self_attn.q_proj.kernel, f"{p}.self_attn.q_proj.weight", transpose=True)
            _assign(layer.self_attn.k_proj.kernel, f"{p}.self_attn.k_proj.weight", transpose=True)
            _assign(layer.self_attn.v_proj.kernel, f"{p}.self_attn.v_proj.weight", transpose=True)
            _assign(layer.self_attn.o_proj.kernel, f"{p}.self_attn.o_proj.weight", transpose=True)

            # QK norm (Qwen3, etc.)
            if getattr(layer.self_attn, "q_norm", None) is not None:
                _assign(layer.self_attn.q_norm.scale, f"{p}.self_attn.q_norm.weight", transpose=False)
                _assign(layer.self_attn.k_norm.scale, f"{p}.self_attn.k_norm.weight", transpose=False)

            _assign(layer.mlp_norm.scale, f"{p}.post_attention_layernorm.weight", transpose=False)

            # Dense MLP (gated)
            if hasattr(layer.mlp, "gate_proj") and layer.mlp.gate_proj is not None:
                _assign(layer.mlp.gate_proj.kernel, f"{p}.mlp.gate_proj.weight", transpose=True)
                _assign(layer.mlp.up_proj.kernel, f"{p}.mlp.up_proj.weight", transpose=True)
                _assign(layer.mlp.down_proj.kernel, f"{p}.mlp.down_proj.weight", transpose=True)
            # Dense MLP (non-gated, BERT-style)
            elif hasattr(layer.mlp, "fc1") and layer.mlp.fc1 is not None:
                _assign(layer.mlp.fc1.kernel, f"{p}.mlp.fc1.weight", transpose=True)
                _assign(layer.mlp.fc1.bias, f"{p}.mlp.fc1.bias", transpose=False)
                _assign(layer.mlp.fc2.kernel, f"{p}.mlp.fc2.weight", transpose=True)
                _assign(layer.mlp.fc2.bias, f"{p}.mlp.fc2.bias", transpose=False)

            # Bias terms (if present)
            if hasattr(layer.self_attn.q_proj, "bias") and layer.self_attn.q_proj.bias is not None:
                _assign(layer.self_attn.q_proj.bias, f"{p}.self_attn.q_proj.bias", transpose=False)
                _assign(layer.self_attn.k_proj.bias, f"{p}.self_attn.k_proj.bias", transpose=False)
                _assign(layer.self_attn.v_proj.bias, f"{p}.self_attn.v_proj.bias", transpose=False)
                _assign(layer.self_attn.o_proj.bias, f"{p}.self_attn.o_proj.bias", transpose=False)

        _assign(mdl.norm.scale, "model.norm.weight", transpose=False)

        if getattr(mdl, "lm_head", None) is not None:
            _assign(mdl.lm_head.kernel, "lm_head.weight", transpose=True)

    if verbose and proc == 0:
        print(f"[Worker {proc}] Loaded {loaded} pretrained tensors", flush=True)
        if missing:
            print(f"[Worker {proc}] MISSING ({len(missing)}): {missing[:5]}", flush=True)
        if skipped_shape:
            print(f"[Worker {proc}] SHAPE MISMATCH ({len(skipped_shape)}): {skipped_shape[:3]}", flush=True)

    return loaded, missing, skipped_shape
