"""NNX model definitions — GenericDecoderLM, GenericEncoderLM, EditFlowModel.

No PyTorch dependency. Uses transformers only for PretrainedConfig parsing.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
import transformers

# Compatibility: flax >= 0.11 has nnx.List; older versions use plain list()
_nnx_list = getattr(nnx, "List", list)

# Flash attention hook: set by training script to a shard_map-wrapped Pallas kernel.
# Signature: _FLASH_ATTN_FN(q, k, v, sm_scale) with layout (batch, heads, seq, dim).
_FLASH_ATTN_FN = None


def get_dtype(name: str):
    if name == "float32":
        return jnp.float32
    if name == "float16":
        return jnp.float16
    return jnp.bfloat16


def activation_fn(name: str):
    lowered = name.lower()
    if lowered in {"gelu", "gelu_new"}:
        return jax.nn.gelu
    if lowered in {"relu"}:
        return jax.nn.relu
    if lowered in {"tanh"}:
        return jnp.tanh
    return jax.nn.silu


def normal_init(std: float):
    return jax.nn.initializers.normal(stddev=std)


@dataclasses.dataclass
class ModelSpec:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    hidden_act: str
    initializer_range: float
    attention_dropout: float
    hidden_dropout: float
    rope_theta: float
    partial_rotary_factor: float
    tie_word_embeddings: bool
    use_bias: bool
    norm_type: str
    norm_eps: float
    is_encoder: bool
    use_moe: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int | None
    num_shared_experts: int
    first_k_dense_replace: int
    moe_layer_freq: tuple[int, ...]
    use_qk_norm: bool
    model_type: str


def _first_not_none(*values, default=None):
    for value in values:
        if value is not None:
            return value
    return default


def model_spec_from_config(config: transformers.PretrainedConfig, task: str) -> ModelSpec:
    model_type = getattr(config, "model_type", type(config).__name__.lower())
    hidden_size = int(
        _first_not_none(
            getattr(config, "hidden_size", None),
            getattr(config, "d_model", None),
        )
    )
    num_hidden_layers = int(
        _first_not_none(
            getattr(config, "num_hidden_layers", None),
            getattr(config, "n_layers", None),
            default=1,
        )
    )
    num_attention_heads = int(
        _first_not_none(
            getattr(config, "num_attention_heads", None),
            getattr(config, "n_heads", None),
            default=1,
        )
    )
    num_key_value_heads = int(
        _first_not_none(
            getattr(config, "num_key_value_heads", None),
            default=num_attention_heads,
        )
    )
    if num_key_value_heads <= 0:
        num_key_value_heads = num_attention_heads

    intermediate_size = int(
        _first_not_none(
            getattr(config, "intermediate_size", None),
            getattr(config, "dense_intermediate_size", None),
            getattr(config, "mlp_hidden_size", None),
            default=hidden_size * int(getattr(config, "mlp_ratio", 4)),
        )
    )
    moe_intermediate_size = int(
        _first_not_none(
            getattr(config, "moe_intermediate_size", None),
            getattr(config, "expert_intermediate_size", None),
            default=intermediate_size,
        )
    )
    shared_expert_intermediate_size = getattr(config, "shared_expert_intermediate_size", None)
    if shared_expert_intermediate_size is not None:
        shared_expert_intermediate_size = int(shared_expert_intermediate_size)

    is_encoder = task == "bert" or "bert" in model_type
    norm_type = "layernorm" if is_encoder else "rmsnorm"
    norm_eps = float(
        _first_not_none(
            getattr(config, "layer_norm_eps", None),
            getattr(config, "rms_norm_eps", None),
            default=1e-5,
        )
    )
    hidden_dropout = float(
        _first_not_none(
            getattr(config, "hidden_dropout", None),
            getattr(config, "embedding_dropout", None),
            getattr(config, "output_dropout", None),
            getattr(config, "resid_pdrop", None),
            default=0.0,
        )
    )
    attention_dropout = float(getattr(config, "attention_dropout", 0.0))
    use_bias = bool(
        _first_not_none(
            getattr(config, "use_bias", None),
            getattr(config, "attention_bias", None),
            default=is_encoder,
        )
    )
    num_experts = int(max(int(getattr(config, "num_experts", 0) or 0), 0))
    num_experts_per_tok = int(
        max(int(getattr(config, "num_experts_per_tok", 1) or 1), 1)
    )

    return ModelSpec(
        vocab_size=int(config.vocab_size),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=int(
            getattr(config, "max_position_embeddings", hidden_size)
        ),
        hidden_act=str(getattr(config, "hidden_act", "silu")),
        initializer_range=float(getattr(config, "initializer_range", 0.02)),
        attention_dropout=attention_dropout,
        hidden_dropout=hidden_dropout,
        rope_theta=float(getattr(config, "rope_theta", 10000.0)),
        partial_rotary_factor=float(getattr(config, "partial_rotary_factor", 1.0)),
        tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", True)),
        use_bias=use_bias,
        norm_type=norm_type,
        norm_eps=norm_eps,
        is_encoder=is_encoder,
        use_moe=num_experts > 0,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        num_shared_experts=int(getattr(config, "num_shared_experts", 0) or 0),
        first_k_dense_replace=int(getattr(config, "first_k_dense_replace", 0) or 0),
        moe_layer_freq=tuple(getattr(config, "moe_layer_freq", ())),
        use_qk_norm=bool(
            _first_not_none(
                getattr(config, "use_qk_norm", None),
                getattr(config, "qk_layernorm", None),
                default=False,
            )
        ),
        model_type=model_type,
    )


def _make_norm(spec: ModelSpec, rngs: nnx.Rngs):
    if spec.norm_type == "layernorm":
        return nnx.LayerNorm(
            spec.hidden_size, epsilon=spec.norm_eps, use_bias=True, rngs=rngs,
        )
    return nnx.RMSNorm(spec.hidden_size, epsilon=spec.norm_eps, rngs=rngs)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def build_rope(position_ids, rotary_dim: int, theta: float, dtype):
    if rotary_dim <= 0:
        return None, None
    rotary_dim = rotary_dim - (rotary_dim % 2)
    inv_freq = 1.0 / (
        theta ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.float32) / rotary_dim)
    )
    freqs = jnp.einsum("bl,d->bld", position_ids.astype(jnp.float32), inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def apply_rope(q, k, cos, sin, rotary_dim: int):
    if cos is None or sin is None or rotary_dim <= 0:
        return q, k
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    cos = cos[:, :, None, :]
    sin = sin[:, :, None, :]
    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return (
        jnp.concatenate([q_rot, q_pass], axis=-1),
        jnp.concatenate([k_rot, k_pass], axis=-1),
    )


def expand_attention_mask(attention_mask, batch_size: int, query_len: int, key_len: int):
    if attention_mask is None:
        return None
    mask = jnp.asarray(attention_mask).astype(bool)
    if mask.ndim == 2:
        if mask.shape == (query_len, key_len):
            return mask[None, None, :, :]
        return jnp.broadcast_to(mask[:, None, None, :], (batch_size, 1, query_len, key_len))
    if mask.ndim == 3:
        return mask[:, None, :, :]
    if mask.ndim == 4:
        return mask
    raise ValueError(f"Unsupported attention mask rank: {mask.ndim}")


def layer_uses_moe(spec: ModelSpec, layer_idx: int) -> bool:
    if not spec.use_moe:
        return False
    if spec.moe_layer_freq and layer_idx < len(spec.moe_layer_freq):
        return bool(spec.moe_layer_freq[layer_idx])
    if spec.first_k_dense_replace:
        return layer_idx >= spec.first_k_dense_replace
    return True


class DenseMLP(nnx.Module):
    def __init__(self, spec: ModelSpec, intermediate_size: int, *, gated: bool, rngs: nnx.Rngs):
        init = normal_init(spec.initializer_range)
        self.gated = gated
        if gated:
            self.gate_proj = nnx.Linear(spec.hidden_size, intermediate_size, use_bias=spec.use_bias, kernel_init=init, bias_init=jax.nn.initializers.zeros, rngs=rngs)
            self.up_proj = nnx.Linear(spec.hidden_size, intermediate_size, use_bias=spec.use_bias, kernel_init=init, bias_init=jax.nn.initializers.zeros, rngs=rngs)
            self.down_proj = nnx.Linear(intermediate_size, spec.hidden_size, use_bias=spec.use_bias, kernel_init=init, bias_init=jax.nn.initializers.zeros, rngs=rngs)
            self.fc1 = None
            self.fc2 = None
        else:
            self.gate_proj = None
            self.up_proj = None
            self.down_proj = None
            self.fc1 = nnx.Linear(spec.hidden_size, intermediate_size, use_bias=True, kernel_init=init, bias_init=jax.nn.initializers.zeros, rngs=rngs)
            self.fc2 = nnx.Linear(intermediate_size, spec.hidden_size, use_bias=True, kernel_init=init, bias_init=jax.nn.initializers.zeros, rngs=rngs)
        self.act = activation_fn(spec.hidden_act)

    def __call__(self, hidden_states):
        if self.gated:
            return self.down_proj(self.act(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))
        return self.fc2(self.act(self.fc1(hidden_states)))


class MoeMLP(nnx.Module):
    def __init__(self, spec: ModelSpec, *, rngs: nnx.Rngs):
        init = normal_init(spec.initializer_range)
        self.num_experts = spec.num_experts
        self.top_k = min(spec.num_experts_per_tok, spec.num_experts)
        self.gate = nnx.Linear(spec.hidden_size, spec.num_experts, use_bias=False, kernel_init=init, rngs=rngs)
        self.experts = _nnx_list([DenseMLP(spec, spec.moe_intermediate_size, gated=True, rngs=rngs) for _ in range(spec.num_experts)])
        if spec.shared_expert_intermediate_size:
            self.shared_expert = DenseMLP(spec, spec.shared_expert_intermediate_size, gated=True, rngs=rngs)
        else:
            self.shared_expert = None

    def __call__(self, hidden_states):
        flat_states = hidden_states.reshape((-1, hidden_states.shape[-1]))
        router_logits = self.gate(flat_states)
        router_weights = jax.nn.softmax(router_logits, axis=-1)
        top_values, top_indices = jax.lax.top_k(router_weights, self.top_k)
        top_values = top_values / jnp.clip(top_values.sum(axis=-1, keepdims=True), min=1e-6)
        expert_outputs = jnp.stack([expert(flat_states) for expert in self.experts], axis=1)
        chosen_outputs = jnp.take_along_axis(expert_outputs, top_indices[..., None], axis=1)
        mixed = jnp.sum(chosen_outputs * top_values[..., None], axis=1)
        if self.shared_expert is not None:
            mixed = mixed + self.shared_expert(flat_states)
        return mixed.reshape(hidden_states.shape)


class SelfAttention(nnx.Module):
    def __init__(self, spec: ModelSpec, *, use_rotary: bool, rngs: nnx.Rngs):
        init = normal_init(spec.initializer_range)
        self.num_heads = spec.num_attention_heads
        self.num_kv_heads = spec.num_key_value_heads
        self.hidden_size = spec.hidden_size
        self.head_dim = spec.hidden_size // spec.num_attention_heads
        self.use_rotary = use_rotary
        self.rotary_dim = int(self.head_dim * spec.partial_rotary_factor) if use_rotary else 0
        self.rope_theta = spec.rope_theta
        self.q_proj = nnx.Linear(spec.hidden_size, self.num_heads * self.head_dim, use_bias=spec.use_bias, kernel_init=init, bias_init=jax.nn.initializers.zeros, rngs=rngs)
        self.k_proj = nnx.Linear(spec.hidden_size, self.num_kv_heads * self.head_dim, use_bias=spec.use_bias, kernel_init=init, bias_init=jax.nn.initializers.zeros, rngs=rngs)
        self.v_proj = nnx.Linear(spec.hidden_size, self.num_kv_heads * self.head_dim, use_bias=spec.use_bias, kernel_init=init, bias_init=jax.nn.initializers.zeros, rngs=rngs)
        self.o_proj = nnx.Linear(self.num_heads * self.head_dim, spec.hidden_size, use_bias=spec.use_bias, kernel_init=init, bias_init=jax.nn.initializers.zeros, rngs=rngs)
        if spec.use_qk_norm:
            self.q_norm = nnx.RMSNorm(self.head_dim, epsilon=spec.norm_eps, rngs=rngs)
            self.k_norm = nnx.RMSNorm(self.head_dim, epsilon=spec.norm_eps, rngs=rngs)
        else:
            self.q_norm = None
            self.k_norm = None

    def _project_qkv(self, hidden_states, position_ids):
        batch_size, query_len, _ = hidden_states.shape
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.arange(query_len)[None, :], (batch_size, query_len)
            )
        q = self.q_proj(hidden_states).reshape(batch_size, query_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(batch_size, query_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(batch_size, query_len, self.num_kv_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = build_rope(position_ids, self.rotary_dim, self.rope_theta, q.dtype)
        q, k = apply_rope(q, k, cos, sin, self.rotary_dim)
        return q, k, v

    def _attention(self, q, k, v, attention_mask):
        batch_size, query_len, _, _ = q.shape
        key_len = k.shape[1]
        if self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, repeats, axis=2)
            v = jnp.repeat(v, repeats, axis=2)
        if _FLASH_ATTN_FN is not None and attention_mask is None:
            output = _FLASH_ATTN_FN(
                q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3), v.transpose(0, 2, 1, 3),
                1.0 / math.sqrt(self.head_dim),
            ).transpose(0, 2, 1, 3).reshape(batch_size, query_len, self.num_heads * self.head_dim)
        else:
            mask = expand_attention_mask(attention_mask, batch_size, query_len, key_len)
            output = jax.nn.dot_product_attention(q, k, v, mask=mask).reshape(
                batch_size, query_len, self.num_heads * self.head_dim,
            )
        return self.o_proj(output)

    def __call__(self, hidden_states, *, attention_mask=None, position_ids=None):
        q, k, v = self._project_qkv(hidden_states, position_ids)
        return self._attention(q, k, v, attention_mask)

    def call_cached(
        self,
        hidden_states,
        past_k,
        past_v,
        cache_position,
        *,
        attention_mask,
        position_ids,
    ):
        """Run attention using a KV cache buffer.

        ``past_k`` / ``past_v`` have shape ``[B, max_seq, num_kv_heads, head_dim]``
        and hold the cached K/V for earlier positions. The current step's K/V
        (computed from ``hidden_states``, which has ``query_len`` positions
        starting at ``cache_position``) is written into the buffers at
        ``[cache_position, cache_position + query_len)`` via
        ``jax.lax.dynamic_update_slice``. Attention then runs over the full
        updated buffers.
        """

        q, k_new, v_new = self._project_qkv(hidden_states, position_ids)
        # The cache buffers fix the attention precision: ``past_k`` should be
        # in the same dtype as Q/K coming out of ``q_norm`` / ``k_norm`` (often
        # float32 for RMSNorm), and ``past_v`` in the model's compute dtype.
        # ``jax.nn.dot_product_attention`` requires Q and K to share dtype;
        # V may differ.
        past_k_updated = jax.lax.dynamic_update_slice(
            past_k, k_new.astype(past_k.dtype), (0, cache_position, 0, 0)
        )
        past_v_updated = jax.lax.dynamic_update_slice(
            past_v, v_new.astype(past_v.dtype), (0, cache_position, 0, 0)
        )
        q = q.astype(past_k.dtype)
        output = self._attention(q, past_k_updated, past_v_updated, attention_mask)
        return output, past_k_updated, past_v_updated


class TransformerBlock(nnx.Module):
    def __init__(self, spec: ModelSpec, layer_idx: int, *, use_rotary: bool, rngs: nnx.Rngs):
        self.attn_norm = _make_norm(spec, rngs)
        self.self_attn = SelfAttention(spec, use_rotary=use_rotary, rngs=rngs)
        self.mlp_norm = _make_norm(spec, rngs)
        self.mlp = (
            MoeMLP(spec, rngs=rngs)
            if layer_uses_moe(spec, layer_idx)
            else DenseMLP(spec, spec.intermediate_size, gated=not spec.is_encoder, rngs=rngs)
        )

    def __call__(self, hidden_states, *, attention_mask=None, position_ids=None):
        hidden_states = hidden_states + self.self_attn(
            self.attn_norm(hidden_states), attention_mask=attention_mask, position_ids=position_ids,
        )
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states

    def call_cached(
        self,
        hidden_states,
        past_k,
        past_v,
        cache_position,
        *,
        attention_mask,
        position_ids,
    ):
        attn_out, new_past_k, new_past_v = self.self_attn.call_cached(
            self.attn_norm(hidden_states),
            past_k,
            past_v,
            cache_position,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states, new_past_k, new_past_v


class GenericDecoderLM(nnx.Module):
    def __init__(self, spec: ModelSpec, *, dtype_name: str, rngs: nnx.Rngs):
        init = normal_init(spec.initializer_range)
        compute_dtype = get_dtype(dtype_name)
        self.spec = spec
        self.dtype_name = dtype_name
        self.embed_tokens = nnx.Embed(spec.vocab_size, spec.hidden_size, dtype=compute_dtype, embedding_init=init, rngs=rngs)
        self.layers = _nnx_list([TransformerBlock(spec, layer_idx, use_rotary=True, rngs=rngs) for layer_idx in range(spec.num_hidden_layers)])
        self.norm = _make_norm(spec, rngs)
        if not spec.tie_word_embeddings:
            self.lm_head = nnx.Linear(spec.hidden_size, spec.vocab_size, use_bias=False, kernel_init=init, rngs=rngs)
        else:
            self.lm_head = None

    def hidden_for_heads(self, input_ids=None, *, inputs_embeds=None, attention_mask=None, position_ids=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Pass either input_ids or inputs_embeds, not both.")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required.")
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        if position_ids is None:
            position_ids = jnp.broadcast_to(
                jnp.arange(hidden_states.shape[1])[None, :],
                (hidden_states.shape[0], hidden_states.shape[1]),
            )
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        return self.norm(hidden_states)

    def _logits(self, hidden_states):
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        return jnp.einsum("bld,vd->blv", hidden_states, self.embed_tokens.embedding[...])

    def __call__(self, input_ids=None, *, inputs_embeds=None, attention_mask=None, position_ids=None):
        hidden_states = self.hidden_for_heads(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return {"hidden_states": hidden_states, "logits": self._logits(hidden_states)}

    def call_cached(
        self,
        *,
        inputs_embeds,
        past_key_values,
        cache_position,
        attention_mask,
        position_ids,
    ):
        """KV-cached forward.

        ``past_key_values`` is a list of ``(past_k, past_v)`` per layer, each of
        shape ``[B, max_seq, num_kv_heads, head_dim]``. The current input
        ``inputs_embeds`` has ``[B, query_len, H]`` and its K/V is written into
        the cache at ``[cache_position, cache_position + query_len)``. Returns
        ``{"logits", "past_key_values"}`` where logits are for the current
        query positions only.
        """

        hidden_states = inputs_embeds
        new_past_key_values = []
        for layer_idx, layer in enumerate(self.layers):
            past_k, past_v = past_key_values[layer_idx]
            hidden_states, new_k, new_v = layer.call_cached(
                hidden_states,
                past_k,
                past_v,
                cache_position,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            new_past_key_values.append((new_k, new_v))
        hidden_states = self.norm(hidden_states)
        logits = self._logits(hidden_states)
        return {
            "hidden_states": hidden_states,
            "logits": logits,
            "past_key_values": new_past_key_values,
        }


class GenericEncoderLM(nnx.Module):
    def __init__(self, spec: ModelSpec, *, dtype_name: str, rngs: nnx.Rngs):
        init = normal_init(spec.initializer_range)
        compute_dtype = get_dtype(dtype_name)
        self.spec = spec
        self.dtype_name = dtype_name
        self.embed_tokens = nnx.Embed(spec.vocab_size, spec.hidden_size, dtype=compute_dtype, embedding_init=init, rngs=rngs)
        self.position_embeddings = nnx.Embed(spec.max_position_embeddings, spec.hidden_size, dtype=compute_dtype, embedding_init=init, rngs=rngs)
        self.embed_norm = _make_norm(spec, rngs)
        self.layers = _nnx_list([TransformerBlock(spec, layer_idx, use_rotary=False, rngs=rngs) for layer_idx in range(spec.num_hidden_layers)])
        self.norm = _make_norm(spec, rngs)
        self.mlm_dense = nnx.Linear(spec.hidden_size, spec.hidden_size, use_bias=True, kernel_init=init, bias_init=jax.nn.initializers.zeros, rngs=rngs)
        self.mlm_norm = _make_norm(spec, rngs)
        self.mlm_act = activation_fn(spec.hidden_act)
        if not spec.tie_word_embeddings:
            self.lm_head = nnx.Linear(spec.hidden_size, spec.vocab_size, use_bias=False, kernel_init=init, rngs=rngs)
        else:
            self.lm_head = None

    def backbone_hidden(self, input_ids=None, *, inputs_embeds=None, attention_mask=None, position_ids=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Pass either input_ids or inputs_embeds, not both.")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds is required.")
            batch_size, seq_len = input_ids.shape
        else:
            batch_size, seq_len = inputs_embeds.shape[:2]
        if position_ids is None:
            position_ids = jnp.broadcast_to(jnp.arange(seq_len)[None, :], (batch_size, seq_len))
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        hidden_states = hidden_states + self.position_embeddings(position_ids)
        hidden_states = self.embed_norm(hidden_states)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids)
        return self.norm(hidden_states)

    def hidden_for_heads(self, input_ids=None, *, inputs_embeds=None, attention_mask=None, position_ids=None):
        hidden_states = self.backbone_hidden(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return self.mlm_norm(self.mlm_act(self.mlm_dense(hidden_states)))

    def _logits(self, hidden_states):
        if self.lm_head is not None:
            return self.lm_head(hidden_states)
        return jnp.einsum("bld,vd->blv", hidden_states, self.embed_tokens.embedding[...])

    def __call__(self, input_ids=None, *, inputs_embeds=None, attention_mask=None, position_ids=None):
        hidden_states = self.hidden_for_heads(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return {"hidden_states": hidden_states, "logits": self._logits(hidden_states)}


class EditFlowModel(nnx.Module):
    def __init__(self, spec: ModelSpec, *, dtype_name: str, rngs: nnx.Rngs):
        init = normal_init(spec.initializer_range)
        if spec.is_encoder:
            self.backbone = GenericEncoderLM(spec, dtype_name=dtype_name, rngs=rngs)
        else:
            self.backbone = GenericDecoderLM(spec, dtype_name=dtype_name, rngs=rngs)
        self.sub_logits = nnx.Linear(spec.hidden_size, spec.vocab_size, use_bias=False, kernel_init=init, rngs=rngs)
        self.ins_logits = nnx.Linear(spec.hidden_size, spec.vocab_size, use_bias=False, kernel_init=init, rngs=rngs)
        self.rate_heads = nnx.Linear(spec.hidden_size, 3, use_bias=True, kernel_init=init, bias_init=jax.nn.initializers.zeros, rngs=rngs)

    def __call__(self, input_ids, *, attention_mask=None, position_ids=None, t=None):
        hidden_states = self.backbone.hidden_for_heads(input_ids, attention_mask=attention_mask, position_ids=position_ids)
        rates = jax.nn.softplus(self.rate_heads(hidden_states))
        sub_rate_hat, del_rate_hat, ins_rate_hat = jnp.split(rates, 3, axis=-1)
        return {
            "hidden_states": hidden_states,
            "sub_logits": self.sub_logits(hidden_states),
            "ins_logits": self.ins_logits(hidden_states),
            "sub_rate_hat": sub_rate_hat[..., 0],
            "del_rate_hat": del_rate_hat[..., 0],
            "ins_rate_hat": ins_rate_hat[..., 0],
        }


def _build_model(spec: ModelSpec, *, task: str, dtype_name: str, seed: int):
    rngs = nnx.Rngs(params=seed, dropout=seed + 1)
    if task == "editflow":
        return EditFlowModel(spec, dtype_name=dtype_name, rngs=rngs)
    if spec.is_encoder:
        return GenericEncoderLM(spec, dtype_name=dtype_name, rngs=rngs)
    return GenericDecoderLM(spec, dtype_name=dtype_name, rngs=rngs)


def build_model_from_config(
    config: transformers.PretrainedConfig,
    *,
    task: str,
    dtype_name: str = "bfloat16",
    seed: int = 0,
):
    spec = model_spec_from_config(config, task=task)
    return _build_model(spec, task=task, dtype_name=dtype_name, seed=seed)


def build_model_from_pretrained(
    model_name_or_path: str,
    *,
    task: str,
    dtype_name: str = "bfloat16",
    seed: int = 0,
    load_weights: bool = True,
):
    """Build model and optionally load pretrained weights via safetensors.

    No PyTorch required. Uses transformers.AutoConfig for config parsing
    and safetensors + numpy for weight loading.
    """
    config = transformers.AutoConfig.from_pretrained(model_name_or_path)
    model = build_model_from_config(config, task=task, dtype_name=dtype_name, seed=seed)
    if load_weights:
        from dllm_jax.weights import load_pretrained_weights
        load_pretrained_weights(model, model_name_or_path)
    return model, config
