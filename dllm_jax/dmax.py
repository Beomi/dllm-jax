"""DMax training and generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from dllm_jax.trainers import resolve_mask_token_id


@dataclass
class DMaxGenerationConfig:
    gen_length: int = 2048
    block_length: int = 32
    steps: int = 32
    minimal_topk: int = 1
    threshold: float = 0.95
    confidence_stop: float = 0.9
    eos_token_id: int | None = None
    mask_token_id: int | None = None
    include_eos: bool = True
    suppress_mask_token: bool = False
    temperature: float = 0.0
    top_k: int = 1


@dataclass
class DMaxGenerationOutput:
    nfe: int
    generated_tokens: jnp.ndarray
    full_tokens: jnp.ndarray


def create_block_causal_attention_mask(seq_len: int, block_size: int) -> jnp.ndarray:
    """Collapsed OPUT inference mask used by DMax generate_spd.

    DMax trains with a two-stream [noised, clean] OPUT layout. At inference,
    already committed previous blocks play the role of the clean stream, while
    the active block is the noised stream. This lower-triangular block mask is
    the single-stream equivalent used by the upstream DMax HF implementation.
    """

    idx = jnp.arange(seq_len)
    block_idx = idx // block_size
    return block_idx[:, None] >= block_idx[None, :]


def _model_vocab_size(model: Any) -> int | None:
    spec = getattr(model, "spec", None)
    if spec is not None and getattr(spec, "vocab_size", None) is not None:
        return int(spec.vocab_size)
    embed_tokens = getattr(model, "embed_tokens", None)
    embedding = getattr(embed_tokens, "embedding", None)
    if embedding is not None and getattr(embedding, "shape", None) is not None:
        return int(embedding.shape[0])
    return None


def resolve_dmax_mask_token_id(model: Any, tokenizer: Any | None = None, mask_token_id: int | None = None) -> int:
    if mask_token_id is not None:
        return int(mask_token_id)
    if tokenizer is not None:
        return resolve_mask_token_id(tokenizer, vocab_size=_model_vocab_size(model))
    vocab_size = _model_vocab_size(model)
    if vocab_size is None:
        raise ValueError("mask_token_id is required when the model vocabulary size cannot be inferred.")
    return vocab_size - 1


def _embed_tokens(model: Any, input_ids: jnp.ndarray) -> jnp.ndarray:
    if hasattr(model, "embed_tokens"):
        return model.embed_tokens(input_ids)
    if hasattr(model, "backbone") and hasattr(model.backbone, "embed_tokens"):
        return model.backbone.embed_tokens(input_ids)
    raise ValueError("DMax SPD generation requires a model with an embed_tokens layer.")


def _select_leftmost_confident_masks(
    mask_index: jnp.ndarray,
    confidence: jnp.ndarray,
    threshold: float,
) -> jnp.ndarray:
    is_low_conf = mask_index & (confidence < threshold)
    has_failed = jnp.cumsum(is_low_conf.astype(jnp.int32), axis=1) > 0
    candidates = mask_index & (~has_failed)
    has_selection = jnp.any(candidates, axis=1, keepdims=True)
    first_mask = (jnp.cumsum(mask_index.astype(jnp.int32), axis=1) == 1) & mask_index
    return jnp.where(has_selection, candidates, first_mask)


def _trim_generated_tokens(
    tokens: jnp.ndarray,
    *,
    prompt_length: int,
    gen_length: int,
    eos_token_id: int | None,
    include_eos: bool,
) -> jnp.ndarray:
    generated = tokens[:, prompt_length : prompt_length + gen_length]
    if eos_token_id is None or generated.shape[0] != 1:
        return generated
    eos_positions = jnp.nonzero(generated[0] == eos_token_id, size=generated.shape[1], fill_value=-1)[0]
    eos_positions = eos_positions[eos_positions >= 0]
    if eos_positions.shape[0] == 0:
        return generated
    end = int(eos_positions[0]) + (1 if include_eos else 0)
    return generated[:, :end]


def _sample_x0(
    active_logits: jnp.ndarray,
    active_probs: jnp.ndarray,
    max_indices: jnp.ndarray,
    max_probs: jnp.ndarray,
    temperature: float,
    rng_key: jax.Array | None,
    dtype,
):
    """Return (x0, chosen_probs) matching reference get_transfer_index_uniform.

    For ``temperature == 0`` (or no rng_key) this returns ``(max_indices, max_probs)``
    so the behavior is identical to the original greedy implementation.
    For ``temperature > 0`` it draws a gumbel sample and reports the sampled
    token's softmax probability for threshold selection.
    """

    if temperature <= 0.0 or rng_key is None:
        return max_indices.astype(dtype), max_probs
    logits_f32 = active_logits.astype(jnp.float32)
    noise = jax.random.gumbel(rng_key, logits_f32.shape, dtype=jnp.float32)
    noisy = logits_f32 / jnp.asarray(temperature, dtype=jnp.float32) + noise
    x0 = jnp.argmax(noisy, axis=-1).astype(dtype)
    chosen_probs = jnp.take_along_axis(
        active_probs, x0[..., None].astype(jnp.int32), axis=-1
    ).squeeze(-1)
    return x0, chosen_probs


def _topk_mixed_embeds(
    model: Any,
    step_topk_probs: jnp.ndarray,
    step_topk_indices: jnp.ndarray,
    block_token_embeds: jnp.ndarray,
    mask_embedding: jnp.ndarray,
    mask_embedding_norm: jnp.ndarray,
    mask_index: jnp.ndarray,
    token_index: jnp.ndarray,
) -> jnp.ndarray:
    """Top-k soft mix matching reference ``decode_uniform`` for ``top_k > 1``.

    ``soft = sum_k topk_probs_k * embed(topk_idx_k) + residual * mask_embed``,
    renormalized to the probability-weighted target norm. The result is then
    placed in ``block_inputs_embeds`` at ``token_index`` positions; mask
    positions use ``mask_embedding`` and the rest stay as ``block_token_embeds``.
    """

    topk_embeds = _embed_tokens(model, step_topk_indices)  # [B, L, K, H]
    tp = step_topk_probs.astype(topk_embeds.dtype)
    topk_weighted = (topk_embeds * tp[..., None]).sum(axis=-2)
    residual_probs = jnp.clip(
        1.0 - step_topk_probs.sum(axis=-1, keepdims=True), min=0.0
    )
    mask_weighted = mask_embedding * residual_probs.astype(mask_embedding.dtype)
    mixed_embeds = topk_weighted + mask_weighted

    topk_norms = jnp.linalg.norm(topk_embeds.astype(jnp.float32), axis=-1)
    expected_topk_norm = (topk_norms * step_topk_probs).sum(axis=-1, keepdims=True)
    expected_mask_norm = mask_embedding_norm * residual_probs
    target_norm = expected_topk_norm + expected_mask_norm
    # Match upstream ``decode_uniform`` (parallel_strategy.py:656) which uses
    # ``target_norm / (current_norm + 1e-6)``.
    current_norm = jnp.linalg.norm(
        mixed_embeds.astype(jnp.float32), axis=-1, keepdims=True
    )
    mixed_embeds = mixed_embeds * (target_norm / (current_norm + 1e-6)).astype(
        mixed_embeds.dtype
    )

    expanded_mask_embedding = jnp.broadcast_to(mask_embedding, block_token_embeds.shape)
    block_inputs_embeds = jnp.where(
        mask_index[..., None], expanded_mask_embedding, block_token_embeds
    )
    block_inputs_embeds = jnp.where(
        token_index[..., None], mixed_embeds, block_inputs_embeds
    )
    return block_inputs_embeds


def _compute_next_topk_state(
    active_probs: jnp.ndarray,
    top_k: int,
    active_block_mask: jnp.ndarray,
    updated_block: jnp.ndarray,
    mask_token_id: int,
    prev_topk_probs: jnp.ndarray,
    prev_topk_indices: jnp.ndarray,
    step_done: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute next-step ``(topk_probs, topk_indices)`` carry.

    Zeroes out non-soft positions so the ``_topk_mixed_embeds`` result at those
    positions collapses to ``mask_embedding`` (and is discarded by the
    ``token_index`` where-guard anyway). Freezes the carry when ``step_done``.
    """

    topk_probs_new, topk_indices_new = jax.lax.top_k(active_probs, top_k)
    soft_cond = active_block_mask & (updated_block != mask_token_id)
    topk_probs_new = jnp.where(
        soft_cond[..., None], topk_probs_new, jnp.zeros_like(topk_probs_new)
    )
    topk_indices_new = jnp.where(
        soft_cond[..., None],
        topk_indices_new.astype(prev_topk_indices.dtype),
        jnp.zeros_like(prev_topk_indices),
    )
    next_topk_probs = jnp.where(
        step_done[:, None, None], prev_topk_probs, topk_probs_new
    )
    next_topk_indices = jnp.where(
        step_done[:, None, None], prev_topk_indices, topk_indices_new
    )
    return next_topk_probs, next_topk_indices


def dmax_generate_spd_fast(
    model: Any,
    input_ids: jnp.ndarray,
    *,
    tokenizer: Any | None = None,
    config: DMaxGenerationConfig | None = None,
    gen_length: int | None = None,
    block_length: int | None = None,
    steps: int | None = None,
    minimal_topk: int | None = None,
    threshold: float | None = None,
    confidence_stop: float | None = None,
    eos_token_id: int | None = None,
    mask_token_id: int | None = None,
    include_eos: bool | None = None,
    suppress_mask_token: bool | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    seed: int | jax.Array | None = None,
    bucket_length: int | None = None,
) -> DMaxGenerationOutput:
    """Generate with DMax SPD using a fixed-shape compiled TPU loop.

    This intentionally runs a fixed number of denoising steps for every block.
    Avoiding Python early-exit checks removes host synchronization from the
    inner generation loop and lets XLA compile the block loop as one program.
    """

    cfg = config if config is not None else DMaxGenerationConfig()
    gen_length = cfg.gen_length if gen_length is None else gen_length
    block_length = cfg.block_length if block_length is None else block_length
    steps = cfg.steps if steps is None else steps
    minimal_topk = cfg.minimal_topk if minimal_topk is None else minimal_topk
    threshold = cfg.threshold if threshold is None else threshold
    confidence_stop = cfg.confidence_stop if confidence_stop is None else confidence_stop
    eos_token_id = cfg.eos_token_id if eos_token_id is None else eos_token_id
    mask_token_id = cfg.mask_token_id if mask_token_id is None else mask_token_id
    include_eos = cfg.include_eos if include_eos is None else include_eos
    suppress_mask_token = (
        cfg.suppress_mask_token if suppress_mask_token is None else suppress_mask_token
    )
    temperature = cfg.temperature if temperature is None else temperature
    top_k = cfg.top_k if top_k is None else top_k

    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]
    mask_token_id = resolve_dmax_mask_token_id(model, tokenizer=tokenizer, mask_token_id=mask_token_id)
    if eos_token_id is None and tokenizer is not None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    steps = min(int(steps), max(1, int(gen_length) // max(1, int(minimal_topk))))
    block_length = int(block_length)
    gen_length = int(gen_length)
    batch_size, prompt_length = input_ids.shape
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length
    new_gen_length = total_length - prompt_length
    prefill_blocks = prompt_length // block_length
    denoising_steps_per_block = min(steps, block_length)
    max_nfe = denoising_steps_per_block * max(0, num_blocks - prefill_blocks)
    threshold = float(threshold)
    confidence_stop = float(confidence_stop)
    suppress_mask_token = bool(suppress_mask_token)
    temperature = float(temperature)
    top_k = max(1, int(top_k))

    if seed is None:
        rng_base = None
    elif isinstance(seed, jax.Array):
        rng_base = seed
    else:
        rng_base = jax.random.PRNGKey(int(seed))

    def trim_output(full_tokens, nfe_value):
        generated_tokens = full_tokens[:, prompt_length : prompt_length + gen_length]
        if eos_token_id is not None:
            generated_tokens = _trim_generated_tokens(
                full_tokens,
                prompt_length=prompt_length,
                gen_length=gen_length,
                eos_token_id=eos_token_id,
                include_eos=bool(include_eos),
            )
        return DMaxGenerationOutput(
            nfe=int(nfe_value), generated_tokens=generated_tokens, full_tokens=full_tokens
        )

    bucket_length = total_length if bucket_length is None else int(bucket_length)
    bucket_length = max(block_length, bucket_length - (bucket_length % block_length))

    if bucket_length < total_length and total_length > 2 * bucket_length:
        x = jnp.full(
            (batch_size, prompt_length + new_gen_length),
            mask_token_id,
            dtype=input_ids.dtype,
        )
        x = x.at[:, :prompt_length].set(input_ids)
        if isinstance(input_ids, jax.Array):
            x = jax.device_put(x, input_ids.sharding)
        bucket_blocks = max(1, bucket_length // block_length)

        def make_generate_bucket(bucket_start_block: int, bucket_end_block: int, window_end: int):
            attention_mask = create_block_causal_attention_mask(window_end, block_length)
            position_ids = jnp.broadcast_to(jnp.arange(window_end)[None, :], (batch_size, window_end))

            @nnx.jit
            def generate_bucket(current_model, full_x):
                x_window = full_x[:, :window_end]
                token_embeds = _embed_tokens(current_model, x_window)
                mask_embedding = _embed_tokens(
                    current_model,
                    jnp.asarray([[mask_token_id]], dtype=full_x.dtype),
                )[:, :1, :]
                mask_embedding_norm = jnp.linalg.norm(
                    mask_embedding.astype(jnp.float32),
                    axis=-1,
                    keepdims=True,
                )
                bucket_nfe = jnp.asarray(0, dtype=jnp.int32)

                for block_id in range(bucket_start_block, bucket_end_block):
                    block_start = block_id * block_length
                    block_positions = block_start + jnp.arange(block_length)[None, :]
                    active_block_mask = block_positions >= prompt_length
                    if top_k > 1:
                        block_state_init = (
                            jnp.zeros((batch_size, block_length, top_k), dtype=jnp.float32),
                            jnp.zeros((batch_size, block_length, top_k), dtype=jnp.int32),
                        )
                    else:
                        block_state_init = jnp.zeros((batch_size, block_length), dtype=jnp.float32)
                    block_done = jnp.zeros((batch_size,), dtype=bool)

                    def step_body(step_carry):
                        step_index, step_x, step_token_embeds, block_state, step_done = step_carry
                        current_block = step_x[:, block_start : block_start + block_length]
                        block_token_embeds = step_token_embeds[:, block_start : block_start + block_length, :]
                        mask_index = current_block == mask_token_id
                        token_index = active_block_mask & (~mask_index)

                        if top_k > 1:
                            step_topk_probs, step_topk_indices = block_state
                            block_inputs_embeds = _topk_mixed_embeds(
                                current_model,
                                step_topk_probs,
                                step_topk_indices,
                                block_token_embeds,
                                mask_embedding,
                                mask_embedding_norm,
                                mask_index,
                                token_index,
                            )
                        else:
                            step_confidence = block_state
                            token_weight = step_confidence.astype(block_token_embeds.dtype)[..., None]
                            expanded_mask_embedding = jnp.broadcast_to(
                                mask_embedding, block_token_embeds.shape
                            )
                            expanded_mask_norm = jnp.broadcast_to(
                                mask_embedding_norm, token_weight.shape
                            )
                            mixed_embeds = token_weight * block_token_embeds + (
                                1.0 - token_weight
                            ) * expanded_mask_embedding
                            token_norm = jnp.linalg.norm(
                                block_token_embeds.astype(jnp.float32),
                                axis=-1,
                                keepdims=True,
                            )
                            target_norm = token_weight.astype(jnp.float32) * token_norm + (
                                1.0 - token_weight.astype(jnp.float32)
                            ) * expanded_mask_norm
                            current_norm = jnp.linalg.norm(
                                mixed_embeds.astype(jnp.float32), axis=-1, keepdims=True
                            )
                            mixed_embeds = mixed_embeds * (
                                target_norm / (current_norm + 1e-6)
                            ).astype(mixed_embeds.dtype)
                            block_inputs_embeds = jnp.where(
                                mask_index[..., None],
                                expanded_mask_embedding,
                                block_token_embeds,
                            )
                            block_inputs_embeds = jnp.where(
                                token_index[..., None],
                                mixed_embeds,
                                block_inputs_embeds,
                            )
                        inputs_embeds = step_token_embeds.at[
                            :, block_start : block_start + block_length, :
                        ].set(block_inputs_embeds)
                        outputs = current_model(
                            inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                        )
                        active_logits = outputs["logits"][
                            :, block_start : block_start + block_length, :
                        ]
                        if suppress_mask_token:
                            active_logits = active_logits.at[..., mask_token_id].set(
                                jnp.asarray(-1e9, dtype=active_logits.dtype)
                            )
                        active_probs = jax.nn.softmax(active_logits.astype(jnp.float32), axis=-1)
                        top1_confidence = jnp.max(active_probs, axis=-1)
                        top1_tokens = jnp.argmax(active_probs, axis=-1).astype(step_x.dtype)

                        step_rng = (
                            jax.random.fold_in(rng_base, step_index)
                            if rng_base is not None and temperature > 0.0
                            else None
                        )
                        x0, chosen_probs = _sample_x0(
                            active_logits, active_probs, top1_tokens, top1_confidence,
                            temperature, step_rng, step_x.dtype,
                        )

                        target_block = jnp.where(token_index, x0, current_block)
                        decode_mask = _select_leftmost_confident_masks(
                            mask_index,
                            chosen_probs,
                            threshold,
                        )
                        target_block = jnp.where(decode_mask, x0, target_block)
                        updated_block = jnp.where(active_block_mask, target_block, current_block)
                        same_as_previous = jnp.all(updated_block == current_block, axis=1)
                        active_confidence = jnp.where(
                            active_block_mask,
                            top1_confidence,
                            jnp.ones_like(top1_confidence),
                        )
                        all_confident = jnp.all(active_confidence >= confidence_stop, axis=1)
                        next_done = step_done | same_as_previous | all_confident
                        updated_block = jnp.where(step_done[:, None], current_block, updated_block)
                        next_x = step_x.at[:, block_start : block_start + block_length].set(updated_block)
                        updated_block_embeds = _embed_tokens(current_model, updated_block)
                        next_token_embeds = step_token_embeds.at[
                            :, block_start : block_start + block_length, :
                        ].set(updated_block_embeds)
                        if top_k > 1:
                            next_block_state = _compute_next_topk_state(
                                active_probs,
                                top_k,
                                active_block_mask,
                                updated_block,
                                mask_token_id,
                                step_topk_probs,
                                step_topk_indices,
                                step_done,
                            )
                        else:
                            computed_confidence = jnp.where(
                                active_block_mask & (updated_block != mask_token_id),
                                top1_confidence,
                                jnp.zeros_like(top1_confidence),
                            )
                            next_block_state = jnp.where(
                                step_done[:, None],
                                step_confidence,
                                computed_confidence,
                            )
                        return (
                            step_index + 1,
                            next_x,
                            next_token_embeds,
                            next_block_state,
                            next_done,
                        )

                    def step_cond(step_carry):
                        step_index, _, _, _, step_done = step_carry
                        return (step_index < denoising_steps_per_block) & (
                            ~jnp.all(step_done)
                        )

                    init_step = (
                        jnp.asarray(0, dtype=jnp.int32),
                        x_window,
                        token_embeds,
                        block_state_init,
                        block_done,
                    )
                    final_step_index, x_window, token_embeds, _, _ = jax.lax.while_loop(
                        step_cond, step_body, init_step
                    )
                    bucket_nfe = bucket_nfe + final_step_index
                return full_x.at[:, :window_end].set(x_window), bucket_nfe

            return generate_bucket

        nfe_running = 0
        for bucket_start in range(prefill_blocks, num_blocks, bucket_blocks):
            bucket_end = min(num_blocks, bucket_start + bucket_blocks)
            window_end = bucket_end * block_length
            x, bucket_nfe = make_generate_bucket(bucket_start, bucket_end, window_end)(model, x)
            nfe_running += int(bucket_nfe)
            if eos_token_id is not None:
                eos_seen = jnp.all(
                    jnp.any(
                        x[:, prompt_length : prompt_length + gen_length] == eos_token_id,
                        axis=1,
                    )
                )
                if bool(eos_seen):
                    break

        return trim_output(x[:, : prompt_length + gen_length], nfe_running)

    attention_mask = create_block_causal_attention_mask(total_length, block_length)
    position_ids = jnp.broadcast_to(jnp.arange(total_length)[None, :], (batch_size, total_length))

    @nnx.jit
    def generate_fixed_shape(current_model, prompt_ids):
        x = jnp.full(
            (batch_size, prompt_length + new_gen_length),
            mask_token_id,
            dtype=prompt_ids.dtype,
        )
        x = x.at[:, :prompt_length].set(prompt_ids)
        token_embeds = _embed_tokens(current_model, x)
        mask_embedding = _embed_tokens(
            current_model,
            jnp.asarray([[mask_token_id]], dtype=prompt_ids.dtype),
        )[:, :1, :]
        mask_embedding_norm = jnp.linalg.norm(
            mask_embedding.astype(jnp.float32),
            axis=-1,
            keepdims=True,
        )

        def block_body(block_id, carry):
            cur_x, cur_token_embeds = carry
            block_start = block_id * block_length
            block_positions = block_start + jnp.arange(block_length)[None, :]
            active_block_mask = block_positions >= prompt_length
            if top_k > 1:
                block_state_init = (
                    jnp.zeros((batch_size, block_length, top_k), dtype=jnp.float32),
                    jnp.zeros((batch_size, block_length, top_k), dtype=jnp.int32),
                )
            else:
                block_state_init = jnp.zeros((batch_size, block_length), dtype=jnp.float32)
            block_done = jnp.zeros((batch_size,), dtype=bool)

            def step_body(step_carry):
                step_index, step_x, step_token_embeds, block_state, step_done = step_carry
                current_block = jax.lax.dynamic_slice(
                    step_x,
                    (0, block_start),
                    (batch_size, block_length),
                )
                block_token_embeds = jax.lax.dynamic_slice(
                    step_token_embeds,
                    (0, block_start, 0),
                    (batch_size, block_length, step_token_embeds.shape[-1]),
                )
                mask_index = current_block == mask_token_id
                token_index = active_block_mask & (~mask_index)

                if top_k > 1:
                    step_topk_probs, step_topk_indices = block_state
                    block_inputs_embeds = _topk_mixed_embeds(
                        current_model,
                        step_topk_probs,
                        step_topk_indices,
                        block_token_embeds,
                        mask_embedding,
                        mask_embedding_norm,
                        mask_index,
                        token_index,
                    )
                else:
                    step_confidence = block_state
                    token_weight = step_confidence.astype(block_token_embeds.dtype)[..., None]
                    expanded_mask_embedding = jnp.broadcast_to(
                        mask_embedding, block_token_embeds.shape
                    )
                    expanded_mask_norm = jnp.broadcast_to(
                        mask_embedding_norm, token_weight.shape
                    )
                    mixed_embeds = token_weight * block_token_embeds + (
                        1.0 - token_weight
                    ) * expanded_mask_embedding
                    token_norm = jnp.linalg.norm(
                        block_token_embeds.astype(jnp.float32), axis=-1, keepdims=True
                    )
                    target_norm = token_weight.astype(jnp.float32) * token_norm + (
                        1.0 - token_weight.astype(jnp.float32)
                    ) * expanded_mask_norm
                    current_norm = jnp.linalg.norm(
                        mixed_embeds.astype(jnp.float32), axis=-1, keepdims=True
                    )
                    mixed_embeds = mixed_embeds * (
                        target_norm / (current_norm + 1e-6)
                    ).astype(mixed_embeds.dtype)
                    block_inputs_embeds = jnp.where(
                        mask_index[..., None],
                        expanded_mask_embedding,
                        block_token_embeds,
                    )
                    block_inputs_embeds = jnp.where(
                        token_index[..., None],
                        mixed_embeds,
                        block_inputs_embeds,
                    )
                inputs_embeds = jax.lax.dynamic_update_slice(
                    step_token_embeds,
                    block_inputs_embeds,
                    (0, block_start, 0),
                )
                outputs = current_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                active_logits = jax.lax.dynamic_slice(
                    outputs["logits"],
                    (0, block_start, 0),
                    (batch_size, block_length, outputs["logits"].shape[-1]),
                )
                if suppress_mask_token:
                    active_logits = active_logits.at[..., mask_token_id].set(
                        jnp.asarray(-1e9, dtype=active_logits.dtype)
                    )
                active_probs = jax.nn.softmax(active_logits.astype(jnp.float32), axis=-1)
                top1_confidence = jnp.max(active_probs, axis=-1)
                top1_tokens = jnp.argmax(active_probs, axis=-1).astype(step_x.dtype)

                step_rng = (
                    jax.random.fold_in(rng_base, step_index)
                    if rng_base is not None and temperature > 0.0
                    else None
                )
                x0, chosen_probs = _sample_x0(
                    active_logits, active_probs, top1_tokens, top1_confidence,
                    temperature, step_rng, step_x.dtype,
                )

                target_block = jnp.where(token_index, x0, current_block)
                decode_mask = _select_leftmost_confident_masks(mask_index, chosen_probs, threshold)
                target_block = jnp.where(decode_mask, x0, target_block)
                updated_block = jnp.where(active_block_mask, target_block, current_block)
                same_as_previous = jnp.all(updated_block == current_block, axis=1)
                active_confidence = jnp.where(
                    active_block_mask,
                    top1_confidence,
                    jnp.ones_like(top1_confidence),
                )
                all_confident = jnp.all(active_confidence >= confidence_stop, axis=1)
                next_done = step_done | same_as_previous | all_confident
                updated_block = jnp.where(step_done[:, None], current_block, updated_block)
                next_x = jax.lax.dynamic_update_slice(step_x, updated_block, (0, block_start))
                updated_block_embeds = _embed_tokens(current_model, updated_block)
                next_token_embeds = jax.lax.dynamic_update_slice(
                    step_token_embeds,
                    updated_block_embeds,
                    (0, block_start, 0),
                )
                if top_k > 1:
                    next_block_state = _compute_next_topk_state(
                        active_probs,
                        top_k,
                        active_block_mask,
                        updated_block,
                        mask_token_id,
                        step_topk_probs,
                        step_topk_indices,
                        step_done,
                    )
                else:
                    computed_confidence = jnp.where(
                        active_block_mask & (updated_block != mask_token_id),
                        top1_confidence,
                        jnp.zeros_like(top1_confidence),
                    )
                    next_block_state = jnp.where(
                        step_done[:, None],
                        step_confidence,
                        computed_confidence,
                    )
                return step_index + 1, next_x, next_token_embeds, next_block_state, next_done

            def step_cond(step_carry):
                step_index, _, _, _, step_done = step_carry
                return (step_index < denoising_steps_per_block) & (~jnp.all(step_done))

            init_step = (
                jnp.asarray(0, dtype=jnp.int32),
                cur_x,
                cur_token_embeds,
                block_state_init,
                block_done,
            )
            final_step_index, cur_x, cur_token_embeds, _, _ = jax.lax.while_loop(
                step_cond, step_body, init_step
            )
            return cur_x, cur_token_embeds, final_step_index

        has_eos_check = eos_token_id is not None

        def while_cond(carry):
            block_id, _, _, _, eos_done = carry
            return (block_id < num_blocks) & (~eos_done)

        def while_body(carry):
            block_id, cur_x, cur_token_embeds, nfe_counter, eos_done = carry
            cur_x, cur_token_embeds, block_steps = block_body(
                block_id, (cur_x, cur_token_embeds)
            )
            nfe_counter = nfe_counter + block_steps
            if has_eos_check:
                # Match reference DecodeRunnerUniform.decode_uniform
                # (generate_uniform.py:402-404): per-batch eos_idx over the
                # just-decoded block, then under early_stop fill the rest of
                # the sequence with eos_id for that batch element.
                block_start_local = block_id * block_length
                block_end_local = block_start_local + block_length
                active_block_slice = jax.lax.dynamic_slice(
                    cur_x, (0, block_start_local), (cur_x.shape[0], block_length)
                )
                eos_seen = jnp.any(active_block_slice == eos_token_id, axis=1)
                positions = jnp.arange(cur_x.shape[1])
                post_mask = positions >= block_end_local
                fill_cond = eos_seen[:, None] & post_mask[None, :]
                cur_x = jnp.where(
                    fill_cond,
                    jnp.asarray(eos_token_id, dtype=cur_x.dtype),
                    cur_x,
                )
                eos_done_next = jnp.all(eos_seen)
            else:
                eos_done_next = eos_done
            return block_id + 1, cur_x, cur_token_embeds, nfe_counter, eos_done_next

        init_carry = (
            jnp.asarray(prefill_blocks, dtype=jnp.int32),
            x,
            token_embeds,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
        )
        _, x_final, _, nfe_actual, _ = jax.lax.while_loop(
            while_cond, while_body, init_carry
        )
        return x_final[:, : prompt_length + gen_length], nfe_actual

    x_generated, nfe_value = generate_fixed_shape(model, input_ids)
    return trim_output(x_generated, nfe_value)


def dmax_generate_spd(
    model: Any,
    input_ids: jnp.ndarray,
    *,
    tokenizer: Any | None = None,
    config: DMaxGenerationConfig | None = None,
    gen_length: int | None = None,
    block_length: int | None = None,
    steps: int | None = None,
    minimal_topk: int | None = None,
    threshold: float | None = None,
    confidence_stop: float | None = None,
    eos_token_id: int | None = None,
    mask_token_id: int | None = None,
    include_eos: bool | None = None,
    suppress_mask_token: bool | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    seed: int | jax.Array | None = None,
) -> DMaxGenerationOutput:
    """Generate with DMax soft parallel decoding.

    This mirrors DMax's block-wise SPD loop: each block starts as MASK tokens,
    confident mask prefixes are committed left-to-right, and already committed
    tokens in the active block are fed back as confidence-weighted blends of
    token and mask embeddings.
    """

    cfg = config if config is not None else DMaxGenerationConfig()
    gen_length = cfg.gen_length if gen_length is None else gen_length
    block_length = cfg.block_length if block_length is None else block_length
    steps = cfg.steps if steps is None else steps
    minimal_topk = cfg.minimal_topk if minimal_topk is None else minimal_topk
    threshold = cfg.threshold if threshold is None else threshold
    confidence_stop = cfg.confidence_stop if confidence_stop is None else confidence_stop
    eos_token_id = cfg.eos_token_id if eos_token_id is None else eos_token_id
    mask_token_id = cfg.mask_token_id if mask_token_id is None else mask_token_id
    include_eos = cfg.include_eos if include_eos is None else include_eos
    suppress_mask_token = (
        cfg.suppress_mask_token if suppress_mask_token is None else suppress_mask_token
    )
    temperature = cfg.temperature if temperature is None else temperature
    top_k = cfg.top_k if top_k is None else top_k

    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]
    mask_token_id = resolve_dmax_mask_token_id(model, tokenizer=tokenizer, mask_token_id=mask_token_id)
    if eos_token_id is None and tokenizer is not None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    steps = min(int(steps), max(1, int(gen_length) // max(1, int(minimal_topk))))
    block_length = int(block_length)
    gen_length = int(gen_length)
    batch_size, prompt_length = input_ids.shape
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length
    new_gen_length = total_length - prompt_length

    x = jnp.full((batch_size, prompt_length + new_gen_length), mask_token_id, dtype=input_ids.dtype)
    x = x.at[:, :prompt_length].set(input_ids)
    position_ids = jnp.broadcast_to(jnp.arange(total_length)[None, :], x.shape)
    attention_mask = create_block_causal_attention_mask(total_length, block_length)
    mask_embedding = _embed_tokens(model, jnp.asarray([[mask_token_id]], dtype=x.dtype))[:, :1, :]
    mask_embedding_norm = jnp.linalg.norm(mask_embedding.astype(jnp.float32), axis=-1, keepdims=True)

    prefill_blocks = prompt_length // block_length
    denoising_steps_per_block = min(steps, block_length)
    nfe = 0
    suppress_mask_token = bool(suppress_mask_token)
    temperature = float(temperature)
    top_k = max(1, int(top_k))

    if seed is None:
        rng_base = None
    elif isinstance(seed, jax.Array):
        rng_base = seed
    else:
        rng_base = jax.random.PRNGKey(int(seed))

    for block_id in range(prefill_blocks, num_blocks):
        block_start = block_id * block_length
        block_end = (block_id + 1) * block_length
        current_window_end = block_end
        cur_x = x[:, :current_window_end]
        cur_token_embeds = _embed_tokens(model, cur_x)
        if top_k > 1:
            block_topk_probs = jnp.zeros((batch_size, block_length, top_k), dtype=jnp.float32)
            block_topk_indices = jnp.zeros((batch_size, block_length, top_k), dtype=jnp.int32)
        else:
            block_confidence = jnp.zeros((batch_size, block_length), dtype=jnp.float32)
        block_positions = jnp.arange(block_start, block_end)[None, :]
        active_block_mask = block_positions >= prompt_length

        for step_idx in range(denoising_steps_per_block):
            current_block = cur_x[:, block_start:block_end]
            mask_index = current_block == mask_token_id
            token_index = active_block_mask & (~mask_index)
            block_token_embeds = cur_token_embeds[:, block_start:block_end, :]

            if top_k > 1:
                block_inputs_embeds = _topk_mixed_embeds(
                    model,
                    block_topk_probs,
                    block_topk_indices,
                    block_token_embeds,
                    mask_embedding,
                    mask_embedding_norm,
                    mask_index,
                    token_index,
                )
            else:
                token_weight = block_confidence.astype(block_token_embeds.dtype)[..., None]
                expanded_mask_embedding = jnp.broadcast_to(mask_embedding, block_token_embeds.shape)
                expanded_mask_norm = jnp.broadcast_to(mask_embedding_norm, token_weight.shape)
                mixed_embeds = token_weight * block_token_embeds + (1.0 - token_weight) * expanded_mask_embedding
                token_norm = jnp.linalg.norm(block_token_embeds.astype(jnp.float32), axis=-1, keepdims=True)
                target_norm = token_weight.astype(jnp.float32) * token_norm + (
                    1.0 - token_weight.astype(jnp.float32)
                ) * expanded_mask_norm
                current_norm = jnp.linalg.norm(
                    mixed_embeds.astype(jnp.float32), axis=-1, keepdims=True
                )
                mixed_embeds = mixed_embeds * (
                    target_norm / (current_norm + 1e-6)
                ).astype(mixed_embeds.dtype)
                block_inputs_embeds = jnp.where(mask_index[..., None], expanded_mask_embedding, block_token_embeds)
                block_inputs_embeds = jnp.where(token_index[..., None], mixed_embeds, block_inputs_embeds)

            cur_inputs_embeds = cur_token_embeds.at[:, block_start:block_end, :].set(block_inputs_embeds)
            outputs = model(
                inputs_embeds=cur_inputs_embeds,
                attention_mask=attention_mask[:current_window_end, :current_window_end],
                position_ids=position_ids[:, :current_window_end],
            )
            nfe += 1

            active_logits = outputs["logits"][:, block_start:block_end, :]
            if suppress_mask_token:
                active_logits = active_logits.at[..., mask_token_id].set(
                    jnp.asarray(-1e9, dtype=active_logits.dtype)
                )
            active_probs = jax.nn.softmax(active_logits.astype(jnp.float32), axis=-1)
            top1_confidence = jnp.max(active_probs, axis=-1)
            top1_tokens = jnp.argmax(active_probs, axis=-1).astype(x.dtype)

            step_rng = (
                jax.random.fold_in(rng_base, step_idx)
                if rng_base is not None and temperature > 0.0
                else None
            )
            x0, chosen_probs = _sample_x0(
                active_logits, active_probs, top1_tokens, top1_confidence,
                temperature, step_rng, x.dtype,
            )

            target_block = jnp.where(token_index, x0, current_block)
            decode_mask = _select_leftmost_confident_masks(mask_index, chosen_probs, float(threshold))
            target_block = jnp.where(decode_mask, x0, target_block)
            updated_block = jnp.where(active_block_mask, target_block, current_block)
            same_as_previous = bool(jnp.all(updated_block == current_block))

            cur_x = cur_x.at[:, block_start:block_end].set(updated_block)
            active_confidence = jnp.where(active_block_mask, top1_confidence, jnp.ones_like(top1_confidence))
            all_confident = bool(jnp.all(active_confidence >= float(confidence_stop)))
            if same_as_previous or all_confident:
                break

            cur_token_embeds = cur_token_embeds.at[:, block_start:block_end, :].set(
                _embed_tokens(model, updated_block)
            )
            if top_k > 1:
                topk_probs_new, topk_indices_new = jax.lax.top_k(active_probs, top_k)
                soft_cond = active_block_mask & (updated_block != mask_token_id)
                block_topk_probs = jnp.where(
                    soft_cond[..., None], topk_probs_new, jnp.zeros_like(topk_probs_new)
                )
                block_topk_indices = jnp.where(
                    soft_cond[..., None],
                    topk_indices_new.astype(block_topk_indices.dtype),
                    jnp.zeros_like(block_topk_indices),
                )
            else:
                block_confidence = jnp.where(
                    active_block_mask & (updated_block != mask_token_id),
                    top1_confidence,
                    jnp.zeros_like(top1_confidence),
                )

        x = x.at[:, :current_window_end].set(cur_x)
        if eos_token_id is not None:
            eos_seen = jnp.any(x[:, prompt_length:current_window_end] == eos_token_id, axis=1)
            if bool(jnp.all(eos_seen)):
                break

    full_tokens = x[:, : prompt_length + gen_length]
    generated_tokens = _trim_generated_tokens(
        full_tokens,
        prompt_length=prompt_length,
        gen_length=gen_length,
        eos_token_id=eos_token_id,
        include_eos=bool(include_eos),
    )
    return DMaxGenerationOutput(nfe=nfe, generated_tokens=generated_tokens, full_tokens=full_tokens)


def dmax_generate_spd_kv_fast(
    model: Any,
    input_ids: jnp.ndarray,
    *,
    tokenizer: Any | None = None,
    config: DMaxGenerationConfig | None = None,
    gen_length: int | None = None,
    block_length: int | None = None,
    steps: int | None = None,
    minimal_topk: int | None = None,
    threshold: float | None = None,
    confidence_stop: float | None = None,
    eos_token_id: int | None = None,
    mask_token_id: int | None = None,
    include_eos: bool | None = None,
    suppress_mask_token: bool | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    seed: int | jax.Array | None = None,
) -> DMaxGenerationOutput:
    """KV-cached fast SPD.

    Per-layer K/V cache so that each denoising step's forward only projects
    K/V for the current block (``block_length`` positions) rather than for the
    whole sequence. Queries cover only the current block; attention runs over
    the cache buffers which are ``[B, total_length, num_kv_heads, head_dim]``
    each. Matches the reference DMax path that sets ``cache='prefix'`` in its
    eval scripts.

    After each block's denoising loop completes, runs one more block-local
    forward with HARD embeddings of the committed tokens to overwrite the
    soft K/V in the cache with hard K/V. This is the JAX analogue of the
    reference's cross-block update.
    """

    cfg = config if config is not None else DMaxGenerationConfig()
    gen_length = cfg.gen_length if gen_length is None else gen_length
    block_length = cfg.block_length if block_length is None else block_length
    steps = cfg.steps if steps is None else steps
    minimal_topk = cfg.minimal_topk if minimal_topk is None else minimal_topk
    threshold = cfg.threshold if threshold is None else threshold
    confidence_stop = cfg.confidence_stop if confidence_stop is None else confidence_stop
    eos_token_id = cfg.eos_token_id if eos_token_id is None else eos_token_id
    mask_token_id = cfg.mask_token_id if mask_token_id is None else mask_token_id
    include_eos = cfg.include_eos if include_eos is None else include_eos
    suppress_mask_token = (
        cfg.suppress_mask_token if suppress_mask_token is None else suppress_mask_token
    )
    temperature = cfg.temperature if temperature is None else temperature
    top_k = cfg.top_k if top_k is None else top_k

    input_ids = jnp.asarray(input_ids, dtype=jnp.int32)
    if input_ids.ndim == 1:
        input_ids = input_ids[None, :]
    mask_token_id = resolve_dmax_mask_token_id(
        model, tokenizer=tokenizer, mask_token_id=mask_token_id
    )
    if eos_token_id is None and tokenizer is not None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None)

    steps = min(int(steps), max(1, int(gen_length) // max(1, int(minimal_topk))))
    block_length = int(block_length)
    gen_length = int(gen_length)
    batch_size, prompt_length = input_ids.shape
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length
    prefill_blocks_count = prompt_length // block_length
    denoising_steps_per_block = min(steps, block_length)
    threshold = float(threshold)
    confidence_stop = float(confidence_stop)
    suppress_mask_token = bool(suppress_mask_token)
    temperature = float(temperature)
    top_k = max(1, int(top_k))

    if seed is None:
        rng_base = None
    elif isinstance(seed, jax.Array):
        rng_base = seed
    else:
        rng_base = jax.random.PRNGKey(int(seed))

    spec = model.spec
    num_layers = int(spec.num_hidden_layers)
    num_kv_heads = int(spec.num_key_value_heads)
    head_dim = int(spec.hidden_size // spec.num_attention_heads)

    position_ids_full = jnp.broadcast_to(
        jnp.arange(total_length)[None, :], (batch_size, total_length)
    )
    full_attn_mask = create_block_causal_attention_mask(total_length, block_length)

    def trim_output(full_tokens, nfe_value):
        generated_tokens = full_tokens[:, prompt_length : prompt_length + gen_length]
        if eos_token_id is not None:
            generated_tokens = _trim_generated_tokens(
                full_tokens,
                prompt_length=prompt_length,
                gen_length=gen_length,
                eos_token_id=eos_token_id,
                include_eos=bool(include_eos),
            )
        return DMaxGenerationOutput(
            nfe=int(nfe_value), generated_tokens=generated_tokens, full_tokens=full_tokens
        )

    @nnx.jit
    def generate_kv(current_model, prompt_ids):
        mask_embedding = _embed_tokens(
            current_model,
            jnp.asarray([[mask_token_id]], dtype=prompt_ids.dtype),
        )[:, :1, :]
        mask_embedding_norm = jnp.linalg.norm(
            mask_embedding.astype(jnp.float32), axis=-1, keepdims=True
        )

        # K/V cache stored in float32 to match the QK-norm precision path that
        # the non-cached ``fast`` attention uses: for models with ``qk_norm``
        # (e.g. Qwen3) ``nnx.RMSNorm`` upcasts to float32, and
        # ``jax.nn.dot_product_attention`` requires Q, K, V to all share dtype.
        # V is upcast from the model's compute dtype (bf16) to float32 when
        # written, which is exact.
        cache_dtype = jnp.float32

        past_kv = [
            (
                jnp.zeros(
                    (batch_size, total_length, num_kv_heads, head_dim), dtype=cache_dtype
                ),
                jnp.zeros(
                    (batch_size, total_length, num_kv_heads, head_dim), dtype=cache_dtype
                ),
            )
            for _ in range(num_layers)
        ]

        x = jnp.full(
            (batch_size, total_length), mask_token_id, dtype=prompt_ids.dtype
        )
        x = x.at[:, :prompt_length].set(prompt_ids)

        prompt_embeds = _embed_tokens(current_model, x[:, :prompt_length])
        # Mask must be shape (prompt_length, total_length) because attention
        # runs Q (prompt) over the full cache (total_length). block-causal
        # within prompt AND gate out K positions >= prompt_length (cache is
        # zero there and would otherwise dilute attention).
        kv_positions_full = jnp.arange(total_length)
        prefill_kv_valid = kv_positions_full < prompt_length
        prefill_attn_mask = full_attn_mask[:prompt_length, :] & prefill_kv_valid[None, :]
        prefill_pos_ids = position_ids_full[:, :prompt_length]
        prefill_out = current_model.call_cached(
            inputs_embeds=prompt_embeds,
            past_key_values=past_kv,
            cache_position=jnp.asarray(0, dtype=jnp.int32),
            attention_mask=prefill_attn_mask,
            position_ids=prefill_pos_ids,
        )
        past_kv = prefill_out["past_key_values"]

        block_pos_template = jnp.arange(block_length)[None, :]

        def block_body_kv(block_id, cur_x, cur_past_kv):
            block_start = block_id * block_length
            block_end = block_start + block_length
            block_positions = block_start + jnp.arange(block_length)[None, :]
            active_block_mask = block_positions >= prompt_length
            if top_k > 1:
                block_state_init = (
                    jnp.zeros(
                        (batch_size, block_length, top_k), dtype=jnp.float32
                    ),
                    jnp.zeros(
                        (batch_size, block_length, top_k), dtype=jnp.int32
                    ),
                )
            else:
                block_state_init = jnp.zeros(
                    (batch_size, block_length), dtype=jnp.float32
                )
            block_done = jnp.zeros((batch_size,), dtype=bool)

            kv_positions = jnp.arange(total_length)
            block_kv_row = kv_positions < block_end
            attn_mask_block = jnp.broadcast_to(
                block_kv_row[None, :], (block_length, total_length)
            )
            block_pos_ids = jnp.broadcast_to(
                block_start + block_pos_template, (batch_size, block_length)
            )

            def step_body(step_carry):
                step_index, step_x, step_past_kv, block_state, step_done = step_carry
                current_block = jax.lax.dynamic_slice(
                    step_x, (0, block_start), (batch_size, block_length)
                )
                mask_index = current_block == mask_token_id
                token_index = active_block_mask & (~mask_index)
                block_token_embeds = _embed_tokens(current_model, current_block)

                if top_k > 1:
                    step_topk_probs, step_topk_indices = block_state
                    block_inputs_embeds = _topk_mixed_embeds(
                        current_model,
                        step_topk_probs,
                        step_topk_indices,
                        block_token_embeds,
                        mask_embedding,
                        mask_embedding_norm,
                        mask_index,
                        token_index,
                    )
                else:
                    step_confidence = block_state
                    token_weight = step_confidence.astype(
                        block_token_embeds.dtype
                    )[..., None]
                    expanded_mask_embedding = jnp.broadcast_to(
                        mask_embedding, block_token_embeds.shape
                    )
                    expanded_mask_norm = jnp.broadcast_to(
                        mask_embedding_norm, token_weight.shape
                    )
                    mixed_embeds = token_weight * block_token_embeds + (
                        1.0 - token_weight
                    ) * expanded_mask_embedding
                    token_norm = jnp.linalg.norm(
                        block_token_embeds.astype(jnp.float32),
                        axis=-1,
                        keepdims=True,
                    )
                    target_norm = token_weight.astype(jnp.float32) * token_norm + (
                        1.0 - token_weight.astype(jnp.float32)
                    ) * expanded_mask_norm
                    current_norm = jnp.linalg.norm(
                        mixed_embeds.astype(jnp.float32),
                        axis=-1,
                        keepdims=True,
                    )
                    mixed_embeds = mixed_embeds * (
                        target_norm / (current_norm + 1e-6)
                    ).astype(mixed_embeds.dtype)
                    block_inputs_embeds = jnp.where(
                        mask_index[..., None],
                        expanded_mask_embedding,
                        block_token_embeds,
                    )
                    block_inputs_embeds = jnp.where(
                        token_index[..., None],
                        mixed_embeds,
                        block_inputs_embeds,
                    )

                out = current_model.call_cached(
                    inputs_embeds=block_inputs_embeds,
                    past_key_values=step_past_kv,
                    cache_position=jnp.asarray(block_start, dtype=jnp.int32),
                    attention_mask=attn_mask_block,
                    position_ids=block_pos_ids,
                )
                active_logits = out["logits"]
                new_past_kv = out["past_key_values"]

                if suppress_mask_token:
                    active_logits = active_logits.at[..., mask_token_id].set(
                        jnp.asarray(-1e9, dtype=active_logits.dtype)
                    )
                active_probs = jax.nn.softmax(
                    active_logits.astype(jnp.float32), axis=-1
                )
                top1_confidence = jnp.max(active_probs, axis=-1)
                top1_tokens = jnp.argmax(active_probs, axis=-1).astype(step_x.dtype)

                step_rng = (
                    jax.random.fold_in(rng_base, step_index)
                    if rng_base is not None and temperature > 0.0
                    else None
                )
                x0, chosen_probs = _sample_x0(
                    active_logits,
                    active_probs,
                    top1_tokens,
                    top1_confidence,
                    temperature,
                    step_rng,
                    step_x.dtype,
                )

                target_block = jnp.where(token_index, x0, current_block)
                decode_mask = _select_leftmost_confident_masks(
                    mask_index, chosen_probs, threshold
                )
                target_block = jnp.where(decode_mask, x0, target_block)
                updated_block = jnp.where(
                    active_block_mask, target_block, current_block
                )
                same_as_previous = jnp.all(
                    updated_block == current_block, axis=1
                )
                active_confidence = jnp.where(
                    active_block_mask,
                    top1_confidence,
                    jnp.ones_like(top1_confidence),
                )
                all_confident = jnp.all(
                    active_confidence >= confidence_stop, axis=1
                )
                next_done = step_done | same_as_previous | all_confident
                updated_block = jnp.where(
                    step_done[:, None], current_block, updated_block
                )
                next_x = jax.lax.dynamic_update_slice(
                    step_x, updated_block, (0, block_start)
                )

                if top_k > 1:
                    next_block_state = _compute_next_topk_state(
                        active_probs,
                        top_k,
                        active_block_mask,
                        updated_block,
                        mask_token_id,
                        step_topk_probs,
                        step_topk_indices,
                        step_done,
                    )
                else:
                    computed_confidence = jnp.where(
                        active_block_mask & (updated_block != mask_token_id),
                        top1_confidence,
                        jnp.zeros_like(top1_confidence),
                    )
                    next_block_state = jnp.where(
                        step_done[:, None],
                        step_confidence,
                        computed_confidence,
                    )
                return (
                    step_index + 1,
                    next_x,
                    new_past_kv,
                    next_block_state,
                    next_done,
                )

            def step_cond(step_carry):
                step_index, _, _, _, step_done = step_carry
                return (step_index < denoising_steps_per_block) & (
                    ~jnp.all(step_done)
                )

            init_step = (
                jnp.asarray(0, dtype=jnp.int32),
                cur_x,
                cur_past_kv,
                block_state_init,
                block_done,
            )
            final_step_index, x_new, past_kv_new, _, _ = jax.lax.while_loop(
                step_cond, step_body, init_step
            )

            # Post-block hard write.
            final_block = jax.lax.dynamic_slice(
                x_new, (0, block_start), (batch_size, block_length)
            )
            hard_embeds = _embed_tokens(current_model, final_block)
            hard_out = current_model.call_cached(
                inputs_embeds=hard_embeds,
                past_key_values=past_kv_new,
                cache_position=jnp.asarray(block_start, dtype=jnp.int32),
                attention_mask=attn_mask_block,
                position_ids=block_pos_ids,
            )
            past_kv_hard = hard_out["past_key_values"]
            return x_new, past_kv_hard, final_step_index + 1

        has_eos_check = eos_token_id is not None

        def outer_cond(outer_carry):
            block_id, _, _, _, eos_done = outer_carry
            return (block_id < num_blocks) & (~eos_done)

        def outer_body(outer_carry):
            block_id, cur_x, cur_past_kv, nfe_counter, eos_done = outer_carry
            cur_x, cur_past_kv, block_steps = block_body_kv(
                block_id, cur_x, cur_past_kv
            )
            nfe_counter = nfe_counter + block_steps
            if has_eos_check:
                block_start_local = block_id * block_length
                block_end_local = block_start_local + block_length
                active_block_slice = jax.lax.dynamic_slice(
                    cur_x,
                    (0, block_start_local),
                    (cur_x.shape[0], block_length),
                )
                eos_seen = jnp.any(active_block_slice == eos_token_id, axis=1)
                positions = jnp.arange(cur_x.shape[1])
                post_mask = positions >= block_end_local
                fill_cond = eos_seen[:, None] & post_mask[None, :]
                cur_x = jnp.where(
                    fill_cond,
                    jnp.asarray(eos_token_id, dtype=cur_x.dtype),
                    cur_x,
                )
                eos_done_next = jnp.all(eos_seen)
            else:
                eos_done_next = eos_done
            return block_id + 1, cur_x, cur_past_kv, nfe_counter, eos_done_next

        init_outer = (
            jnp.asarray(prefill_blocks_count, dtype=jnp.int32),
            x,
            past_kv,
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
        )
        _, x_final, _, nfe_actual, _ = jax.lax.while_loop(
            outer_cond, outer_body, init_outer
        )
        return x_final[:, : prompt_length + gen_length], nfe_actual

    x_generated, nfe_value = generate_kv(model, input_ids)
    return trim_output(x_generated, nfe_value)


