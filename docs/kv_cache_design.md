# KV Cache Design for DMax SPD Inference

This doc captures the design for adding KV-cache support to `dmax_generate_spd_fast`.
The current fast path runs a full-sequence forward at every denoising step, which
is the biggest remaining performance gap vs the reference
`czg1225/DMax` implementation (which uses prefix KV cache via the `cache='prefix'`
flag in its eval scripts).

## Motivation

Per-step compute today (fast path, `kv_cache=None` equivalent):

- Attention: O(total_length^2) per layer. For `total_length=160, block_length=32`
  that's 25600 attention ops per layer per step; 160 steps × 36 layers = 147M ops
  just for Q·K^T at a 128-token generation.
- K/V projections: over the full sequence each step. Block + all future-mask
  positions are re-computed even though block-causal masking means they do not
  affect the active block's logits.

With KV cache, within a block's step loop:

- K/V projections only for the active block's `block_length` positions.
- Attention: O(block_length × block_end) ≈ O(block_length × total_length / 2)
  averaged across blocks. For the example above the last block drops from 25600
  to 5120 per layer — roughly 5× end-to-end.

## Reference's scheme

In `dInfer/python/dinfer/decoding/generate_uniform.py`:

```python
if kv_cache is None:
    # Full-sequence forward over [0, block_end]. Equivalent to current dmax.py.
else:
    # Block-only forward:
    output = model(
        inputs_embeds=embeddings,        # [B, block_length, H]
        past_key_values=past_key_values, # per-layer cached K/V at [0, block_start)
        replace_position=(start, end),   # where to write new K/V in the cache
        attention_mask=...,
        use_cache=True,
    )
```

And in `DecodeRunnerUniform.decode_uniform`:

```python
if self.need_cross_block_update:
    # First step of each new block: pass TWO blocks [prev_block, cur_block]
    # so the previous committed block's hard K/V gets written into the cache.
    cross_block_loc = BlockLoc(block_loc.start - block_length, block_loc.end)
    output, Breakflag, embeddings = self.diff_iteration.forward_uniform(
        ..., cross_block_x, cross_block_loc, ..., is_cross_block=True, ...
    )
    kv_cache.range_update(..., block_loc.start, block_length)
    past_key_values, replace_position = kv_cache.get_key_values(
        block_loc.start, block_loc.end
    )
    self.need_cross_block_update = False
else:
    # Subsequent steps within the same block: only block_length input.
    output, Breakflag, embeddings = self.diff_iteration.forward_uniform(
        ..., block, block_loc, ..., is_cross_block=False, ...
    )
```

Key invariants:

- Prior blocks' K/V in the cache are *hard* K/V (based on committed tokens).
- The active block's K/V is *soft* (based on the current step's soft-mix input)
  and is overwritten each step inside `replace_position=(block_start, block_end)`.
- The cross-block update runs a 2-block forward on entering a new block so the
  previously-committed block's hard K/V is written into the cache; this is the
  only forward that sees the previous block's final hard embeddings.

## Proposed JAX implementation

### Cache data structure

One pair of buffers per attention layer:

```python
past_k: [B, max_seq, num_kv_heads, head_dim]   # bf16
past_v: [B, max_seq, num_kv_heads, head_dim]   # bf16
```

Stacked across layers:

```python
cache = {
  "k": jnp.zeros([num_layers, B, max_seq, num_kv_heads, head_dim], dtype=bf16),
  "v": jnp.zeros([num_layers, B, max_seq, num_kv_heads, head_dim], dtype=bf16),
}
```

Size budget for Qwen3-8B-ish (`num_layers=36, num_kv_heads=8, head_dim=128`) at
`max_seq=2048, B=1, bf16`: `36 × 1 × 2048 × 8 × 128 × 2 × 2` = 300 MB per model
replica. Fine on TPU v4 HBM.

### Model changes

`SelfAttention.__call__` gains three optional kwargs:

- `past_k`, `past_v` — cache buffers for THIS layer, shape `[B, max_seq, num_kv_heads, head_dim]`.
- `cache_position` — traced scalar int32; where to write the current step's K/V.

When `past_k is None` the method is unchanged. When provided, the method:

1. Computes Q/K/V for the current `query_len` input (block_length).
2. Applies RoPE using `position_ids` (which are absolute in the total sequence).
3. Writes the new K/V into `past_k`/`past_v` at positions `[cache_position, cache_position + query_len)` via `dynamic_update_slice`.
4. Uses the UPDATED `past_k`/`past_v` as full K/V for attention.
5. Returns `(output, new_past_k, new_past_v)`.

`TransformerBlock.__call__` and `GenericDecoderLM.__call__` plumb the kwargs
through. The LM returns `(logits, new_cache)` where `new_cache` is the updated
stacked K/V.

### Generation function

New function `dmax_generate_spd_kv_fast`:

```python
@nnx.jit
def generate_fixed_shape_kv(current_model, prompt_ids):
    x = full_with_prompt_and_masks(prompt_ids)
    cache_k = zeros([num_layers, B, total_length, num_kv_heads, head_dim])
    cache_v = zeros([num_layers, B, total_length, num_kv_heads, head_dim])

    # Prefill: one full-length forward over prompt to fill cache for prompt positions.
    prompt_embeds = _embed_tokens(current_model, x[:, :prompt_length])
    _, cache_k, cache_v = current_model(
        inputs_embeds=prompt_embeds,
        attention_mask=...,                     # causal, prompt_length^2
        position_ids=arange(prompt_length),
        past_k=cache_k, past_v=cache_v,
        cache_position=0,
    )
    # Now cache [0, prompt_length) is filled.

    def block_body(block_id, carry):
        x, cache_k, cache_v = carry
        block_start = block_id * block_length
        block_end = block_start + block_length

        # Prior block's hard embeds have to be written into cache already; handled
        # via the previous block's "final hard pass" (see end of block_body).

        block_state_init = init_top_k_or_confidence()
        block_done = zeros(B, bool)

        def step_body(step_carry):
            step_idx, x, cache_k, cache_v, block_state, step_done = step_carry
            block_inputs_embeds = build_mixed_embeds(...)
            # Block-only forward with kv cache.
            logits, cache_k_new, cache_v_new = current_model(
                inputs_embeds=block_inputs_embeds,
                attention_mask=q_block_mask_over_full_kv,  # [block_length, total_length]
                position_ids=arange(block_start, block_end),
                past_k=cache_k, past_v=cache_v,
                cache_position=block_start,
            )
            # logits shape: [B, block_length, V]
            ...decoding + update x...
            return (step_idx+1, x_new, cache_k_new, cache_v_new, block_state_new, step_done_new)

        def step_cond(step_carry):
            step_idx, _, _, _, _, step_done = step_carry
            return (step_idx < denoising_steps_per_block) & (~jnp.all(step_done))

        _, x, cache_k, cache_v, _, _ = jax.lax.while_loop(
            step_cond, step_body,
            (0, x, cache_k, cache_v, block_state_init, block_done),
        )

        # Post-block hard write: one more forward with HARD embeds of decoded
        # tokens to commit the block's final hard K/V. This is the JAX-friendly
        # analogue of reference's cross-block update.
        final_block = x[:, block_start:block_end]
        final_hard_embeds = _embed_tokens(current_model, final_block)
        _, cache_k, cache_v = current_model(
            inputs_embeds=final_hard_embeds,
            attention_mask=q_block_mask_over_full_kv,
            position_ids=arange(block_start, block_end),
            past_k=cache_k, past_v=cache_v,
            cache_position=block_start,
        )

        return x, cache_k, cache_v

    x, cache_k, cache_v = jax.lax.while_loop(
        while_cond, block_body,
        (x, cache_k, cache_v),
    )
    return x[:, : prompt_length + gen_length]
```

### Attention mask

`q_block_mask_over_full_kv`: `[block_length, total_length]` boolean. For block K:
- Positions `[0, block_end)` are True (block-causal allows block K to attend there).
- Positions `[block_end, total_length)` are False (future blocks).

Built from `create_block_causal_attention_mask(total_length, block_length)[block_start:block_end, :]`.

### Correctness testing

Compare `dmax_generate_spd_kv_fast` against `dmax_generate_spd_fast` token-by-token
on the same checkpoint and settings. They must produce identical output text
and identical `updated_block` at every step (modulo float32 rounding in attention).

## Complexity and risk

- **Model forward refactor**: touches training and eval paths. Must preserve the
  `past_k=None` behavior bit-exact to avoid regressions.
- **RoPE with block-local position_ids**: positions are absolute; RoPE applied
  per query/key position; cached K has RoPE baked in. Standard HF-style caching.
- **Static cache shape**: `max_seq` is a Python int; compile once per gen_length.
- **Correctness verification**: need a direct A/B test on the Qwen3 checkpoint.

## Rollout plan

1. Add `past_k/past_v/cache_position` plumbing to model with no-op default.
2. Implement `dmax_generate_spd_kv_fast` + post-block hard-write pass.
3. Add `INFER_IMPL=kv_fast` to `tpu_dmax_infer_checkpoint.py`.
4. A/B test: `kv_fast` vs `fast` on the same prompt/settings. Compare text and `nfe`.
5. Benchmark: measure `generate_seconds` at `gen_length=128, 1024, 4096`. Expect
   3–5× speedup on longer contexts.
