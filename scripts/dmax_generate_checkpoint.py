"""Generate with DMax from a saved dllm_jax trainer checkpoint."""

from __future__ import annotations

import argparse

import jax.numpy as jnp
import transformers

from dllm_jax import (
    build_model_from_config,
    dmax_generate_spd,
    dmax_generate_spd_fast,
    dmax_generate_spd_kv_fast,
    restore_model_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--confidence-stop", type=float, default=0.9)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--mask-token-id", type=int, default=None)
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--impl", choices=["fast", "legacy", "kv_fast"], default="fast")
    parser.add_argument("--bucket-length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--suppress-mask-token", action="store_true")
    return parser.parse_args()


def encode_prompt(tokenizer, prompt: str, use_chat_template: bool):
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=None,
        )
    else:
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    return jnp.asarray([ids], dtype=jnp.int32)


def main():
    args = parse_args()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.checkpoint_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    config = transformers.AutoConfig.from_pretrained(args.checkpoint_dir)
    model = build_model_from_config(config, task="llada", dtype_name=args.dtype)
    restore_model_checkpoint(model, args.checkpoint_dir)

    input_ids = encode_prompt(tokenizer, args.prompt, args.chat_template)
    generate_fn = {
        "fast": dmax_generate_spd_fast,
        "legacy": dmax_generate_spd,
        "kv_fast": dmax_generate_spd_kv_fast,
    }[args.impl]
    kwargs = dict(
        tokenizer=tokenizer,
        gen_length=args.gen_length,
        block_length=args.block_length,
        steps=args.steps,
        threshold=args.threshold,
        confidence_stop=args.confidence_stop,
        mask_token_id=args.mask_token_id,
        eos_token_id=args.eos_token_id,
        suppress_mask_token=args.suppress_mask_token,
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    if generate_fn is dmax_generate_spd_fast and args.bucket_length is not None:
        kwargs["bucket_length"] = args.bucket_length
    output = generate_fn(model, input_ids, **kwargs)
    generated = output.generated_tokens[0].tolist()
    print(tokenizer.decode(generated, skip_special_tokens=True))
    print(f"nfe={output.nfe} generated_tokens={len(generated)}")


if __name__ == "__main__":
    main()
