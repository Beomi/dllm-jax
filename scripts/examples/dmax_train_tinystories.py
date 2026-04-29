"""Train a small from-scratch DMax model on TinyStories."""

from __future__ import annotations

import argparse
import os

import transformers
from datasets import load_dataset

from dllm_jax import (
    DMaxConfig,
    DMaxDataCollator,
    DMaxTrainer,
    build_model_from_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train[:1024]")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output-dir", default="/tmp/dmax-tinystories")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--noise-low", type=float, default=0.75)
    parser.add_argument("--noise-high", type=float, default=0.75)
    parser.add_argument("--on-policy-ratio", type=float, default=0.5)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def prepare_tokenizer(name: str):
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    if tokenizer.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "<|dmax_mask|>"})
    tokenizer.padding_side = "right"
    return tokenizer


def build_tiny_config(tokenizer, args):
    return transformers.LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        max_position_embeddings=args.max_length,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        attention_bias=False,
        tie_word_embeddings=True,
    )


def make_preprocessor(tokenizer, args):
    eos = tokenizer.eos_token or ""

    def preprocess(example):
        text = str(example[args.text_field])
        if eos and not text.endswith(eos):
            text = text + eos
        encoded = tokenizer(
            text,
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=False,
        )
        encoded["labels"] = list(encoded["input_ids"])
        return encoded

    return preprocess


def main():
    args = parse_args()
    tokenizer = prepare_tokenizer(args.tokenizer)
    config = build_tiny_config(tokenizer, args)
    model = build_model_from_config(config, task="llada", dtype_name=args.dtype)

    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        split=args.split,
    )
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    dataset = dataset.map(
        make_preprocessor(tokenizer, args),
        remove_columns=dataset.column_names,
        desc="Tokenizing TinyStories",
    )
    dataset = dataset.filter(lambda example: len(example["input_ids"]) > 0)

    train_args = DMaxConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        noise_range_low=args.noise_low,
        noise_range_high=args.noise_high,
        on_policy_ratio=args.on_policy_ratio,
        block_size=args.block_size,
    )
    trainer = DMaxTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=dataset,
        data_collator=DMaxDataCollator(tokenizer=tokenizer, label_pad_token_id=-100),
        config=config,
    )
    result = trainer.train()
    final_step = int(result["global_step"])
    final_checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{final_step}")
    trainer.save_model(final_checkpoint_dir)
    print(f"final_checkpoint_dir={final_checkpoint_dir}")


if __name__ == "__main__":
    main()
