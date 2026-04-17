"""Train DMax OPUT with the dllm_jax trainer."""

from __future__ import annotations

import argparse

import transformers
from datasets import load_dataset

from dllm_jax import (
    DMaxConfig,
    DMaxDataCollator,
    DMaxTrainer,
    build_model_from_pretrained,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--data-files", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--messages-field", default="messages")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--flag-field", default="flag")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--output-dir", default="./out-dmax")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--noise-low", type=float, default=0.75)
    parser.add_argument("--noise-high", type=float, default=0.75)
    parser.add_argument("--on-policy-ratio", type=float, default=0.5)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--no-load-weights", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def make_preprocessor(tokenizer, args):
    def preprocess(example):
        prompt_len = 0
        add_special_tokens = False
        if args.messages_field in example and example[args.messages_field] is not None:
            if not hasattr(tokenizer, "apply_chat_template"):
                raise ValueError("Dataset has messages but tokenizer has no chat template.")
            messages = example[args.messages_field]
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            if len(messages) > 1:
                prompt_text = tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_len = len(
                    tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                )
        elif args.question_field in example and args.answer_field in example:
            question = example[args.question_field]
            answer = example[args.answer_field]
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False)
                prompt_text = tokenizer.apply_chat_template(
                    messages[:1],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompt_len = len(
                    tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                )
            else:
                text = f"{question}\n{answer}"
                prompt_len = len(
                    tokenizer(f"{question}\n", add_special_tokens=True)["input_ids"]
                )
                add_special_tokens = True
        else:
            text = example[args.text_field]
            add_special_tokens = True

        encoded = tokenizer(
            text,
            max_length=args.max_length,
            truncation=True,
            add_special_tokens=add_special_tokens,
        )
        labels = list(encoded["input_ids"])
        if prompt_len:
            labels[: min(prompt_len, len(labels))] = [-100] * min(prompt_len, len(labels))
        encoded["labels"] = labels
        if args.flag_field in example:
            encoded["flag"] = bool(example[args.flag_field])
        return encoded

    return preprocess


def main():
    args = parse_args()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model, config = build_model_from_pretrained(
        args.model,
        task="llada",
        dtype_name=args.dtype,
        load_weights=not args.no_load_weights,
    )
    dataset = load_dataset(
        args.dataset,
        args.dataset_config,
        data_files=args.data_files,
        split=args.split,
    )
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    dataset = dataset.map(
        make_preprocessor(tokenizer, args),
        remove_columns=dataset.column_names,
        desc="Tokenizing DMax data",
    )

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
    trainer.train()


if __name__ == "__main__":
    main()
