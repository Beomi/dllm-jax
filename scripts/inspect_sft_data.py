"""Dump SFT data pipeline batches to stdout for inspection.

Mirrors the packing path in ``scripts/tpu_train.py`` (``refill_buffer`` +
``get_batch``) without any JAX / training code, so we can cheaply check:

- does ``apply_chat_template`` round-trip correctly (decode == human-readable)?
- does concat-packing produce the expected ``<|im_start|>...<|im_end|>`` structure?
- does labels == input_ids (full-text SFT) and is the MASK token absent from data?
- does the EOS token sit between docs, not inside them?
- are there suspicious id==-1 / id>vocab_size / repeated pathological tokens?

Run locally or on any TPU worker:
  python3 scripts/inspect_sft_data.py  [--max-len 4096] [--rows 2] [--batches 1]
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


_ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
    "user": "user",
    "assistant": "assistant",
}


def _sft_row_to_messages(row):
    convs = row.get("conversations")
    if not convs:
        return None
    messages = []
    system = row.get("system")
    if system:
        messages.append({"role": "system", "content": system})
    for c in convs:
        if isinstance(c, dict) and "from" in c:
            messages.append({"role": _ROLE_MAP.get(c["from"], c["from"]), "content": c["value"]})
        else:
            messages.append(c)
    return messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B"))
    ap.add_argument("--max-len", type=int, default=int(os.environ.get("MAX_LEN", "4096")))
    ap.add_argument("--batch", type=int, default=int(os.environ.get("GLOBAL_BATCH", "4")))
    ap.add_argument("--rows", type=int, default=2, help="how many packed rows to decode in full")
    ap.add_argument("--batches", type=int, default=1, help="how many batches to draw")
    ap.add_argument("--mask-id", type=int, default=int(os.environ.get("MASK_TOKEN_ID", "151662")))
    args = ap.parse_args()

    print(f"[inspect] model={args.model} max_len={args.max_len} batch={args.batch} mask_id={args.mask_id}")

    import transformers
    from datasets import load_dataset

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    eos_id = tokenizer.eos_token_id
    print(f"[inspect] eos_id={eos_id} pad_id={tokenizer.pad_token_id} vocab={tokenizer.vocab_size}")
    # Decode the special MASK id to see what Qwen3 calls 151662
    try:
        mask_tok = tokenizer.decode([args.mask_id])
        print(f"[inspect] mask_id {args.mask_id} -> {mask_tok!r}")
    except Exception as exc:
        print(f"[inspect] decoding mask_id failed: {exc}")

    ds = load_dataset("open-thoughts/OpenThoughts-114k", "default", split="train", streaming=True)
    ds_iter = iter(ds)

    # Replicate refill_buffer + get_batch packing
    token_buffer: list[int] = []
    total_rows_consumed = 0
    total_rows_dropped = 0

    def refill(needed: int):
        nonlocal total_rows_consumed, total_rows_dropped
        while len(token_buffer) < needed:
            try:
                row = next(ds_iter)
            except StopIteration:
                break
            messages = _sft_row_to_messages(row)
            total_rows_consumed += 1
            if messages is None:
                total_rows_dropped += 1
                continue
            out = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False, return_dict=True,
            )
            token_buffer.extend(out["input_ids"])
            token_buffer.append(eos_id)

    for b in range(args.batches):
        print("\n" + "=" * 80)
        print(f"BATCH {b + 1}/{args.batches}")
        print("=" * 80)
        needed = args.batch * args.max_len
        refill(needed)
        if not token_buffer:
            print("[inspect] dataset exhausted before first batch")
            return

        ids = np.full((args.batch, args.max_len), tokenizer.pad_token_id, dtype=np.int64)
        for i in range(args.batch):
            length = min(args.max_len, len(token_buffer))
            if length > 0:
                ids[i, :length] = token_buffer[:length]
                token_buffer = token_buffer[length:]
        labels = ids.copy()  # full-text SFT

        # Batch-level diagnostics
        counts = Counter(ids.ravel().tolist())
        top = counts.most_common(10)
        print(f"[diag] rows consumed so far: {total_rows_consumed} (dropped={total_rows_dropped})")
        print(f"[diag] ids shape={ids.shape} dtype={ids.dtype}")
        print(f"[diag] labels == ids: {np.array_equal(ids, labels)}")
        print(f"[diag] min/max id: {ids.min()} / {ids.max()}")
        print(f"[diag] mask_id {args.mask_id} occurrences in batch: {(ids == args.mask_id).sum()}")
        print(f"[diag] eos_id {eos_id} occurrences in batch: {(ids == eos_id).sum()}")
        print(f"[diag] pad_id {tokenizer.pad_token_id} occurrences in batch: {(ids == tokenizer.pad_token_id).sum()}")
        print(f"[diag] top-10 tokens in batch:")
        for tok_id, cnt in top:
            try:
                s = tokenizer.decode([tok_id])
            except Exception:
                s = "?"
            print(f"           {tok_id:6d} x {cnt:7d}  -> {s!r}")

        # Decode head and tail of a few rows
        for r in range(min(args.rows, args.batch)):
            print("\n" + "-" * 60)
            print(f"row {r} — first 400 chars")
            print("-" * 60)
            head_text = tokenizer.decode(ids[r, :400], skip_special_tokens=False)
            print(head_text)
            print("-" * 60)
            print(f"row {r} — last 400 chars")
            print("-" * 60)
            tail_text = tokenizer.decode(ids[r, -400:], skip_special_tokens=False)
            print(tail_text)
            # EOS locations for doc boundaries
            eos_positions = np.where(ids[r] == eos_id)[0]
            print(f"[row {r}] eos at positions: {eos_positions[:20].tolist()}{'...' if len(eos_positions) > 20 else ''}  (count={len(eos_positions)})")


if __name__ == "__main__":
    main()
