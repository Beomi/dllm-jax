"""Torch-free utilities for the dllm-jax package.

Inlined from dllm.utils to remove PyTorch/accelerate dependencies.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from itertools import chain


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_with_base_env(path: str, env_name: str) -> str:
    """Resolve a relative path using an environment variable as base directory."""
    base = os.getenv(env_name, "").strip()
    if not base:
        return path
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    candidate = os.path.join(base.rstrip("/"), path.lstrip("/"))
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Path not found: {candidate}")


# ---------------------------------------------------------------------------
# Spec parsing  (e.g. "foo/bar[a:1,b:hello]")
# ---------------------------------------------------------------------------

def parse_spec(spec: str):
    """Parse 'name[key:value,...]' or 'key=value,...' style specifications.

    Returns (name, kv_dict) where name may be None.
    """
    def _parse_kv_string(s: str) -> dict:
        return dict(part.split("=", 1) for part in s.split(",") if "=" in part)

    s = spec.strip()
    m = re.search(r"\[(.*?)\]$", s)
    bracket_kvs: dict = {}
    numeric_kvs: dict = {}
    if m:
        bracket = m.group(1).strip()
        if bracket:
            for part in bracket.split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" not in part:
                    raise ValueError(f"Invalid entry '{part}' in '{spec}' (expected key:value).")
                key, value = part.split(":", 1)
                key, value = key.strip(), value.strip()
                if re.fullmatch(r"\d(?:_?\d)*", value):
                    numeric_kvs[key] = int(value.replace("_", ""))
                else:
                    bracket_kvs[key] = value
        s = s[: m.start()].rstrip()

    if "=" in s:
        kv_dict = dict(_parse_kv_string(s))
    else:
        kv_dict = {}
    name = s if s and "=" not in s else None
    kv_dict.update(bracket_kvs)
    kv_dict.update(numeric_kvs)
    return name, kv_dict


# ---------------------------------------------------------------------------
# Logging (uses jax.process_index instead of accelerate.PartialState)
# ---------------------------------------------------------------------------

def get_default_logger(name: str) -> logging.Logger:
    """Create a logger that only logs INFO on the main JAX process."""
    logger = logging.getLogger(name)
    try:
        import jax
        is_main = jax.process_index() == 0
    except Exception:
        is_main = True
    logger.setLevel(logging.INFO if is_main else logging.WARNING)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt=(
                "\x1b[38;5;110m[%(asctime)s "
                "\x1b[38;5;174m%(levelname)s "
                "\x1b[38;5;109m%(name)s"
                "/%(lineno)d-%(processName)s\x1b[38;5;110m] "
                "\x1b[0m%(message)s"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Data utilities (torch-free versions from dllm.utils.data)
# ---------------------------------------------------------------------------

def tokenize_and_group(
    examples,
    tokenizer,
    text_field: str = "text",
    seq_length: int = 1024,
    insert_eos: bool = False,
    drop_tail: bool = True,
    add_special_tokens: bool = False,
):
    """Tokenize text and group into fixed-length sequences."""
    tokenized = tokenizer(examples[text_field], add_special_tokens=add_special_tokens)
    ids = tokenized["input_ids"]
    if insert_eos:
        eos_id = tokenizer.eos_token_id
        assert eos_id is not None
        ids = [seq + ([] if (seq and seq[-1] == eos_id) else [eos_id]) for seq in ids]
    concatenated = list(chain.from_iterable(ids))
    if not concatenated:
        return {"input_ids": [], "labels": []}
    if drop_tail:
        total_len = (len(concatenated) // seq_length) * seq_length
        concatenated = concatenated[:total_len]
    else:
        total_len = len(concatenated)
    chunks = [concatenated[i : i + seq_length] for i in range(0, total_len, seq_length)]
    return {"input_ids": chunks, "labels": [c[:] for c in chunks]}


def clip_row(row: dict, max_length: int, truncation: str = "right") -> dict:
    """Truncate sequence fields to max_length."""
    for key in ("input_ids", "labels", "attention_mask"):
        if key in row:
            if truncation == "right":
                row[key] = row[key][:max_length]
            elif truncation == "left":
                row[key] = row[key][-max_length:]
            else:
                raise NotImplementedError(f"Unknown truncation: {truncation}")
    return row


def post_process_dataset(dataset, data_args) -> "datasets.DatasetDict":
    """Filter or truncate dataset sequences based on data_args."""
    if data_args.truncation == "filter":
        return dataset.filter(
            lambda row: len(row["input_ids"]) <= data_args.max_length,
            num_proc=data_args.num_proc,
            desc=f"Filtering samples with length <= {data_args.max_length}",
        )
    if data_args.truncation == "right":
        if "prompt_len" in dataset.column_names.get("train", []):
            dataset = dataset.filter(
                lambda row: row["prompt_len"] <= data_args.max_length,
                num_proc=data_args.num_proc,
                desc=f"Filtering samples with prompt_len <= {data_args.max_length}",
            )
        return dataset.map(
            lambda row: clip_row(row, data_args.max_length, truncation="right"),
            num_proc=data_args.num_proc,
            desc=f"Right-truncating samples to max_length={data_args.max_length}",
        )
    raise NotImplementedError(f"Unknown truncation: {data_args.truncation}")


def default_sft_map_fn(row, *, tokenizer, mask_prompt_loss: bool = True) -> dict:
    """Build input_ids and labels for SFT from a row with 'messages'."""
    prompt_response_tokens = tokenizer.apply_chat_template(
        row["messages"], tokenize=True, add_generation_prompt=False
    )
    labels = prompt_response_tokens.copy()
    if mask_prompt_loss:
        prompt_tokens = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=True, add_generation_prompt=True
        )
        labels[: len(prompt_tokens)] = [-100] * len(prompt_tokens)
        return {
            "input_ids": prompt_response_tokens,
            "labels": labels,
            "prompt_len": len(prompt_tokens),
        }
    return {"input_ids": prompt_response_tokens, "labels": labels}


# ---------------------------------------------------------------------------
# Dataset loading (simplified, no custom dataset-specific loaders)
# ---------------------------------------------------------------------------

def load_dataset_from_spec(
    dataset_args: str,
    *,
    streaming: bool = False,
) -> "datasets.DatasetDict | datasets.IterableDatasetDict":
    """Load a HuggingFace dataset from a spec string.

    Examples:
        "wikimedia/wikipedia[name:20231101.en]"
        "tatsu-lab/alpaca[train:5000]"
        "wikitext[name:wikitext-103-v1]"

    For custom dataset loaders, load datasets directly and pass to trainers.
    """
    from datasets import DatasetDict, IterableDatasetDict, load_dataset

    specs = [p.strip() for p in re.split(r"[|+]", dataset_args) if p.strip()]
    if not specs:
        raise ValueError("Empty dataset_args.")

    parts = []
    for raw in specs:
        name_or_path, kvs = parse_spec(raw)
        name_or_path = resolve_with_base_env(name_or_path, "BASE_DATASETS_DIR")
        subset_name = kvs.pop("name", None)
        ds = load_dataset(name_or_path, name=subset_name, streaming=streaming)

        # Normalize to dict-of-splits
        if not isinstance(ds, (DatasetDict, IterableDatasetDict, dict)):
            ds = {"train": ds}

        # Apply per-split limits (e.g. train:5000)
        out = {}
        for split_key, split_data in (ds.items() if isinstance(ds, dict) else ds.items()):
            limit = kvs.get(split_key)
            if limit is not None:
                split_data = split_data.select(range(min(int(limit), len(split_data)))) if not streaming else split_data.take(int(limit))
            out[split_key] = split_data

        if streaming:
            parts.append(IterableDatasetDict(out))
        else:
            parts.append(DatasetDict(out))

    # Merge if multiple specs
    result = parts[0]
    for p in parts[1:]:
        merged = {}
        for split in set(list(result.keys()) + list(p.keys())):
            a, b = result.get(split), p.get(split)
            if a is None:
                merged[split] = b
            elif b is None:
                merged[split] = a
            else:
                from datasets import concatenate_datasets
                merged[split] = concatenate_datasets([a, b])
        result = type(result)(merged)

    return result
