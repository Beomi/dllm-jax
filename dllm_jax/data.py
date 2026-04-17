"""Data helpers, collators, and batch iterators."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import transformers

from dllm_jax.utils import parse_spec


def parse_interval(total_steps: int, interval: int | float) -> int:
    if not interval:
        return 0
    if isinstance(interval, float) and 0.0 < interval < 1.0:
        return max(1, int(math.ceil(total_steps * interval)))
    return max(1, int(interval))


@dataclass
class CollatorWrapper:
    collator: Any

    def before(self, features):
        return features

    def after(self, outputs):
        return outputs

    def __call__(self, features, return_tensors=None):
        features = self.before(features)
        outputs = self.collator(features, return_tensors=return_tensors)
        return self.after(outputs)

    def __getattr__(self, name: str):
        collator = self.__dict__.get("collator", None)
        if collator is not None:
            return getattr(collator, name)
        raise AttributeError(name)


@dataclass
class NoAttentionMaskWrapper(CollatorWrapper):
    def after(self, outputs):
        outputs.pop("attention_mask", None)
        return outputs


@dataclass
class AppendEOSBlockWrapper(CollatorWrapper):
    tokenizer: transformers.PreTrainedTokenizerBase
    block_size: int = 32

    def before(self, features):
        for feature in features:
            ids = feature["input_ids"]
            labels = feature["labels"]
            length = len(ids)
            target = ((length + self.block_size - 1) // self.block_size) * self.block_size
            pad_len = target - length
            if pad_len > 0:
                feature["input_ids"] = ids + [self.tokenizer.eos_token_id] * pad_len
                feature["labels"] = labels + [self.tokenizer.eos_token_id] * pad_len
        return features


@dataclass
class DreamSFTCollator(transformers.DataCollatorForSeq2Seq):
    perbatch_cutoff: bool = True
    resp_cutoff_ratio: float = 0.0

    def apply_perbatch_cutoff(self, features):
        response_lengths = np.asarray(
            [len(f["input_ids"]) - f["prompt_len"] for f in features], dtype=np.int32
        )
        kept_len = int(np.random.choice(response_lengths))
        for feature, response_length in zip(features, response_lengths):
            remove_len = max(int(response_length) - kept_len, 0)
            if remove_len > 0:
                for key in ("input_ids", "labels", "attention_mask"):
                    if key in feature:
                        feature[key] = feature[key][:-remove_len]
        return features

    def apply_resp_cutoff(self, batch, features):
        response_lengths = np.asarray(
            [len(f["input_ids"]) - f["prompt_len"] for f in features], dtype=np.int32
        )
        min_response = int(response_lengths.min())
        if min_response <= 1:
            return batch
        cutoff_len = int(np.random.randint(1, min_response))
        new_seq_len = max(len(f["input_ids"]) for f in features) - cutoff_len
        for key in ("input_ids", "labels", "attention_mask"):
            if key in batch:
                batch[key] = np.asarray(batch[key])[:, :new_seq_len]
        return batch

    def __call__(self, features, return_tensors=None):
        if self.perbatch_cutoff:
            features = self.apply_perbatch_cutoff(features)
        base = [
            {k: f[k] for k in ("input_ids", "labels", "attention_mask") if k in f}
            for f in features
        ]
        batch = super().__call__(base, return_tensors=return_tensors)
        if (
            not self.perbatch_cutoff
            and self.resp_cutoff_ratio > 0
            and np.random.rand() < self.resp_cutoff_ratio
        ):
            batch = self.apply_resp_cutoff(batch, features)
        batch.pop("prompt_len", None)
        return batch


@dataclass
class DMaxDataCollator(transformers.DataCollatorForSeq2Seq):
    """Seq2Seq-style padding that preserves DMax OPUT flags and noisy ids."""

    def _pad_noisy_input_ids(self, sequences, target_length: int):
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0
        output = np.full(
            (len(sequences), target_length),
            int(pad_token_id),
            dtype=np.int64,
        )
        padding_side = getattr(self.tokenizer, "padding_side", "right")
        for idx, sequence in enumerate(sequences):
            values = np.asarray(sequence, dtype=np.int64)
            if values.size == 0:
                continue
            if padding_side == "left":
                output[idx, -min(values.size, target_length) :] = values[-target_length:]
            else:
                output[idx, : min(values.size, target_length)] = values[:target_length]
        return output

    def __call__(self, features, return_tensors=None):
        clean_features = []
        flags = []
        noisy_input_ids = []
        has_flag = any("flag" in feature for feature in features)
        has_noisy = any("noisy_input_ids" in feature for feature in features)
        for feature in features:
            copied = dict(feature)
            if has_flag:
                flags.append(bool(copied.pop("flag", False)))
            if has_noisy:
                noisy_input_ids.append(copied.pop("noisy_input_ids", copied["input_ids"]))
            clean_features.append(copied)
        batch = super().__call__(clean_features, return_tensors=return_tensors)
        if has_noisy:
            target_length = np.asarray(batch["input_ids"]).shape[1]
            batch["noisy_input_ids"] = self._pad_noisy_input_ids(
                noisy_input_ids,
                target_length,
            )
        if has_flag:
            batch["flag"] = np.asarray(flags, dtype=np.bool_)
        return batch


@dataclass
class X0Sampler:
    tokenizer: transformers.PreTrainedTokenizerBase | None = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class SampleX0Empty(X0Sampler):
    def __call__(self, *args, **kwargs):
        return []


@dataclass
class SampleX0Masks(X0Sampler):
    length: int = 128

    def __call__(self, *args, **kwargs):
        mask_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_id is None:
            raise ValueError("tokenizer.mask_token_id is required for mask x0 sampling")
        return [int(mask_id)] * self.length


_X0_SAMPLERS: dict[str, type[X0Sampler]] = {
    "empty": SampleX0Empty,
    "masks": SampleX0Masks,
}


def make_x0_sampler(name: str, tokenizer: Any, **kwargs) -> X0Sampler:
    parsed_name, parsed_kwargs = parse_spec(name)
    cls = _X0_SAMPLERS[parsed_name.lower()]
    return cls(tokenizer=tokenizer, **parsed_kwargs, **kwargs)


@dataclass
class EditFlowCollator:
    tokenizer: transformers.PreTrainedTokenizerBase
    x0_sampler: Callable | str = "masks[length:64]"

    def __post_init__(self):
        if isinstance(self.x0_sampler, str):
            self.x0_sampler = make_x0_sampler(self.x0_sampler, tokenizer=self.tokenizer)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, list[Any]]:
        if not features:
            return {}
        batch = {key: [feature[key] for feature in features] for key in features[0]}
        batch["x1_ids"] = batch["input_ids"]
        if "prompt_len" not in batch:
            bos = self.tokenizer.bos_token_id
            batch["x1_ids"] = [
                ids if ids and ids[0] == bos else [bos] + ids for ids in batch["x1_ids"]
            ]
            batch["x0_ids"] = [
                ids[:1] + self.x0_sampler(x1_ids=ids[1:]) for ids in batch["x1_ids"]
            ]
        else:
            batch["x0_ids"] = [
                ids[:prompt_len] + self.x0_sampler(x1_ids=ids[prompt_len:])
                for ids, prompt_len in zip(batch["x1_ids"], batch["prompt_len"])
            ]
        return batch


def _iter_indices(length: int, batch_size: int, shuffle: bool, seed: int) -> Iterator[list[int]]:
    indices = list(range(length))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    for start in range(0, length, batch_size):
        yield indices[start : start + batch_size]


def iter_dataset_batches(
    dataset,
    batch_size: int,
    collator,
    *,
    shuffle: bool = False,
    seed: int = 0,
    max_steps: int | None = None,
) -> Iterator[dict[str, Any]]:
    if hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
        steps = 0
        for batch_indices in _iter_indices(len(dataset), batch_size, shuffle, seed):
            features = [dataset[int(i)] for i in batch_indices]
            yield collator(features, return_tensors="np")
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        return

    iterator = iter(dataset)
    steps = 0
    while True:
        features = []
        try:
            for _ in range(batch_size):
                features.append(next(iterator))
        except StopIteration:
            if not features:
                break
        if not features:
            break
        yield collator(features, return_tensors="np")
        steps += 1
        if max_steps is not None and steps >= max_steps:
            break


def num_batches(dataset, batch_size: int) -> int | None:
    if hasattr(dataset, "__len__"):
        return int(math.ceil(len(dataset) / batch_size))
    return None
