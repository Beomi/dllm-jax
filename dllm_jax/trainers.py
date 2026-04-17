"""NNX trainers for MDLM, BD3LM, Dream, and EditFlow."""

from __future__ import annotations

import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from flax.training import checkpoints
import numpy as np
import optax

from dllm_jax.configs import BD3LMConfig, DMaxConfig, DreamConfig, EditFlowConfig, MDLMConfig
from dllm_jax.data import iter_dataset_batches, num_batches, parse_interval
from dllm_jax.schedulers import (
    BaseAlphaScheduler,
    BaseKappaScheduler,
    CubicKappaScheduler,
    LinearAlphaScheduler,
)

BLANK = -1


def resolve_mask_token_id(tokenizer, vocab_size: int | None = None) -> int:
    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is None:
        mask_token_id = getattr(tokenizer, "mask_id", None)
    if mask_token_id is None and vocab_size is not None:
        mask_token_id = int(vocab_size) - 1
    if mask_token_id is None:
        raise ValueError(
            "A mask token id is required. Set tokenizer.mask_token_id or pass a "
            "model with spec.vocab_size so the final vocabulary id can be used."
        )
    return int(mask_token_id)


def prepend_bos(batch: dict[str, jnp.ndarray], bos_token_id: int, label_pad_token_id: int = -100):
    bos = jnp.full((batch["input_ids"].shape[0], 1), bos_token_id, dtype=batch["input_ids"].dtype)
    batch["input_ids"] = jnp.concatenate([bos, batch["input_ids"]], axis=1)
    if "labels" in batch:
        pad = jnp.full((batch["labels"].shape[0], 1), label_pad_token_id, dtype=batch["labels"].dtype)
        batch["labels"] = jnp.concatenate([pad, batch["labels"]], axis=1)
    if "attention_mask" in batch:
        attn = jnp.ones((batch["attention_mask"].shape[0], 1), dtype=batch["attention_mask"].dtype)
        batch["attention_mask"] = jnp.concatenate([attn, batch["attention_mask"]], axis=1)
    return batch


def cart_weight(masked_mask: jnp.ndarray, p: float = 0.3) -> jnp.ndarray:
    seq_len = masked_mask.shape[1]
    idx = jnp.arange(seq_len)
    dist = idx[None, :] - idx[:, None]
    geo = jnp.exp(jnp.log(p) + jnp.maximum(jnp.abs(dist) - 1, 0) * jnp.log(1 - p)) * 0.5
    geo = jnp.where(dist == 0, 0.0, geo)
    valid_mask = (~masked_mask).astype(jnp.float32)
    return (valid_mask @ geo.T) * masked_mask.astype(jnp.float32)


def create_bd3lm_attention_mask(seq_len: int, block_size: int) -> jnp.ndarray:
    q_idx = jnp.arange(seq_len * 2)[:, None]
    kv_idx = jnp.arange(seq_len * 2)[None, :]
    x0_q = q_idx >= seq_len
    x0_kv = kv_idx >= seq_len
    block_q = jnp.where(x0_q, (q_idx - seq_len) // block_size, q_idx // block_size)
    block_kv = jnp.where(x0_kv, (kv_idx - seq_len) // block_size, kv_idx // block_size)
    block_diagonal = (block_q == block_kv) & (x0_q == x0_kv)
    offset_block_causal = (block_q > block_kv) & x0_kv & (~x0_q)
    block_causal = (block_q >= block_kv) & x0_kv & x0_q
    return block_diagonal | offset_block_causal | block_causal


def align_with_blanks(x0: list[int], x1: list[int], sub_cost: int = 1, gap_cost: int = 1):
    n, m = len(x0), len(x1)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    ptr = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i * gap_cost
        ptr[i][0] = "up"
    for j in range(1, m + 1):
        dp[0][j] = j * gap_cost
        ptr[0][j] = "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_diag = dp[i - 1][j - 1] + (0 if x0[i - 1] == x1[j - 1] else sub_cost)
            cost_up = dp[i - 1][j] + gap_cost
            cost_left = dp[i][j - 1] + gap_cost
            best = min(cost_diag, cost_up, cost_left)
            dp[i][j] = best
            ptr[i][j] = "diag" if best == cost_diag else "up" if best == cost_up else "left"
    z0, z1 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        move = ptr[i][j]
        if move == "diag":
            z0.append(x0[i - 1])
            z1.append(x1[j - 1])
            i -= 1
            j -= 1
        elif move == "up":
            z0.append(x0[i - 1])
            z1.append(BLANK)
            i -= 1
        else:
            z0.append(BLANK)
            z1.append(x1[j - 1])
            j -= 1
    z0.reverse()
    z1.reverse()
    return {"z0": z0, "z1": z1}


def strip_blanks(z):
    return [token for token in z if token != BLANK]


@dataclass
class Edit:
    kind: str
    pos: int
    token: int | None


def build_remaining_edits(zt: list[int], z1: list[int]) -> list[Edit]:
    edits: list[Edit] = []

    def count_nonblank_prefix(z: list[int], idx: int):
        return sum(1 for token in z[:idx] if token != BLANK)

    for idx, (src, tgt) in enumerate(zip(zt, z1)):
        if src == tgt:
            continue
        nonblank = count_nonblank_prefix(zt, idx)
        if src == BLANK and tgt != BLANK:
            edits.append(Edit("INS", max(nonblank - 1, 0), tgt))
        elif src != BLANK and tgt == BLANK:
            edits.append(Edit("DEL", nonblank, None))
        else:
            edits.append(Edit("SUB", nonblank, tgt))
    return edits


def pad_1d(batch_lists: list[list[int]], pad_val: int):
    batch_size = len(batch_lists)
    max_len = max((len(values) for values in batch_lists), default=0)
    output = np.full((batch_size, max_len), pad_val, dtype=np.int32)
    mask = np.zeros((batch_size, max_len), dtype=np.int32)
    for idx, values in enumerate(batch_lists):
        if not values:
            continue
        output[idx, : len(values)] = np.asarray(values, dtype=np.int32)
        mask[idx, : len(values)] = 1
    return output, mask


class BaseTrainer:
    def __init__(
        self,
        *,
        model,
        tokenizer,
        args,
        train_dataset,
        eval_dataset=None,
        data_collator=None,
        config=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.global_step = 0
        self.rng = jax.random.key(args.seed)
        self.total_steps = self._resolve_total_steps()
        self._lr_schedule = self._build_schedule()
        self.optimizer = nnx.Optimizer(
            self.model,
            self._build_optimizer(),
            wrt=nnx.Param,
        )
        self._compiled_train_step = nnx.cached_partial(
            self._build_train_step(),
            self.model,
            self.optimizer,
        )
        self._compiled_eval_step = nnx.cached_partial(
            self._build_eval_step(),
            self.model,
        )

    def _resolve_total_steps(self) -> int:
        if self.args.max_steps and self.args.max_steps > 0:
            return int(self.args.max_steps)
        train_batches = num_batches(self.train_dataset, self.args.per_device_train_batch_size)
        if train_batches is None:
            raise ValueError("Streaming training requires --max_steps for the JAX backend.")
        return max(1, int(math.ceil(self.args.num_train_epochs * train_batches)))

    def _build_schedule(self):
        if self.total_steps <= 1:
            return optax.constant_schedule(self.args.learning_rate)
        warmup_steps = min(
            max(1, int(self.total_steps * self.args.warmup_ratio)),
            self.total_steps - 1,
        )
        if self.args.lr_scheduler_type == "linear":
            return optax.join_schedules(
                schedules=[
                    optax.linear_schedule(init_value=0.0, end_value=self.args.learning_rate, transition_steps=warmup_steps),
                    optax.linear_schedule(init_value=self.args.learning_rate, end_value=self.args.min_learning_rate, transition_steps=max(1, self.total_steps - warmup_steps)),
                ],
                boundaries=[warmup_steps],
            )
        if self.args.lr_scheduler_type == "cosine":
            return optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=self.args.learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=max(2, self.total_steps),
                end_value=self.args.min_learning_rate,
            )
        return optax.constant_schedule(self.args.learning_rate)

    def _build_optimizer(self):
        return optax.chain(
            optax.clip_by_global_norm(self.args.grad_clip_norm),
            optax.adamw(
                learning_rate=self._lr_schedule,
                weight_decay=self.args.weight_decay,
            ),
        )

    def prepare_batch(self, raw_batch: dict[str, Any], rng_key):
        raise NotImplementedError

    def loss_fn(self, model, batch: dict[str, jnp.ndarray]):
        raise NotImplementedError

    def _build_train_step(self):
        loss_fn = self.loss_fn

        @nnx.jit
        def train_step(model, optimizer, batch):
            def objective(current_model):
                return loss_fn(current_model, batch)
            (_, metrics), grads = nnx.value_and_grad(objective, has_aux=True)(model)
            optimizer.update(grads)
            return metrics

        return train_step

    def _build_microbatch_grad_step(self):
        """Compute grads for one micro-batch without applying them.

        Used by the gradient-accumulation path to accumulate grads across
        ``gradient_accumulation_steps`` micro-batches before calling
        ``optimizer.update``. Mirrors reference train_llada2_bd_oput.py which
        divides each micro-batch's loss by ``len(micro_batches)`` and sums.
        """

        loss_fn = self.loss_fn

        @nnx.jit
        def grad_step(model, batch, scale):
            def objective(current_model):
                loss, metrics = loss_fn(current_model, batch)
                scaled = loss * scale
                return scaled, metrics
            (_, metrics), grads = nnx.value_and_grad(objective, has_aux=True)(model)
            return grads, metrics

        return grad_step

    def _apply_accumulated_grads(self):
        optimizer = self.optimizer
        model = self.model

        @nnx.jit
        def apply(model, optimizer, grads):
            optimizer.update(grads)
        return apply

    def _build_eval_step(self):
        loss_fn = self.loss_fn

        @nnx.jit
        def eval_step(model, batch):
            _, metrics = loss_fn(model, batch)
            return metrics

        return eval_step

    def evaluate(self):
        if self.eval_dataset is None:
            return {}
        losses = []
        iterator = iter_dataset_batches(
            self.eval_dataset,
            self.args.per_device_eval_batch_size,
            self.data_collator,
            shuffle=False,
            seed=self.args.seed,
        )
        for raw_batch in iterator:
            self.rng, step_key = jax.random.split(self.rng)
            batch = self.prepare_batch(raw_batch, step_key)
            metrics = self._compiled_eval_step(batch)
            losses.append(float(metrics["loss"]))
        if not losses:
            return {}
        return {"eval_loss": float(np.mean(losses))}

    def save_model(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        model_state = jax.tree.map(jax.device_get, nnx.state(self.model))
        if self.args.save_only_model:
            with open(os.path.join(output_dir, "model_state.pkl"), "wb") as f:
                pickle.dump(
                    {"model": model_state, "step": np.asarray(self.global_step, dtype=np.int32)},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        else:
            target = {
                "model": model_state,
                "step": np.asarray(self.global_step, dtype=np.int32),
            }
            target["optimizer"] = jax.tree.map(jax.device_get, nnx.state(self.optimizer))
            checkpoints.save_checkpoint(ckpt_dir=output_dir, target=target, step=self.global_step, overwrite=True)
        if self.config is not None:
            self.config.save_pretrained(output_dir)
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "training_state.json"), "w") as f:
            json.dump({"global_step": int(self.global_step)}, f)

    def train(self):
        grad_accum = max(1, int(self.args.gradient_accumulation_steps))
        logging_steps = parse_interval(self.total_steps, self.args.logging_steps)
        eval_steps = parse_interval(self.total_steps, self.args.eval_steps)
        save_steps = parse_interval(self.total_steps, self.args.save_steps)

        if grad_accum > 1:
            micro_grad_step = nnx.cached_partial(
                self._build_microbatch_grad_step(), self.model
            )
            apply_step = nnx.cached_partial(
                self._apply_accumulated_grads(), self.model, self.optimizer
            )
            scale = jnp.asarray(1.0 / grad_accum, dtype=jnp.float32)

        if self.args.eval_on_start and self.eval_dataset is not None:
            print(self.evaluate())

        step = 0
        epoch = 0
        while step < self.total_steps:
            iterator = iter_dataset_batches(
                self.train_dataset,
                self.args.per_device_train_batch_size,
                self.data_collator,
                shuffle=self.args.shuffle,
                seed=self.args.seed + epoch,
                max_steps=(self.total_steps - step) * grad_accum,
            )
            micro_batches: list[dict[str, Any]] = []
            last_metrics = None
            for raw_batch in iterator:
                self.rng, step_key = jax.random.split(self.rng)
                batch = self.prepare_batch(raw_batch, step_key)

                if grad_accum == 1:
                    last_metrics = self._compiled_train_step(batch)
                    step += 1
                else:
                    micro_batches.append(batch)
                    if len(micro_batches) < grad_accum:
                        continue
                    accumulated = None
                    accumulated_loss = 0.0
                    for mb in micro_batches:
                        grads, metrics = micro_grad_step(mb, scale)
                        accumulated = (
                            grads
                            if accumulated is None
                            else jax.tree.map(jnp.add, accumulated, grads)
                        )
                        accumulated_loss += float(metrics["loss"])
                    apply_step(accumulated)
                    micro_batches = []
                    last_metrics = {"loss": jnp.asarray(accumulated_loss / grad_accum)}
                    step += 1

                self.global_step = step
                if logging_steps and step % logging_steps == 0 and last_metrics is not None:
                    print({
                        "step": step,
                        "loss": float(last_metrics["loss"]),
                        "learning_rate": float(self._lr_schedule(step)),
                    })
                if eval_steps and self.eval_dataset is not None and step % eval_steps == 0:
                    print({"step": step, **self.evaluate()})
                if save_steps and step % save_steps == 0:
                    self.save_model(os.path.join(self.args.output_dir, f"checkpoint-{step}"))
                if step >= self.total_steps:
                    break
            epoch += 1
        return {"global_step": self.global_step}


class MDLMTrainer(BaseTrainer):
    def __init__(self, *, model, tokenizer, args: MDLMConfig, train_dataset, eval_dataset=None, data_collator=None, config=None, scheduler: BaseAlphaScheduler | None = None):
        self.scheduler = scheduler if scheduler is not None else LinearAlphaScheduler()
        super().__init__(model=model, tokenizer=tokenizer, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator, config=config)

    def compute_loss_weights(self, t, inputs, masked_mask):
        if self.args.loss_weight_type == "scheduler":
            return self.scheduler.weight(t)[:, None].repeat(inputs["input_ids"].shape[1], axis=1)
        if self.args.loss_weight_type == "uniform":
            return jnp.ones_like(inputs["input_ids"], dtype=jnp.float32)
        raise NotImplementedError(self.args.loss_weight_type)

    def prepare_batch(self, raw_batch: dict[str, Any], rng_key):
        batch = {key: jnp.asarray(value) for key, value in raw_batch.items()}
        if self.args.right_shift_logits:
            labels = batch.get("labels")
            needs_bos = labels is None or not jnp.all(labels[:, 0] == -100)
            if needs_bos:
                batch = prepend_bos(batch, bos_token_id=self.tokenizer.bos_token_id, label_pad_token_id=-100)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch.get("attention_mask")
        maskable_mask = labels != -100
        rng_t, rng_mask = jax.random.split(rng_key)
        t = self.args.time_epsilon + (1.0 - self.args.time_epsilon) * jax.random.uniform(rng_t, (input_ids.shape[0],), dtype=jnp.float32)
        p_mask = 1.0 - self.scheduler(t)[:, None]
        masked_mask = jax.random.bernoulli(rng_mask, p_mask, shape=input_ids.shape) & maskable_mask
        noised_input_ids = jnp.where(masked_mask, self.tokenizer.mask_token_id, input_ids)
        loss_weights = self.compute_loss_weights(t=t, inputs=batch, masked_mask=masked_mask)
        return {
            "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "maskable_mask": maskable_mask, "masked_mask": masked_mask,
            "loss_weights": loss_weights, "model_input_ids": noised_input_ids,
        }

    def postprocess_logits(self, logits):
        if not self.args.right_shift_logits:
            return logits
        return jnp.concatenate([logits[:, :1], logits[:, :-1]], axis=1)

    def loss_fn(self, model, batch: dict[str, jnp.ndarray]):
        outputs = model(batch["model_input_ids"], attention_mask=batch.get("attention_mask"))
        logits = self.postprocess_logits(outputs["logits"])
        token_nll = optax.softmax_cross_entropy_with_integer_labels(logits, batch["input_ids"])
        token_nll = token_nll * batch["loss_weights"] * batch["masked_mask"].astype(token_nll.dtype)
        if self.args.loss_norm_type == "token":
            token_nll = token_nll / jnp.clip(batch["maskable_mask"].sum(), min=1)
        elif self.args.loss_norm_type == "sequence":
            token_nll = token_nll / (jnp.clip(batch["maskable_mask"].sum(axis=-1, keepdims=True), min=1) * batch["maskable_mask"].shape[0])
        elif self.args.loss_norm_type == "batch":
            token_nll = token_nll / batch["maskable_mask"].shape[0]
        else:
            raise ValueError(f"Invalid loss_norm_type: {self.args.loss_norm_type}")
        loss = token_nll.sum()
        return loss, {"loss": loss}


class BD3LMTrainer(MDLMTrainer):
    def __init__(self, *, model, tokenizer, args: BD3LMConfig, train_dataset, eval_dataset=None, data_collator=None, config=None, scheduler: BaseAlphaScheduler | None = None):
        super().__init__(model=model, tokenizer=tokenizer, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator, config=config, scheduler=scheduler)

    def prepare_batch(self, raw_batch: dict[str, Any], rng_key):
        batch = super().prepare_batch(raw_batch, rng_key)
        seq_len = batch["input_ids"].shape[1]
        base_pos = jnp.broadcast_to(jnp.arange(seq_len)[None, :], batch["input_ids"].shape)
        batch["attention_mask"] = create_bd3lm_attention_mask(seq_len, self.args.block_size)
        batch["position_ids"] = jnp.concatenate([base_pos, base_pos], axis=1)
        batch["model_input_ids"] = jnp.concatenate([batch["model_input_ids"], batch["input_ids"]], axis=1)
        return batch

    def loss_fn(self, model, batch: dict[str, jnp.ndarray]):
        outputs = model(batch["model_input_ids"], attention_mask=batch["attention_mask"], position_ids=batch["position_ids"])
        logits = self.postprocess_logits(outputs["logits"])[:, : batch["input_ids"].shape[1]]
        token_nll = optax.softmax_cross_entropy_with_integer_labels(logits, batch["input_ids"])
        token_nll = token_nll * batch["loss_weights"] * batch["masked_mask"].astype(token_nll.dtype)
        if self.args.loss_norm_type == "token":
            token_nll = token_nll / jnp.clip(batch["maskable_mask"].sum(), min=1)
        elif self.args.loss_norm_type == "sequence":
            token_nll = token_nll / (jnp.clip(batch["maskable_mask"].sum(axis=-1, keepdims=True), min=1) * batch["maskable_mask"].shape[0])
        elif self.args.loss_norm_type == "batch":
            token_nll = token_nll / batch["maskable_mask"].shape[0]
        else:
            raise ValueError(f"Invalid loss_norm_type: {self.args.loss_norm_type}")
        loss = token_nll.sum()
        return loss, {"loss": loss}


class DMaxTrainer(BaseTrainer):
    """On-Policy Uniform Training (OPUT) for DMax-style dLLMs."""

    def __init__(
        self,
        *,
        model,
        tokenizer,
        args: DMaxConfig,
        train_dataset,
        eval_dataset=None,
        data_collator=None,
        config=None,
    ):
        vocab_size = getattr(getattr(model, "spec", None), "vocab_size", None)
        self.mask_token_id = resolve_mask_token_id(tokenizer, vocab_size=vocab_size)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        # If the tokenizer falls back pad_token_id onto eos/unk whose id happens
        # to collide with mask_token_id, the trainer would silently treat padding
        # as supervised masked positions.
        if pad_token_id is not None and int(pad_token_id) == int(self.mask_token_id):
            raise ValueError(
                f"tokenizer.pad_token_id ({pad_token_id}) collides with "
                f"mask_token_id ({self.mask_token_id}); set a distinct pad_token "
                "before constructing DMaxTrainer."
            )
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            config=config,
        )

    def prepare_batch(self, raw_batch: dict[str, Any], rng_key):
        batch = {key: jnp.asarray(value) for key, value in raw_batch.items()}
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        if labels is None:
            labels = input_ids
            batch["labels"] = labels
        maskable_mask = labels != -100
        attention_mask = batch.get("attention_mask")

        rng_noise, rng_mask, rng_flag = jax.random.split(rng_key, 3)
        if "noisy_input_ids" in batch:
            noised_input_ids = batch["noisy_input_ids"]
            masked_mask = (noised_input_ids == self.mask_token_id) & maskable_mask
        else:
            if self.args.noise_range_low == self.args.noise_range_high:
                p_mask = jnp.full(
                    (input_ids.shape[0], 1),
                    self.args.noise_range_low,
                    dtype=jnp.float32,
                )
            else:
                p_mask = jax.random.uniform(
                    rng_noise,
                    (input_ids.shape[0], 1),
                    minval=self.args.noise_range_low,
                    maxval=self.args.noise_range_high,
                    dtype=jnp.float32,
                )
            masked_mask = jax.random.bernoulli(rng_mask, p_mask, shape=input_ids.shape) & maskable_mask
            noised_input_ids = jnp.where(masked_mask, self.mask_token_id, input_ids)

        if "flag" in batch:
            on_policy_flag = batch["flag"].astype(bool)
            if on_policy_flag.ndim > 1:
                on_policy_flag = on_policy_flag.reshape((input_ids.shape[0], -1))[:, 0]
            elif on_policy_flag.ndim == 0:
                on_policy_flag = jnp.broadcast_to(on_policy_flag, (input_ids.shape[0],))
        else:
            on_policy_flag = jax.random.bernoulli(
                rng_flag,
                self.args.on_policy_ratio,
                shape=(input_ids.shape[0],),
            )

        seq_len = input_ids.shape[1]
        base_pos = jnp.broadcast_to(jnp.arange(seq_len)[None, :], input_ids.shape)
        position_ids = jnp.concatenate([base_pos, base_pos], axis=1)
        return {
            "input_ids": input_ids,
            "attention_mask": create_bd3lm_attention_mask(seq_len, self.args.block_size),
            "raw_attention_mask": attention_mask,
            "labels": labels,
            "maskable_mask": maskable_mask,
            "masked_mask": masked_mask,
            "loss_weights": jnp.ones_like(input_ids, dtype=jnp.float32),
            "model_input_ids": jnp.concatenate([noised_input_ids, input_ids], axis=1),
            "position_ids": position_ids,
            "on_policy_flag": on_policy_flag,
        }

    def _loss_mask_and_targets(self, batch: dict[str, jnp.ndarray], logits: jnp.ndarray):
        # DMax OPUT trains the noised stream to predict x0 at every supervised
        # position (labels != -100), not only at the currently-masked ones.
        # Under the two-stream block-causal attention the clean stream provides
        # context and is excluded from the loss; the noised stream's logits at
        # position i are compared to the clean token at i (no AR-style shift).
        if not self.args.same_token_labels:
            raise NotImplementedError(
                "DMax OPUT only supports same_token_labels=True; the AR-shift "
                "branch was a vestigial autoregressive loss and has been removed."
            )
        targets = batch["input_ids"]
        loss_mask = batch["maskable_mask"]
        return logits, targets, loss_mask, batch["loss_weights"]

    def loss_fn(self, model, batch: dict[str, jnp.ndarray]):
        seq_len = batch["input_ids"].shape[1]
        model_input_ids = batch["model_input_ids"]

        rollout_logits = jax.lax.stop_gradient(
            model(
                model_input_ids,
                attention_mask=batch["attention_mask"],
                position_ids=batch["position_ids"],
            )["logits"]
        )
        semi_input_ids = jnp.argmax(rollout_logits[:, :seq_len], axis=-1)
        on_policy = batch["on_policy_flag"][:, None]
        noised_input_ids = jnp.where(
            batch["masked_mask"] & on_policy,
            semi_input_ids,
            model_input_ids[:, :seq_len],
        )
        model_input_ids = jnp.concatenate([noised_input_ids, model_input_ids[:, seq_len:]], axis=1)

        outputs = model(
            model_input_ids,
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
        )
        logits, targets, loss_mask, loss_weights = self._loss_mask_and_targets(
            batch,
            outputs["logits"][:, :seq_len],
        )
        token_nll = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        token_nll = token_nll * loss_weights * loss_mask.astype(token_nll.dtype)
        if self.args.loss_norm_type == "token":
            token_nll = token_nll / jnp.clip(loss_mask.sum(), min=1)
        elif self.args.loss_norm_type == "sequence":
            token_nll = token_nll / (
                jnp.clip(loss_mask.sum(axis=-1, keepdims=True), min=1)
                * loss_mask.shape[0]
            )
        elif self.args.loss_norm_type == "batch":
            token_nll = token_nll / loss_mask.shape[0]
        else:
            raise ValueError(f"Invalid loss_norm_type: {self.args.loss_norm_type}")
        loss = token_nll.sum()
        return loss, {"loss": loss}


class DreamTrainer(MDLMTrainer):
    def __init__(self, *, model, tokenizer, args: DreamConfig, train_dataset, eval_dataset=None, data_collator=None, config=None, scheduler: BaseAlphaScheduler | None = None):
        super().__init__(model=model, tokenizer=tokenizer, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator, config=config, scheduler=scheduler)

    def compute_loss_weights(self, t, inputs, masked_mask):
        if self.args.loss_weight_type.startswith("cart"):
            match = re.search(r"geo_p:(0\.\d+)", self.args.loss_weight_type)
            geo_p = float(match.group(1)) if match else 0.3
            return cart_weight(masked_mask, p=geo_p)
        return super().compute_loss_weights(t=t, inputs=inputs, masked_mask=masked_mask)


class EditFlowTrainer(BaseTrainer):
    def __init__(self, *, model, tokenizer, args: EditFlowConfig, train_dataset, eval_dataset=None, data_collator=None, config=None, scheduler: BaseKappaScheduler | None = None):
        self.scheduler = scheduler if scheduler is not None else CubicKappaScheduler()
        super().__init__(model=model, tokenizer=tokenizer, args=args, train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator, config=config)

    def _edits_to_dense(self, edits_list: list[list[Edit]], kind: str):
        counts = [sum(1 for edit in edits if edit.kind == kind) for edits in edits_list]
        width = max(max(counts, default=0), 1)
        batch_size = len(edits_list)
        positions = np.zeros((batch_size, width), dtype=np.int32)
        tokens = np.zeros((batch_size, width), dtype=np.int32)
        mask = np.zeros((batch_size, width), dtype=np.float32)
        for batch_idx, edits in enumerate(edits_list):
            selected = [edit for edit in edits if edit.kind == kind]
            for edit_idx, edit in enumerate(selected):
                positions[batch_idx, edit_idx] = edit.pos
                tokens[batch_idx, edit_idx] = -1 if edit.token is None else edit.token
                mask[batch_idx, edit_idx] = 1.0
        return positions, tokens, mask

    def prepare_batch(self, raw_batch: dict[str, Any], rng_key):
        x0_ids = raw_batch["x0_ids"]
        x1_ids = raw_batch["x1_ids"]
        aligns = [align_with_blanks(x0, x1) for x0, x1 in zip(x0_ids, x1_ids)]
        z0_list = [align["z0"] for align in aligns]
        z1_list = [align["z1"] for align in aligns]
        batch_size = len(z0_list)
        rng_t, rng_mix = jax.random.split(rng_key)
        t = (1.0 - self.args.time_epsilon) * jax.random.uniform(rng_t, (batch_size,), dtype=jnp.float32)
        kappa = np.asarray(self.scheduler.kappa(t[:, None]))
        w = np.asarray(self.scheduler.weight(t[:, None]).squeeze(1))
        if self.args.max_w:
            w = np.minimum(w, self.args.max_w)

        mix_keys = jax.random.split(rng_mix, batch_size)
        zt_list = []
        for z0, z1, kappa_value, mix_key in zip(z0_list, z1_list, kappa.squeeze(1), mix_keys):
            choose_target = np.asarray(jax.random.bernoulli(mix_key, float(kappa_value), (len(z0),)))
            zt_list.append([target if choose_target[idx] else src for idx, (src, target) in enumerate(zip(z0, z1))])

        xt_list = [strip_blanks(zt) for zt in zt_list]
        edits_list = [build_remaining_edits(zt, z1) for zt, z1 in zip(zt_list, z1_list)]
        x_tok, x_mask = pad_1d(xt_list, pad_val=self.tokenizer.pad_token_id)
        sub_pos, sub_tok, sub_mask = self._edits_to_dense(edits_list, "SUB")
        ins_pos, ins_tok, ins_mask = self._edits_to_dense(edits_list, "INS")
        del_pos, _, del_mask = self._edits_to_dense(edits_list, "DEL")
        denom = (
            np.asarray([len(x1) for x1 in x1_ids], dtype=np.float32)
            if self.args.normalize_per_position
            else np.ones(batch_size, dtype=np.float32)
        )
        return {
            "x_tok": jnp.asarray(x_tok), "x_mask": jnp.asarray(x_mask),
            "t": jnp.asarray(t[:, None]), "w": jnp.asarray(w),
            "denom": jnp.asarray(np.clip(denom, 1.0, None)),
            "sub_pos": jnp.asarray(sub_pos), "sub_tok": jnp.asarray(np.clip(sub_tok, 0, None)), "sub_mask": jnp.asarray(sub_mask),
            "ins_pos": jnp.asarray(ins_pos), "ins_tok": jnp.asarray(np.clip(ins_tok, 0, None)), "ins_mask": jnp.asarray(ins_mask),
            "del_pos": jnp.asarray(del_pos), "del_mask": jnp.asarray(del_mask),
        }

    def loss_fn(self, model, batch: dict[str, jnp.ndarray]):
        outputs = model(batch["x_tok"], attention_mask=batch["x_mask"], t=batch["t"])
        sub_rate_hat = outputs["sub_rate_hat"]
        del_rate_hat = outputs["del_rate_hat"]
        ins_rate_hat = outputs["ins_rate_hat"]
        log_q_sub = jax.nn.log_softmax(outputs["sub_logits"], axis=-1)
        log_q_ins = jax.nn.log_softmax(outputs["ins_logits"], axis=-1)

        mask_f = batch["x_mask"].astype(jnp.float32)
        lambda_hat = ((sub_rate_hat + del_rate_hat + ins_rate_hat) * mask_f).sum(axis=1)
        loss_surv = ((batch["w"] * lambda_hat) / batch["denom"]).mean()

        batch_indices = jnp.arange(batch["x_tok"].shape[0])[:, None]
        sub_terms = (
            log_q_sub[batch_indices, batch["sub_pos"], batch["sub_tok"]]
            + jnp.log(jnp.clip(sub_rate_hat[batch_indices, batch["sub_pos"]], min=1e-12))
        ) * batch["sub_mask"]
        ins_terms = (
            log_q_ins[batch_indices, batch["ins_pos"], batch["ins_tok"]]
            + jnp.log(jnp.clip(ins_rate_hat[batch_indices, batch["ins_pos"]], min=1e-12))
        ) * batch["ins_mask"]
        del_terms = (
            jnp.log(jnp.clip(del_rate_hat[batch_indices, batch["del_pos"]], min=1e-12))
            * batch["del_mask"]
        )
        loss_pos_per = -(sub_terms.sum(axis=1) + ins_terms.sum(axis=1) + del_terms.sum(axis=1))
        loss_pos = ((batch["w"] * loss_pos_per) / batch["denom"]).mean()
        loss = loss_surv + loss_pos
        return loss, {"loss": loss}
