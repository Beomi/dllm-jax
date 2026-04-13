"""Training configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field

from dllm_jax.utils import resolve_with_base_env


@dataclass
class ModelArguments:
    model_name_or_path: str | None = None
    dtype: str = "bfloat16"

    def __post_init__(self):
        if self.model_name_or_path:
            self.model_name_or_path = resolve_with_base_env(
                self.model_name_or_path, "BASE_MODELS_DIR"
            )


@dataclass
class DataArguments:
    dataset_args: str | None = None
    num_proc: int = 8
    disable_caching: bool = False
    max_length: int = 1024
    truncation: str = "right"
    streaming: bool = False
    load_preprocessed_data: bool = False
    text_field: str = "text"
    insert_eos: bool = False
    drop_tail: bool = True
    mask_prompt_loss: bool = True
    perbatch_cutoff: bool = True
    resp_cutoff_ratio: float = 0.0


@dataclass
class TrainingArguments:
    output_dir: str | None = None
    seed: int = 42
    num_train_epochs: float = 10.0
    max_steps: int = -1
    learning_rate: float = 2e-5
    min_learning_rate: float = 0.0
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    logging_steps: int | float = 10
    eval_steps: int | float = 0
    save_steps: int | float = 0
    eval_on_start: bool = False
    bf16: bool = True
    grad_clip_norm: float = 1.0
    shuffle: bool = True
    group_by_length: bool = False
    report_to: str = "none"
    save_only_model: bool = True


@dataclass
class MDLMConfig(TrainingArguments):
    time_epsilon: float = 1e-3
    loss_weight_type: str = "scheduler"
    loss_norm_type: str = "token"
    right_shift_logits: bool = False


@dataclass
class BD3LMConfig(MDLMConfig):
    block_size: int = 32


@dataclass
class DreamConfig(MDLMConfig):
    loss_weight_type: str = "cart[geo_p:0.3]"
    right_shift_logits: bool = True


@dataclass
class EditFlowConfig(TrainingArguments):
    time_epsilon: float = 1e-3
    normalize_per_position: bool = True
    max_w: float = 20.0
    scheduler_cls: str = "LinearKappaScheduler"
    x0_sampler: str = field(
        default="masks[length:64]",
        metadata={"help": "EditFlow x0 source sampler specification."},
    )
