"""dllm-jax: JAX backend for Diffusion Language Modeling.

A standalone package — no PyTorch or CUDA required.
Weight loading uses safetensors + numpy via huggingface_hub.
"""

from .configs import (
    BD3LMConfig,
    DataArguments,
    DreamConfig,
    EditFlowConfig,
    MDLMConfig,
    ModelArguments,
    TrainingArguments,
)
from .data import (
    AppendEOSBlockWrapper,
    DreamSFTCollator,
    EditFlowCollator,
    NoAttentionMaskWrapper,
    iter_dataset_batches,
    num_batches,
)
from .models import (
    EditFlowModel,
    GenericDecoderLM,
    GenericEncoderLM,
    ModelSpec,
    build_model_from_config,
    build_model_from_pretrained,
    model_spec_from_config,
)
from .schedulers import (
    BaseAlphaScheduler,
    BaseKappaScheduler,
    CosineAlphaScheduler,
    CosineKappaScheduler,
    CubicKappaScheduler,
    LinearAlphaScheduler,
    LinearKappaScheduler,
    make_alpha_scheduler,
    make_kappa_scheduler,
)
from .trainers import BD3LMTrainer, DreamTrainer, EditFlowTrainer, MDLMTrainer
from .weights import load_pretrained_weights

__all__ = [
    # Configs
    "BD3LMConfig",
    "DataArguments",
    "DreamConfig",
    "EditFlowConfig",
    "MDLMConfig",
    "ModelArguments",
    "TrainingArguments",
    # Data
    "AppendEOSBlockWrapper",
    "DreamSFTCollator",
    "EditFlowCollator",
    "NoAttentionMaskWrapper",
    "iter_dataset_batches",
    "num_batches",
    # Models
    "EditFlowModel",
    "GenericDecoderLM",
    "GenericEncoderLM",
    "ModelSpec",
    "build_model_from_config",
    "build_model_from_pretrained",
    "load_pretrained_weights",
    "model_spec_from_config",
    # Schedulers
    "BaseAlphaScheduler",
    "BaseKappaScheduler",
    "CosineAlphaScheduler",
    "CosineKappaScheduler",
    "CubicKappaScheduler",
    "LinearAlphaScheduler",
    "LinearKappaScheduler",
    "make_alpha_scheduler",
    "make_kappa_scheduler",
    # Trainers
    "BD3LMTrainer",
    "DreamTrainer",
    "EditFlowTrainer",
    "MDLMTrainer",
]
