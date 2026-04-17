"""dllm-jax: JAX backend for Diffusion Language Modeling.

A standalone package — no PyTorch or CUDA required.
Weight loading uses safetensors + numpy via huggingface_hub.
"""

from .configs import (
    BD3LMConfig,
    DMaxConfig,
    DataArguments,
    DreamConfig,
    EditFlowConfig,
    MDLMConfig,
    ModelArguments,
    TrainingArguments,
)
from .checkpoints import restore_model_checkpoint
from .data import (
    AppendEOSBlockWrapper,
    DMaxDataCollator,
    DreamSFTCollator,
    EditFlowCollator,
    NoAttentionMaskWrapper,
    iter_dataset_batches,
    num_batches,
)
from .dmax import (
    DMaxGenerationConfig,
    DMaxGenerationOutput,
    create_block_causal_attention_mask,
    dmax_generate_spd_fast,
    dmax_generate_spd,
    dmax_generate_spd_kv_fast,
    resolve_dmax_mask_token_id,
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
from .trainers import BD3LMTrainer, DMaxTrainer, DreamTrainer, EditFlowTrainer, MDLMTrainer
from .weights import load_pretrained_weights

__all__ = [
    # Configs
    "BD3LMConfig",
    "DMaxConfig",
    "DataArguments",
    "DreamConfig",
    "EditFlowConfig",
    "MDLMConfig",
    "ModelArguments",
    "TrainingArguments",
    "restore_model_checkpoint",
    # Data
    "AppendEOSBlockWrapper",
    "DMaxDataCollator",
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
    # DMax
    "DMaxGenerationConfig",
    "DMaxGenerationOutput",
    "create_block_causal_attention_mask",
    "dmax_generate_spd_fast",
    "dmax_generate_spd",
    "dmax_generate_spd_kv_fast",
    "resolve_dmax_mask_token_id",
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
    "DMaxTrainer",
    "DreamTrainer",
    "EditFlowTrainer",
    "MDLMTrainer",
]
