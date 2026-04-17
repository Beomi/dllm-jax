"""Checkpoint helpers for NNX models."""

from __future__ import annotations

import os
import pickle
from typing import Any

from flax import nnx
from flax.training import checkpoints


def restore_model_checkpoint(model: Any, checkpoint_dir: str) -> dict[str, Any]:
    """Restore a trainer checkpoint into an existing NNX model."""

    model_state_path = os.path.join(checkpoint_dir, "model_state.pkl")
    if os.path.exists(model_state_path):
        with open(model_state_path, "rb") as f:
            restored = pickle.load(f)
        if not restored or "model" not in restored:
            raise FileNotFoundError(f"No model state found in {model_state_path!r}.")
        nnx.update(model, restored["model"])
        return restored

    target = {"model": nnx.state(model)}
    restored = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_dir, target=target)
    if not restored or "model" not in restored:
        raise FileNotFoundError(f"No model checkpoint found in {checkpoint_dir!r}.")
    nnx.update(model, restored["model"])
    return restored
