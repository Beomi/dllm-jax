#!/bin/bash
set -euo pipefail

export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
export MAX_LEN="${MAX_LEN:-1024}"
export GLOBAL_BATCH="${GLOBAL_BATCH:-64}"
export NUM_EPOCHS="${NUM_EPOCHS:-3}"
export WARMUP_STEPS="${WARMUP_STEPS:-500}"
export PEAK_LR="${PEAK_LR:-1e-4}"
export LOG_EVERY="${LOG_EVERY:-50}"
export TOKENIZERS_PARALLELISM=false
export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:---xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true --xla_tpu_data_parallel_opt_different_sized_ops=true}"

cd "$(dirname "$0")/../.."
exec python3 scripts/examples/tpu_train_v4_32_3epoch.py 2>&1
