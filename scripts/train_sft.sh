#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
GPU_IDS="${GPU_IDS:-0,1}"
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/data/kd_data.jsonl}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/sft}"
CKPT_DIR="${CKPT_DIR:-$OUTPUT_DIR/trainer_state}"
MAX_EPOCHS="${MAX_EPOCHS:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-100000000}"
SEED="${SEED:-42}"

if [[ "$(awk -F, '{print NF}' <<<"$GPU_IDS")" -ne 2 ]]; then
  echo "GPU_IDS must contain exactly two GPU ids, for example GPU_IDS=0,1" >&2
  exit 2
fi
[[ -f "$VENV_DIR/bin/activate" ]] || { echo "Run scripts/setup_env.sh first." >&2; exit 3; }
[[ -f "$DATA_PATH" ]] || { echo "Missing SFT data: $DATA_PATH" >&2; exit 3; }

source "$VENV_DIR/bin/activate"
mkdir -p "$OUTPUT_DIR" "$CKPT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline

load_args=()
if find "$CKPT_DIR" -mindepth 1 -print -quit | grep -q .; then
  load_args+=(--load_checkpoint)
fi

printf 'RUNNING stage=sft gpu_ids=%s time=%s\n' "$GPU_IDS" "$(date --iso-8601=seconds)" > "$OUTPUT_DIR/status.txt"

set +e
deepspeed --num_gpus 2 --module openrlhf.cli.train_sft \
  --pretrain "$BASE_MODEL" \
  --dataset "$DATA_PATH" \
  --input_key inputs \
  --output_key targets \
  --input_template '{}' \
  --save_path "$OUTPUT_DIR" \
  --ckpt_path "$CKPT_DIR" \
  "${load_args[@]}" \
  --max_samples "$MAX_SAMPLES" \
  --max_epochs "$MAX_EPOCHS" \
  --max_len 5120 \
  --micro_train_batch_size 2 \
  --train_batch_size 128 \
  --zero_stage 2 \
  --bf16 \
  --flash_attn \
  --packing_samples \
  --gradient_checkpointing \
  --learning_rate 2e-5 \
  --lr_scheduler cosine_with_min_lr \
  --lr_warmup_ratio 0.03 \
  --l2 1e-4 \
  --logging_steps 1 \
  --seed "$SEED" \
  --use_tensorboard "$OUTPUT_DIR/tensorboard" \
  2>&1 | tee -a "$OUTPUT_DIR/train.log"
rc=${PIPESTATUS[0]}
set -e

if (( rc != 0 )); then
  printf 'FAILED stage=sft rc=%s time=%s\n' "$rc" "$(date --iso-8601=seconds)" > "$OUTPUT_DIR/status.txt"
  exit "$rc"
fi

[[ -s "$OUTPUT_DIR/model.safetensors" ]] || {
  printf 'FAILED stage=sft missing_model time=%s\n' "$(date --iso-8601=seconds)" > "$OUTPUT_DIR/status.txt"
  exit 4
}

printf 'COMPLETE stage=sft gpu_ids=%s time=%s\n' "$GPU_IDS" "$(date --iso-8601=seconds)" > "$OUTPUT_DIR/status.txt"
