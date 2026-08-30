#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"
GPU_IDS="${GPU_IDS:-0,1}"
SFT_MODEL="${SFT_MODEL:-$PROJECT_ROOT/outputs/sft}"
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/data/rl_data.jsonl}"
REWARD_PATH="${REWARD_PATH:-$PROJECT_ROOT/opsd/reward.py}"
RUN_NAME="${RUN_NAME:-faro2_qwen3_0p6b_s42}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/$RUN_NAME}"
CKPT_DIR="${CKPT_DIR:-$OUTPUT_DIR/trainer_state}"
SEED="${SEED:-42}"

NUM_EPISODES="${NUM_EPISODES:-3}"
MAX_SAMPLES="${MAX_SAMPLES:-100000000}"
VLLM_TARGET_MIB="${VLLM_TARGET_MIB:-40960}"
A800_CARD_LIMIT_MIB="${A800_CARD_LIMIT_MIB:-73728}"
MONITOR_INTERVAL_SEC="${MONITOR_INTERVAL_SEC:-5}"

if [[ "$(awk -F, '{print NF}' <<<"$GPU_IDS")" -ne 2 ]]; then
  echo "GPU_IDS must contain exactly two GPU ids, for example GPU_IDS=0,1" >&2
  exit 2
fi
for required in "$VENV_DIR/bin/activate" "$SFT_MODEL/model.safetensors" "$DATA_PATH" "$REWARD_PATH"; do
  [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 3; }
done

mapfile -t GPU_TOTALS < <(
  nvidia-smi -i "$GPU_IDS" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' '
)
[[ "${#GPU_TOTALS[@]}" -eq 2 ]] || { echo "Could not inspect both GPUs." >&2; exit 3; }
MIN_TOTAL_MIB="${GPU_TOTALS[0]}"
if (( GPU_TOTALS[1] < MIN_TOTAL_MIB )); then MIN_TOTAL_MIB="${GPU_TOTALS[1]}"; fi
if (( VLLM_TARGET_MIB >= MIN_TOTAL_MIB )); then
  echo "The requested 40 GiB vLLM pool requires two GPUs with more than 40 GiB each." >&2
  exit 3
fi
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-$(
  awk -v target="$VLLM_TARGET_MIB" -v total="$MIN_TOTAL_MIB" 'BEGIN {printf "%.6f", target / total}'
)}"

source "$VENV_DIR/bin/activate"
mkdir -p "$OUTPUT_DIR" "$CKPT_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NCCL_CUMEM_ENABLE=0
export WANDB_MODE=offline
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/opsd_${RUN_NAME}_$SEED}"
unset RAY_ADDRESS || true

load_args=()
if find "$CKPT_DIR" -mindepth 1 -print -quit | grep -q .; then
  load_args+=(--load_checkpoint)
fi

TRACE_FILE="$OUTPUT_DIR/resource_trace.csv"
printf 'timestamp,gpu_index,memory_used_mib,memory_free_mib,memory_total_mib,utilization_gpu_pct\n' > "$TRACE_FILE"
(
  while true; do
    ts="$(date --iso-8601=seconds)"
    nvidia-smi -i "$GPU_IDS" \
      --query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu \
      --format=csv,noheader,nounits \
      | awk -v ts="$ts" -F, '{gsub(/[[:space:]]/, ""); print ts "," $1 "," $2 "," $3 "," $4 "," $5}' \
      >> "$TRACE_FILE" || true
    sleep "$MONITOR_INTERVAL_SEC"
  done
) &
MONITOR_PID=$!

stop_monitor() {
  if kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}
trap stop_monitor EXIT

cat > "$OUTPUT_DIR/run_config.txt" <<EOF
stage=faro2-opsd
pretrain=$SFT_MODEL
prompt_data=$DATA_PATH
external_teacher=none
frozen_reference=same_as_pretrain
gpu_ids=$GPU_IDS
seed=$SEED
num_episodes=$NUM_EPISODES
vllm_target_mib=$VLLM_TARGET_MIB
vllm_gpu_memory_utilization=$VLLM_GPU_MEMORY_UTILIZATION
a800_card_limit_mib=$A800_CARD_LIMIT_MIB
EOF
printf 'RUNNING stage=faro2-opsd gpu_ids=%s time=%s\n' "$GPU_IDS" "$(date --iso-8601=seconds)" > "$OUTPUT_DIR/status.txt"

set +e
python -m openrlhf.cli.train_ppo_ray \
  --colocate_actor_ref \
  --ref_num_nodes 1 --ref_num_gpus_per_node 1 \
  --reward_num_nodes 0 --reward_num_gpus_per_node 0 \
  --actor_num_nodes 1 --actor_num_gpus_per_node 1 \
  --vllm_num_engines 1 --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --vllm_sync_backend nccl --vllm_sync_with_ray \
  --enable_prefix_caching --enforce_eager \
  --pretrain "$SFT_MODEL" \
  --remote_rm_url "$REWARD_PATH" \
  --prompt_data "$DATA_PATH" \
  --input_key inputs --label_key targets \
  --save_path "$OUTPUT_DIR" \
  --ckpt_path "$CKPT_DIR" \
  "${load_args[@]}" \
  --save_steps 100 --save_hf_ckpt --max_ckpt_num 3 \
  --micro_train_batch_size 1 --train_batch_size 128 \
  --micro_rollout_batch_size 2 --rollout_batch_size 128 \
  --n_samples_per_prompt 8 \
  --max_epochs 1 --num_episodes "$NUM_EPISODES" --max_samples "$MAX_SAMPLES" \
  --prompt_max_len 5120 --generate_max_len 512 \
  --advantage_estimator group_norm \
  --normalize_reward \
  --dynamic_filtering --dynamic_filtering_reward_range 0.01 0.99 \
  --init_kl_coef 1e-3 --use_kl_loss --kl_estimator k2 \
  --zero_stage 1 --bf16 --flash_attn --packing_samples --gradient_checkpointing \
  --actor_learning_rate 3e-7 \
  --seed "$SEED" \
  --use_tensorboard "$OUTPUT_DIR/tensorboard" \
  2>&1 | tee -a "$OUTPUT_DIR/train.log"
rc=${PIPESTATUS[0]}
set -e

stop_monitor
trap - EXIT

PEAK_FILE="$OUTPUT_DIR/resource_peak.tsv"
printf 'gpu_index\tpeak_used_mib\tmin_free_mib\tmemory_total_mib\n' > "$PEAK_FILE"
awk -F, 'NR > 1 {
  gpu=$2; used=$3+0; free=$4+0; total=$5+0;
  if (!(gpu in peak) || used > peak[gpu]) peak[gpu]=used;
  if (!(gpu in minfree) || free < minfree[gpu]) minfree[gpu]=free;
  totals[gpu]=total;
} END {
  for (gpu in peak) printf "%s\t%d\t%d\t%d\n", gpu, peak[gpu], minfree[gpu], totals[gpu];
}' "$TRACE_FILE" | sort -n >> "$PEAK_FILE"

if (( rc != 0 )); then
  printf 'FAILED stage=faro2-opsd rc=%s time=%s\n' "$rc" "$(date --iso-8601=seconds)" > "$OUTPUT_DIR/status.txt"
  exit "$rc"
fi

[[ -s "$OUTPUT_DIR/model.safetensors" ]] || {
  printf 'FAILED stage=faro2-opsd missing_model time=%s\n' "$(date --iso-8601=seconds)" > "$OUTPUT_DIR/status.txt"
  exit 4
}

if awk -F'\t' -v limit="$A800_CARD_LIMIT_MIB" 'NR > 1 && $2 > limit {bad=1} END {exit bad ? 0 : 1}' "$PEAK_FILE"; then
  printf 'RESOURCE_BUDGET_EXCEEDED limit_mib=%s time=%s\n' "$A800_CARD_LIMIT_MIB" "$(date --iso-8601=seconds)" > "$OUTPUT_DIR/status.txt"
  exit 5
fi

printf 'COMPLETE_A800_FIT stage=faro2-opsd gpu_ids=%s time=%s\n' "$GPU_IDS" "$(date --iso-8601=seconds)" > "$OUTPUT_DIR/status.txt"
