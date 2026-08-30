# Full Agent RL with OPSD on Just 2 A800s

This repository contains only the two training stages needed to reproduce the compact pipeline:

1. supervised fine-tuning on `kd_data.jsonl`;
2. teacher-free OPSD on `rl_data.jsonl`, using the SFT checkpoint as both the trainable policy initialization and the frozen reference.

The OPSD launcher uses exactly two GPUs: actor and frozen reference share one GPU, while a single vLLM engine receives an absolute 40 GiB memory pool on the other GPU. Two 80GB A800s or H800s are sufficient. Two 40GB cards are not supported by the default full-shape configuration.

## Repository contents

```text
opsd/reward.py              function-calling reward
scripts/setup_env.sh        isolated Python environment
scripts/train_sft.sh        SFT stage
scripts/train_opsd_2xa800.sh  two-GPU OPSD stage
agent.md                    server-agent runbook
```

The repository is deliberately limited to the two training stages.

## Install

Linux:

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

The pinned stack uses OpenRLHF 0.8.9, vLLM 0.10.0, PyTorch 2.7.1, CUDA 12.8 wheels, BF16, FlashAttention, Ray, and DeepSpeed.

## Download the released data

The complete internally collected corpus is not published. The Hugging Face dataset contains an exact, reproducible 40% random sample of each training file.

Install the Hugging Face CLI on PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

Or on Linux:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

Download the public subset:

```bash
hf download Camellia86/Full_Agent_RL_OPSD_with_Just_2_A800s \
  --repo-type dataset \
  --local-dir data
```

Expected files:

```text
data/kd_data.jsonl
data/rl_data.jsonl
```

<!-- DATASET_STATS_START -->
| File |Rows |
|---|---:|
| `kd_data.jsonl` | 429,218 | 
| `rl_data.jsonl` | 57,796 |

<!-- DATASET_STATS_END -->

## Stage 1: SFT

```bash
GPU_IDS=0,1 bash scripts/train_sft.sh
```

Useful overrides:

```bash
GPU_IDS=0,1 \
BASE_MODEL=/path/to/Qwen3-0.6B \
DATA_PATH=data/kd_data.jsonl \
OUTPUT_DIR=outputs/sft \
MAX_EPOCHS=1 \
bash scripts/train_sft.sh
```

The completed model is written to `outputs/sft/model.safetensors`.

## Stage 2: OPSD

Start OPSD only after SFT has completed:

```bash
GPU_IDS=0,1 \
SFT_MODEL=outputs/sft \
bash scripts/train_opsd_2xa800.sh
```

The default full shape uses 5120 prompt tokens, 512 generated tokens, 8 rollouts per prompt, global rollout/train batch 128, dynamic reward filtering, group-normalized on-policy updates, and a frozen-reference KL term. No external teacher model is loaded.

The launcher records `resource_trace.csv` and `resource_peak.tsv`. A run is marked `COMPLETE_A800_FIT` only when the trained model exists and neither card exceeds the default 72 GiB observed-memory ceiling.

## Detached launch

```bash
tmux new-session -d -s opsd-sft \
  "cd $PWD && GPU_IDS=0,1 bash scripts/train_sft.sh"

tmux new-session -d -s opsd-rl \
  "cd $PWD && GPU_IDS=0,1 SFT_MODEL=outputs/sft bash scripts/train_opsd_2xa800.sh"
```

Do not start the second command until `outputs/sft/status.txt` begins with `COMPLETE`.
