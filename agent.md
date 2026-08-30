# Server agent runbook

This repository is training-only. Keep the working tree limited to the SFT stage, the teacher-free OPSD stage, their reward function, and launch documentation.

## Hardware admission

- Run training only on a GPU host.
- Reserve exactly two cards with at least 80GB each. Supported targets include A800 80GB, H800 80GB, and H200.
- Check both free memory and utilization before launch. Never terminate, pause, or migrate another user's process.
- Do not use the default full-shape OPSD command on 40GB cards.

## Server layout

Recommended checkout:

```text
/volume/wzhang/nzx/Full_Agent_RL_OPSD_with_Just_2_A800s
```

Perform downloads and environment setup on an internet-enabled CPU host. Run SFT and OPSD on the GPU host. The checkout, model cache, data, and outputs must be on storage shared by those hosts.

## One-time setup

```bash
cd /volume/wzhang/nzx/Full_Agent_RL_OPSD_with_Just_2_A800s
bash scripts/setup_env.sh
source .venv/bin/activate

hf download Camellia86/Full_Agent_RL_OPSD_with_Just_2_A800s \
  --repo-type dataset \
  --local-dir data
```

If the GPU host has no internet, download `Qwen/Qwen3-0.6B` on the CPU host and pass its shared local path through `BASE_MODEL`.

## Start SFT

Choose two safe physical GPU ids, then launch in tmux:

```bash
tmux new-session -d -s opsd-sft \
  "cd /volume/wzhang/nzx/Full_Agent_RL_OPSD_with_Just_2_A800s && \
   GPU_IDS=0,1 BASE_MODEL=/shared/path/Qwen3-0.6B bash scripts/train_sft.sh"
```

The required completion sentinel is:

```text
outputs/sft/status.txt -> COMPLETE stage=sft
```

Do not start OPSD until `outputs/sft/model.safetensors` exists and the SFT status is complete.

## Start OPSD

Re-check the same two cards immediately before launch. OPSD uses no external teacher.

```bash
tmux new-session -d -s opsd-rl \
  "cd /volume/wzhang/nzx/Full_Agent_RL_OPSD_with_Just_2_A800s && \
   GPU_IDS=0,1 SFT_MODEL=outputs/sft bash scripts/train_opsd_2xa800.sh"
```

Handoff locations:

```text
outputs/opsd_qwen3_0p6b_s42/status.txt
outputs/opsd_qwen3_0p6b_s42/train.log
outputs/opsd_qwen3_0p6b_s42/resource_peak.tsv
outputs/opsd_qwen3_0p6b_s42/model.safetensors
```

The successful resource sentinel is `COMPLETE_A800_FIT`. If the status is `RESOURCE_BUDGET_EXCEEDED`, preserve the logs and reduce only micro-batch sizes or context limits; do not silently change the algorithm or add another model.
