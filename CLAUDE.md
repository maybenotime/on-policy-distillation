
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for **on-policy distillation** of LLMs on math reasoning tasks. Two distinct distillation approaches are implemented:

1. **GKD (Generalized Knowledge Distillation)**: Student generates rollouts on-policy; teacher scores them. The teacher is a separate, larger model (e.g., Qwen2.5-14B) accessed via a vLLM server.
2. **OPSD (On-Policy Self-Distillation)**: The *same* model weights serve as both student and teacher, but the teacher sees the problem plus a correct reference solution as privileged context. Controlled by `source/opsd_plugin.py` loaded as a ms-swift external plugin.

Both use [ms-swift](https://github.com/modelscope/ms-swift) (`swift rlhf --rlhf_type gkd`) for training and evaluation.

## Environment Setup

The repo runs on two SLURM clusters: **Fir** and **Solar**. Each has its own set of job scripts under `slurm_jobs/`, distinguished by file suffix (`_fir.sh` / `_solar.sh`).

| | Fir | Solar |
|---|---|---|
| GPU | H100 | L40S |
| Account | `rrg-msavva` | (none) |
| Repo path | `/home/sya225/scratch/on-policy-distillation` | `/home/sya225/on-policy-distillation` |
| Env setup | `module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv` | `source ~/.bashrc` |
| Virtualenv | `source test/bin/activate` | `source test_env/bin/activate` |
| `$SLURM_TMPDIR` | Set by SLURM | Falls back to `/localscratch/sya225` |

For local or interactive use:

```bash
module load StdEnv/2023 python/3.12 cuda/12.6 arrow/21.0.0 gcc opencv
source test_env/bin/activate    # virtualenv at repo root
```

Key dependencies (see `requirements.txt`): `ms-swift==4.0.0`, `vllm==0.15.1`, `torch==2.9.1`, `deepspeed==0.18.1`, `flash_attn==2.8.3`, `evalscope==1.4.2`.

## Key Commands

All scripts under `scripts/` should be run from inside `scripts/` (they use relative `..` paths), or the SLURM wrappers in `slurm_jobs/` handle `cd` automatically.

### GKD On-Policy Distillation (multi-GPU)

```bash
# Start rollout (vLLM) server on GPU 0
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 bash scripts/rollout.sh Qwen/Qwen2.5-3B-Instruct &

# Then launch training on remaining GPUs
CUDA_VISIBLE_DEVICES=1,2,3 NPROC_PER_NODE=3 bash scripts/on-policy-distill.sh \
    Qwen/Qwen2.5-3B-Instruct \
    Qwen/Qwen2.5-14B-Instruct \
    5000
```

Arguments: `$1` = student model, `$2` = teacher model, `$3` = dataset size.

### OPSD (On-Policy Self-Distillation)

```bash
# Default: single-teacher mode using deepseek-r1
bash scripts/opsd.sh

# Specify teacher model or use all-correct mode
OPSD_MODE=single OPSD_TEACHER_MODEL=Qwen2.5-72B-Instruct bash scripts/opsd.sh
OPSD_MODE=all_correct bash scripts/opsd.sh

# Data path (defaults to /localhome/sya225/Projects/Self-Distillation/data/final_math_dataset_updated.jsonl)
OPSD_DATA_PATH=/path/to/data.jsonl bash scripts/opsd.sh
```

### SFT Baseline

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 bash scripts/SFT.sh Qwen/Qwen2.5-0.5B-Instruct
```

Trains on `open-thoughts/OpenThoughts-114k#80000` with deepseek as the implicit teacher.

### Evaluation

```bash
# Evaluate a model on GSM8K, AIME24/25, AMC, MATH-500
bash scripts/eval_math_reasoning.sh Qwen/Qwen2.5-0.5B-Instruct
# or a checkpoint path
bash scripts/eval_math_reasoning.sh ../output/student_Qwen2.5-0.5B-Instruct_teacher_Qwen2.5-7B-Instruct_data5000/checkpoint-500
```

### Token-Level Reward Analysis

```bash
# Step 1: Generate rollouts and compute per-token rewards (output: .pkl)
bash scripts/on-policy-token-level-reward-analysis.sh

# Step 2: Print statistics (correct vs. incorrect rollouts)
bash scripts/calculate_reward_stats.sh

# Step 3: Render interactive HTML visualization
bash scripts/visualize.sh
```

### SLURM Job Submission

All SLURM scripts have two variants: `*_fir.sh` (H100, Fir cluster) and `*_solar.sh` (L40S, Solar cluster). Replace `<cluster>` below with `fir` or `solar`.

```bash
# Submit training + auto-chain eval
bash slurm_jobs/submit_train_plus_eval_<cluster>.sh

# Individual jobs
sbatch slurm_jobs/submit_on_policy_distillation_<cluster>.sh
sbatch slurm_jobs/submit_sft_<cluster>.sh
sbatch slurm_jobs/submit_token_reward_analysis_<cluster>.sh
```

## Architecture

### Data

- `data/prompts_math_swift.jsonl` — prompts in ms-swift format (used for GKD training).
- `data/prompts_math_with_answers_correct_100.jsonl` — 100 math prompts with ground-truth answers (used for reward analysis).
- OPSD training data (`final_math_dataset_updated.jsonl`) lives outside this repo at `/localhome/sya225/Projects/Self-Distillation/data/`.

### Source Modules (`source/`)

| File | Purpose |
|------|---------|
| `opsd_plugin.py` | ms-swift external plugin; registers `local-opsd-math` dataset and two preprocessors: `SingleTeacherPreprocessor` (one teacher model's correct answer) and `AllCorrectTeachersPreprocessor` (concatenates all correct answers). Configured via `OPSD_MODE` and `OPSD_TEACHER_MODEL` env vars. |
| `token_reward_analysis.py` | Loads student + teachers, generates student rollouts, computes per-token `log p_teacher − log p_student` rewards and student entropy. Saves results as `.pkl`. Requires student and teacher to share the same vocabulary. |
| `reward_stats.py` | Reads `.pkl` output; prints per-sample and group-mean stats (positive/negative reward %, mean, variance) split by correctness. |
| `visualize_token_rewards.py` | Renders a self-contained HTML file with color-coded token spans (reward variance = background color) and line charts per sample. |
| `math_evaluator.py` | `MathSolutionEvaluator` using `math_verify` library; extracts `\boxed{}` answers and verifies against ground truth. |

### Outputs

- `output/student_<STUDENT>_teacher_<TEACHER>_data<N>/` — training checkpoints
- `eval_output/<MODEL_TAG>/` — evaluation results
- `results/token_rewards_*.pkl` — reward analysis data
- `results/token_rewards_*_viz.html` — interactive HTML visualizations
- `checkpoint_path_<SLURM_JOB_ID>.txt` — checkpoint path written on job exit (consumed by chained eval job)

### Training Architecture (GKD)

The on-policy GKD setup runs two processes concurrently on one node:
- **GPU 0**: `swift rollout` — vLLM server serving the student model for on-policy generation (port 8000)
- **GPUs 1–3**: `swift rlhf --rlhf_type gkd` — training with DeepSpeed ZeRO-2 (student), no DeepSpeed for teacher

The OPSD setup uses a single process with `--vllm_mode colocate` and `--sleep_level 1` to share GPU memory between student training and vLLM inference.

### Known Issues & Fixes (GKD)

**Student must use `--deepspeed zero2`, NOT `zero3`.**
With ZeRO-3, student parameters are sharded across GPUs. Every `update_flattened_params` weight-sync (training → vLLM rollout server) requires a NCCL `GatheredParameters` collective that can corrupt the training-side parameter buffers, causing student weights to silently become NaN. With ZeRO-2, parameters are fully replicated so the sync is safe.

Teacher can be run without DeepSpeed (or with `zero3`) since it only does inference (`torch.no_grad()`) and never needs weight syncing.

**`logging_nan_inf_filter=True` masks NaN loss as `0.0`.**
If student weights go NaN, every subsequent step reports `loss=0.0` and `grad_norm=1.41421356` (√2, from weight decay only). This looks like training is running but nothing is being learned. Always check that loss varies across steps and grad_norm is not a fixed constant.
