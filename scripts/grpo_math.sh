# GRPO on 10k math prompts with rule-based math reward (math_verify on \boxed{}).
#
# Reward is ms-swift's built-in `accuracy` ORM (MathAccuracy in
# swift/rewards/orm.py): parses the model's `\boxed{...}` from the completion
# and verifies against `solution` from the dataset.
#
# Args:
#   $1 = model               (default Qwen/Qwen2.5-3B-Instruct)
#   $2 = data size           (default 10000)
#   $3 = num_train_epochs    (default 1)
#
# Env (optional):
#   PER_DEVICE_BS            default 1 (1 prompt × num_generations=8 = 8 trajectories per process)
#   GRAD_ACCUM               default 2
#   NUM_GENERATIONS          default 8 (G in GRPO paper)
#   GRPO_BETA                default 0.04 (KL penalty vs ref model)
#   GRPO_TEMPERATURE         default 1.0
#   GRPO_LR                  default 1e-6
#   VLLM_GPU_MEM_UTIL        default 0.4 (lower than GKD because GRPO holds both train and ref params)
#   MAX_COMPLETION_LENGTH    default 4096 (set to 8192 for larger models)
#   MAX_LENGTH               default 2*MAX_COMPLETION_LENGTH (prompt + completion)
#   RESUME_FROM              checkpoint dir to resume from (default unset)

set -e

MODEL="${1:-Qwen/Qwen2.5-3B-Instruct}"
DATA_SIZE="${2:-10000}"
NUM_EPOCHS="${3:-1}"

MODEL_TAG="${MODEL##*/}"

CACHE_BASE="${CACHE_BASE:-/home/sya225/.cache}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-/localscratch/sya225/output}"

export HF_HOME="${HF_HOME:-$CACHE_BASE}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$CACHE_BASE/modelscope}"
export TORCH_HOME="${TORCH_HOME:-$CACHE_BASE/torch}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$CACHE_BASE/vllm}"
export TMPDIR="${TMPDIR:-$CACHE_BASE/tmp}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE" "$TORCH_HOME" "$VLLM_CACHE_ROOT" "$TMPDIR"

export WANDB_API_KEY="${WANDB_API_KEY:-8b34bbf9238dfc1c55c378dc0dea31921d5f420b}"
export WANDB_PROJECT="${WANDB_PROJECT:-grpo_math}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default to the 10k-row GRPO file built by scripts/prep_grpo_data.py
DATASET_SPEC="${GRPO_DATASET:-${REPO_DIR}/data/prompts_math_grpo_10k.jsonl#${DATA_SIZE}}"

PER_DEVICE_BS="${PER_DEVICE_BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
GRPO_BETA="${GRPO_BETA:-0.04}"
GRPO_TEMPERATURE="${GRPO_TEMPERATURE:-1.0}"
GRPO_LR="${GRPO_LR:-1e-6}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.4}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4096}"
MAX_LENGTH="${MAX_LENGTH:-$((MAX_COMPLETION_LENGTH * 2))}"

OUTPUT_DIR="${OUTPUT_BASE_DIR}/grpo_${MODEL_TAG}_epochs${NUM_EPOCHS}_data${DATA_SIZE}"

RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then RESUME_ARG="--resume_from_checkpoint $RESUME_FROM"; fi

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
swift rlhf \
    --rlhf_type grpo \
    --reward_funcs accuracy \
    --num_generations "$NUM_GENERATIONS" \
    --beta "$GRPO_BETA" \
    --temperature "$GRPO_TEMPERATURE" \
    --use_hf true \
    --model "$MODEL" \
    --train_type full \
    --dataset "$DATASET_SPEC" \
    --torch_dtype bfloat16 \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_BS" \
    --learning_rate "$GRPO_LR" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --save_strategy epoch \
    --save_steps 1 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length "$MAX_LENGTH" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --report_to wandb \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio 0.05 \
    --save_only_model false \
    --dataloader_num_workers 16 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --gradient_checkpointing true \
    --attn_impl flash_attn \
    --use_vllm true \
    --vllm_mode colocate \
    --sleep_level 1 \
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEM_UTIL" \
    --log_completions true \
    $RESUME_ARG

CHECKPOINT=$(find "$OUTPUT_DIR" -name "checkpoint-*" -type d | sort -V | tail -1)
echo "$CHECKPOINT" > ../checkpoint_path_${SLURM_JOB_ID:-local}.txt
echo "Checkpoint saved: $CHECKPOINT"
