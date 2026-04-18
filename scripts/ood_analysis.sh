cd ..

STUDENT_MODEL=${1:-"Qwen/Qwen2.5-3B-Instruct"}
TEACHER_MODEL=${2:-"Qwen/Qwen2.5-7B-Instruct"}
NUM_ROLLOUTS=${3:-10}

STUDENT_TAG=$(basename "$STUDENT_MODEL")
TEACHER_TAG=$(basename "$TEACHER_MODEL")

python source/ood_analysis.py \
    --student_model "$STUDENT_MODEL" \
    --teacher_model "$TEACHER_MODEL" \
    --prompts_file data/prompts_math_with_answers_correct_100.jsonl \
    --output_path "results/ood_analysis_${STUDENT_TAG}_${TEACHER_TAG}.pkl" \
    --max_new_tokens 8192 \
    --num_rollouts "$NUM_ROLLOUTS" \
    --num_sample_positions 10 \
    --temperature 1.0 \
    --batch_size 2 \
    --use_vllm
