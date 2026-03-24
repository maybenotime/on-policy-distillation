cd ..

python source/token_reward_analysis.py \
    --student_model Qwen/Qwen2.5-0.5B-Instruct \
    --teacher_models Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct \
    --prompts_file data/prompts_math_with_answers_5000.jsonl \
    --output_path results/token_rewards_student0.5b.pkl \
    --max_new_tokens 16384 \
    --temperature 1.0 \
    --batch_size 16