cd ..

python source/token_reward_analysis.py \
    --student_model Qwen/Qwen2.5-3B-Instruct \
    --teacher_models Qwen/Qwen2.5-7B-Instruct \
    --prompts_file data/prompts_math_with_answers_correct_100.jsonl \
    --output_path results/token_rewards_student3b.pkl \
    --max_new_tokens 8192 \
    --temperature 1.0 \
    --batch_size 2