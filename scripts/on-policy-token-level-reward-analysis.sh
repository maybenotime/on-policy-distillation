cd ..

python source/token_reward_analysis.py \
    --student_model Qwen/Qwen2.5-0.5B-Instruct \
    --teacher_models Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct \
    --prompts_file data/prompts_math_with_answers_5.jsonl \
    --output_path results/token_rewards_test.pkl \
    --max_new_tokens 1024 \
    --temperature 1.0 \
    --batch_size 2 