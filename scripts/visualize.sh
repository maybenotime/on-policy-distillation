cd ..

python source/visualize_token_rewards.py \
    --input  results/token_rewards_student3b.pkl \
    --output results/token_rewards_student3b_viz.html \
    --model  Qwen/Qwen2.5-3B-Instruct \
    --contrast_teachers Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-7B-Instruct \
    --sample_n 5 \
    --seed 42                