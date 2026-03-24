cd ..

python source/visualize_token_rewards.py \
    --input  results/token_rewards_test.pkl \
    --output results/token_rewards_viz.html \
    --model  Qwen/Qwen2.5-3B-Instruct \
    --contrast_teachers Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-7B-Instruct \
    --sample_n 10 \
    --seed 42                