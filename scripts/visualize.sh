cd ..

python source/visualize_token_rewards.py \
    --input  results/token_rewards_test.pkl \
    --output results/token_rewards_viz.html \
    --model  Qwen/Qwen2.5-3B-Instruct \
    --sample_n 10 \
    --seed 42                