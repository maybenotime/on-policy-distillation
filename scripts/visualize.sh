cd ..

python source/visualize_token_rewards.py \
    --input  results/token_rewards_test.pkl \
    --output results/token_rewards_viz.html \
    --model  Qwen/Qwen2.5-0.5B-Instruct \
    --max_samples 10                    