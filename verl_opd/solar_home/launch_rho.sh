set -u
W=/home/sya225/opd/scripts/rho_worker.sh
V=/home/sya225/opd/verl_data/deepmath/verl_format
DS=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
Q3=Qwen/Qwen3-1.7B
JR=hbx/JustRL-DeepSeek-1.5B
M5=Keven16/Qwen3-4B-Non-Thinking-RL-Math-Step500
L=/home/sya225/logs; mkdir -p $L
launch () { # sess tag fam stu data gpu tname teacher
  tmux kill-session -t $1 2>/dev/null
  tmux new-session -d -s $1 "srun --jobid 270249 --overlap bash $W $2 $3 $4 $5 $6 $7 $8 7 8192 > $L/rho_$2.out 2>&1"
  echo "launched $1 -> $L/rho_$2.out"
}
launch rho_p1w p1w_JustRL ds      $DS $V/p1w/train.parquet 0 JustRL $JR
launch rho_p1c p1c_JustRL ds      $DS $V/p1c/train.parquet 1 JustRL $JR
launch rho_p2w p2w_M500   qwen3nt $Q3 $V/p2w/train.parquet 2 M500   $M5
launch rho_p2c p2c_M500   qwen3nt $Q3 $V/p2c/train.parquet 3 M500   $M5
sleep 5; tmux ls | grep rho_p
