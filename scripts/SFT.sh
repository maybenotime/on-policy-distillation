STUDENT_MODEL="${1:-Qwen/Qwen2.5-0.5B-Instruct}"
STUDENT_MODEL_TAG="${STUDENT_MODEL##*/}"

HF_TOKEN="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"
huggingface-cli login --token $HF_TOKEN

export WANDB_API_KEY="8b34bbf9238dfc1c55c378dc0dea31921d5f420b"
export WANDB_PROJECT="SFT"


swift sft \
  --model "$STUDENT_MODEL" \
  --use-hf true \
  --tuner_type "full" \
  --dataset "open-thoughts/OpenThoughts-114k#10000" \
  --torch_dtype "bfloat16" \
  --num_train_epochs "1" \
  --per_device_train_batch_size "1" \
  --learning_rate "1e-5" \
  --gradient_accumulation_steps "8" \
  --deepspeed zero3 \
  --attn_impl flash_attn \
  --padding_free True \
  --report_to wandb \
  --output_dir "../output/student_${STUDENT_MODEL_TAG}_teacher_deepseek" \
  --save_strategy "epoch" \
  --save_total_limit "2" \
  --logging_steps "10" \
  --max_length "16384" \
  --per_device_eval_batch_size "30" \
  --eval_use_evalscope \
  --eval_dataset "gsm8k" "aime24" "aime25" "amc" "math_500" \
