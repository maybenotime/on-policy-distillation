# 22GB
export HUGGINGFACE_HUB_CACHE="/localscratch/sya225/.cache/model"
nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --use_hf true \
    --train_type full \
    --dataset 'modelscope/self-cognition#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --learning_rate 1e-4 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --deepspeed zero3 \
    --model_name swift-robot