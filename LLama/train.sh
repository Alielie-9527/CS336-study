#!/bin/bash

# 确保脚本抛出遇到的错误
set -e

# 设置单卡 GPU (例如使用 0 号卡)
export CUDA_VISIBLE_DEVICES=0

echo "Starting training on single GPU..."

# 运行训练
# 注意：请在项目根目录下运行此脚本，例如: bash LLama/train.sh

python LLama/train_model.py \
    --out_dir "base_model_215M" \
    --epochs 3 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --device "cuda:0" \
    --dtype "float16" \
    --num_workers 8 \
    --data_path "./LLama/seq_monkey_datawhale.jsonl" \
    --accumulation_steps 8 \
    --grad_clip 1.0 \
    --warmup_iters 1000 \
    --log_interval 10 \
    --save_interval 500

echo "Training finished."


# sft 训练示例

# python LLama/train_model.py \
#     --out_dir "sft_model_215M" \
#     --epochs 3 \
#     --batch_size 32 \
#     --learning_rate 3e-4 \
#     --device "cuda:0" \
#     --dtype "float16" \
#     --num_workers 8 \
#     --data_path "./LLama/BellGroup_sft.jsonl" \
#     --accumulation_steps 8 \
#     --grad_clip 1.0 \
#     --warmup_iters 1000 \
#     --log_interval 10 \
#     --save_interval 500

# echo "Training finished."