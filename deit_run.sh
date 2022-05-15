#!/usr/bin/env bash

PORT=${PORT:-39500}

CUDA_VISIBLE_DEVICES=6 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=$PORT \
    $(dirname "$0")/main.py --model deit_tiny_patch16_224 --fpmodel fpdeit_tiny_patch16_224 --batch-size 32 \
    --min-lr 1e-6 --data-path /home/wzw/SSD/ILSVRC2012/ --warmup-epochs 0 \
    --finetune /home/wzw/wcy/deit-result/4bit-p1/checkpoint.pth \
    --fpfinetune /home/wzw/wcy/deit-result/32bit/deit_tiny_patch16_224-a1311bcf.pth \
    --epoch 5 --lr 20e-6 --bit-width 4 --aux-weight 20 --dm-weight 0.025 --group-num 2 --pnorm 2 --search False
