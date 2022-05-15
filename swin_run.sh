#!/usr/bin/env bash

PORT=${PORT:-29500}

CUDA_VISIBLE_DEVICES=1 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=$PORT \
    $(dirname "$0")/main.py --model swin_small_patch4_window7_224 --fpmodel fpswin_small_patch4_window7_224 --batch-size 2 \
    --min-lr 1e-6 --data-path /home/wzw/SSD/ILSVRC2012/ --warmup-epochs 0 \
    --finetune /home/wzw/wcy/deit-result/4bit-swin-supernet/checkpoint.pth \
    --fpfinetune /home/wzw/wcy/deit-result/swin-baseline/swin_small_patch4_window7_224.pth \
    --epoch 5 --lr 20e-6 --bit-width 4 --aux-weight 20 --dm-weight 0.025 --group-num 8 --pnorm 2 --search True
