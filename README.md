# Quantformer

​	This is the official pytorch implementation for the paper: *Quantformer: Learning Extremely Low-precision Vision Transformers*. This repository contains differentiable searching and finetuning process for patch group assignment in group-wise quantization, and testing on ILSVRC2012 dataset using our proposed Quantformer.

## Quick Start

### Prerequisites

- python>=3.6
- pytorch>=1.6.0
- torchvision>=0.7.0 
- other packages like numpy and timm

### Dataset Preparation

Please follow the instruction in [this](https://github.com/zhirongw/lemniscate.pytorch) to download the ImageNet dataset.SSD is highly recommended for training on ImageNet.

### Pretrained Models

You can get full-precision pretrained models from [DeiT](https://github.com/facebookresearch/deit) and [Swin Transformer](https://github.com/microsoft/Swin-Transformer).

## Training and Testing

The following experiments were performed in TITAN Xp with 12GB memory.

### Pretraining

You can run the following command to get the quantized pretrained model in low bitwidths with shared quantization strategy(group number is 1).

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py --model deit_tiny_patch16_224 --fpmodel fpdeit_tiny_patch16_224 --min-lr 1e-6 --batch-size 64 --data-path <imagenet path> --finetune <full-precision pretrained model path> --output_dir <output path> --epoch 120 --lr 1e-4 --bit-width 4 --group-num 1
```

### Searching

After you get quantized pretrained model, you can run the following command to train the best patch group assignment using differentiable search.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py --model deit_tiny_patch16_224 --fpmodel fpdeit_tiny_patch16_224 --min-lr 1e-6 --batch-size 8 --data-path <imagenet path> --finetune <quantized pretrained model path> --fpfinetune <full-precision pretrained model path> --output_dir <output path> --epoch 1 --lr 2e-6 --bit-width 4 --aux-weight 20 --dm-weight 0.025 --group-num 8 --pnorm 2 --search True
```

### Finetuning

After searching, You can run the following command to finetune the model.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py --model deit_tiny_patch16_224 --fpmodel fpdeit_tiny_patch16_224 --min-lr 1e-6 --batch-size 32 --data-path <imagenet path> --finetune <quantized pretrained model path> --fpfinetune <full-precision pretrained model path> --output_dir <output path> --epoch 5 --lr 5e-6 --bit-width 4 --aux-weight 20 --dm-weight 0.025 --group-num 8 --pnorm 2 --search False
```

## Acknowledgements

We thank the authors of following works for opening source their excellent codes.

- [DeiT](https://github.com/facebookresearch/deit)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer)
- [HAQ](https://github.com/mit-han-lab/haq)
- [EdMIPS](https://github.com/zhaoweicai/EdMIPS)