# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
from custom_timm.loss import SoftTargetCrossEntropy
from custom_timm.models.swin_transformer import QWindowAttention, WindowAttention
import torch.nn.functional as F
import utils
from lib.utils.quantize_utils import QLinear

def cal_entropy(attn):
    return -1 * torch.sum((attn * torch.log(attn)), dim=-1).mean()

def cal_l2loss(x, y):
    return (F.normalize(x.view(x.size(0), -1)) - F.normalize(y.view(y.size(0), -1))).pow(2).mean()

def train_one_epoch(args,
                    model: torch.nn.Module, fpmodel: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, arch_optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, 
                    aux_weight: float = 0.5, dm_weight: float = 0.025,
                    pnorm: int = 3, set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        aux_loss = 0
        dm_loss = 0
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        if args.group_num > 1:
            fpmodel.eval()
            with torch.cuda.amp.autocast():
                fpoutputs = fpmodel(samples)

            attn = []
            for i, layer in enumerate(model.modules()):
                if type(layer) in [QWindowAttention]:
                    attn.append(layer.attent)
            j = 0
            for i, layer in enumerate(fpmodel.modules()):
                if type(layer) in [WindowAttention]:
                    fpattn = layer.attent.detach()
                    fpattn = torch.pow(fpattn, pnorm)
                    aux_loss += cal_l2loss(fpattn, attn[j])
                    j = j + 1
            
            if args.search == True:
                for i, layer in enumerate(model.modules()):
                    if type(layer) in [QLinear]:
                        alpha = layer.sw
                        group_n, dim = alpha.shape
                        dm_loss_t = 0
                        for k in range(group_n):
                            dm_loss_t += cal_entropy(alpha[k])
                        dm_loss += dm_loss_t / (group_n * dim)
        
        loss = loss + dm_weight * dm_loss + aux_weight * aux_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        optimizer.zero_grad()
        arch_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        arch_optimizer.step()
        
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = SoftTargetCrossEntropy()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
