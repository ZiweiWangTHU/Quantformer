# Code for "[HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
# Kuan Wang*, Zhijian Liu*, Yujun Lin*, Ji Lin, Song Han
# {kuanwang, zhijian, yujunlin, jilin, songhan}@mit.edu

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _single, _pair, _triple

class QModule1(nn.Module):
    def __init__(self, a_bit=8, half_wave='Q'):
        super(QModule1, self).__init__()

        if half_wave == 'A' or half_wave == 'Q':
            self._a_bit = a_bit
        else:
            self._a_bit = a_bit - 1
        self._b_bit = 32
        self._half_wave = half_wave
        # self._half_wave = False

        self.init_range = 8.
        if half_wave == 'A':
            self.activation_range = nn.Parameter(torch.Tensor([self.init_range]))
        elif half_wave == 'Q':
            self.cw_1 = nn.Parameter(torch.Tensor([0.0]))
            self.dw_1 = nn.Parameter(torch.Tensor([self.init_range]))
            
        self.weight_range = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)
        self._quantized = True
        self._tanh_weight = False
        self._fix_weight = False
        self._trainable_activation_range = True
        self._calibrate = False

    @property
    def a_bit(self):
        if self._half_wave == 'A' or self._half_wave == 'Q':
            return self._a_bit
        else:
            return self._a_bit + 1

    @a_bit.setter
    def a_bit(self, a_bit):
        if self._half_wave == 'A' or self._half_wave == 'Q':
            self._a_bit = a_bit
        else:
            self._a_bit = a_bit - 1

    @property
    def b_bit(self):
        return self._b_bit

    @property
    def half_wave(self):
        return self._half_wave

    @property
    def quantized(self):
        return self._quantized

    @property
    def tanh_weight(self):
        return self._tanh_weight

    def set_quantize(self, quantized):
        self._quantized = quantized

    def set_tanh_weight(self, tanh_weight):
        self._tanh_weight = tanh_weight
        if self._tanh_weight:
            self.weight_range.data[0] = 1.0

    def set_fix_weight(self, fix_weight):
        self._fix_weight = fix_weight

    def set_activation_range(self, activation_range):
        self.activation_range.data[0] = activation_range

    def set_weight_range(self, weight_range):
        self.weight_range.data[0] = weight_range

    def set_trainable_activation_range(self, trainable_activation_range=True):
        self._trainable_activation_range = trainable_activation_range
        self.activation_range.requires_grad_(trainable_activation_range)

    def set_calibrate(self, calibrate=True):
        self._calibrate = calibrate

    def set_tanh(self, tanh=True):
        self._tanh_weight = tanh

    def _compute_threshold(self, data, bitwidth):
        mn = 0
        mx = np.abs(data).max()
        if np.isclose(mx, 0.0):
            return 0.0
        data_hist = np.abs(data).astype(np.float32)
        bins = np.histogram_bin_edges(data_hist, bins='sqrt')
        hist, bin_edges = np.histogram(data_hist, bins=bins, range=(mn, mx), density=True)
        #hist, bin_edges = np.histogram(np.abs(data), bins='sqrt', range=(mn, mx), density=True)
        hist = hist / np.sum(hist)
        cumsum = np.cumsum(hist)
        n = pow(2, int(bitwidth) - 1)
        threshold = []
        scaling_factor = []
        d = []
        if n + 1 > len(bin_edges) - 1:
            th_layer_out = bin_edges[-1]
            # sf_layer_out = th_layer_out / (pow(2, bitwidth - 1) - 1)
            return float(th_layer_out)
        for i in range(n + 1, len(bin_edges), 1):
            threshold_tmp = (i + 0.5) * (bin_edges[1] - bin_edges[0])
            threshold = np.concatenate((threshold, [threshold_tmp]))
            if bitwidth > 1:
                scaling_factor_tmp = threshold_tmp / (pow(2, bitwidth - 1) - 1)
            else:
                scaling_factor_tmp = threshold_tmp
                #scaling_factor_tmp = np.mean(np.abs(data))
            scaling_factor = np.concatenate((scaling_factor, [scaling_factor_tmp]))
            p = np.copy(cumsum)
            p[(i - 1):] = 1
            x = np.linspace(0.0, 1.0, n)
            xp = np.linspace(0.0, 1.0, i)
            fp = p[:i]
            p_interp = np.interp(x, xp, fp)
            x = np.linspace(0.0, 1.0, i)
            xp = np.linspace(0.0, 1.0, n)
            fp = p_interp
            q_interp = np.interp(x, xp, fp)
            q = np.copy(p)
            q[:i] = q_interp
            d_tmp = np.sum((cumsum - q) * np.log2(cumsum / q))  # Kullback-Leibler-J
            d = np.concatenate((d, [d_tmp]))

        th_layer_out = threshold[np.argmin(d)]
        # sf_layer_out = scaling_factor[np.argmin(d)]
        threshold = float(th_layer_out)
        return threshold

    def _quantize_activation(self, inputs):
        #self._calibrate = True
        if self._quantized and self._a_bit > 0:
            #print('a_bit:', self._a_bit)
            if self._calibrate:
                if self._a_bit < 5:
                    # threshold = self._compute_threshold(inputs.data.cpu().numpy(), self._a_bit)
                    # estimate_activation_range = min(min(self.init_range, inputs.abs().max().item()), threshold)
                    estimate_activation_range = min(self.init_range, inputs.abs().max().item())
                else:
                    estimate_activation_range = min(self.init_range, inputs.abs().max().item())
                # print(str(ab)+'a+threshold:', estimate_activation_range, inputs.abs().max().item(), find_scale_by_percentile(inputs))
                # print('range:', estimate_activation_range, '  shape:', inputs.shape, '  inp_abs_max:', inputs.abs().max())
                if self._half_wave == 'A':
                    self.activation_range.data = torch.tensor([estimate_activation_range], device=inputs.device)
                elif self._half_wave == 'Q':
                    self.dw_1.data = torch.tensor([estimate_activation_range], device=inputs.device)
                return inputs
            if self._trainable_activation_range:
                if self._half_wave == 'A':
                    ori_x = 0.5 * (inputs.abs() - (inputs - self.activation_range).abs() + self.activation_range)
                    # print(self.activation_range)
                elif self._half_wave == 'Q':
                    ori_x = 0.5 * ((-inputs + self.cw_1 - self.dw_1).abs() - (inputs - (self.cw_1 + self.dw_1)).abs() + 2 * self.cw_1)
            else:
                if self._half_wave:
                    ori_x = inputs.clamp(0.0, self.activation_range.item())
                else:
                    ori_x = inputs.clamp(-self.activation_range.item(), self.activation_range.item())
            # print(self._a_bit)
            # print(str(ab)+'a+threshold:', self.activation_range.item())
            if self._a_bit > 1:
                if self._half_wave == 'A':
                    scaling_factor = self.activation_range.item() / (2. ** self._a_bit - 1.)
                elif self._half_wave == 'Q':
                    scaling_factor = self.dw_1.item() / (2. ** self._a_bit - 1.)
                x = ori_x.detach().clone()
                x.div_(scaling_factor).round_().mul_(scaling_factor)
                # print('a+scaling_factor:', scaling_factor)
            else:
                scaling_factor = ori_x.mean().item()
                # print('a', scaling_factor)
                x = ori_x.detach().clone()
                # x.sign_().mul_(scaling_factor)
                x = BinActiveBiReal()(x)
                # STE
                # x = ori_x + x.detach() - ori_x.detach()

            return STE.apply(ori_x, x)
        else:
            return inputs

    def _quantize(self, inputs):
        inputs = self._quantize_activation(inputs=inputs)
        # weight = self._quantize_weight(weight=weight)
        # bias = self._quantize_bias(bias=bias)
        return inputs

    def forward(self, *inputs):
        raise NotImplementedError


class STE(torch.autograd.Function):
    # for faster inference
    @staticmethod
    def forward(ctx, origin_inputs, wanted_inputs):
        return wanted_inputs.detach()

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None

class QActvation(QModule1):
    def __init__(self, in_features, a_bit=8, half_wave='Q', dim=192):
        super(QActvation, self).__init__(a_bit=a_bit, half_wave=half_wave)
        print("QActvation a bitwidth:"+str(a_bit))

    def forward(self, inputs):
        return self._quantize(inputs=inputs)


def set_fix_weight(model, fix_weight=True):
    if fix_weight:
        print('\n==> set weight fixed')
    for name, module in model.named_modules():
        if isinstance(module, QModule1):
            module.set_fix_weight(fix_weight=fix_weight)


def find_scale_by_percentile(x, percentile=0.9999):
    x_cpu = x.abs().flatten().detach().cpu().numpy()
    max_k = int(x_cpu.size * percentile)
    # print(max_k)
    return np.partition(x_cpu, max_k)[max_k]

