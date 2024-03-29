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


def reconstruct_weight_from_k_means_result(centroids, labels):
    weight = torch.zeros_like(labels).float().cuda()
    for i, c in enumerate(centroids.cpu().numpy().squeeze()):
        weight[labels == i] = c.item()
    return weight

def kmeans_update_model(model, quantizable_idx, centroid_label_dict, free_high_bit=False):
    for i, layer in enumerate(model.modules()):
        if i not in quantizable_idx:
            continue
        new_weight_data = layer.weight.data.clone()
        new_weight_data.zero_()
        this_cl_list = centroid_label_dict[i]
        num_centroids = this_cl_list[0][0].numel()
        if num_centroids > 2**6 and free_high_bit:
            # quantize weight with high bit will not lead accuracy loss, so we can omit them to save time
            continue
        for j in range(num_centroids):
            mask_cl = (this_cl_list[0][1] == j).float()
            new_weight_data += (layer.weight.data * mask_cl).sum() / mask_cl.sum() * mask_cl
        layer.weight.data = new_weight_data

class Quant(nn.Module):
    def __init__(self, range=6, dim=192):
        super(Quant, self).__init__()
        self.range = torch.tensor([range], requires_grad=True)

    def forward(self, inputs, half_wave, a_bit):
        B, N, C = inputs.shape
        if self.range == 0:
            a = torch.tensor(torch.zeros(B, N, C), device=inputs.device)
            return a, a
        activation_r = torch.tensor(self.range.clone().repeat(C), device=inputs.device, requires_grad=True)
        lw_1 = torch.tensor((self.range).clone().repeat(C), device=inputs.device, requires_grad=True)
        rw_1 = torch.tensor((self.range).clone().repeat(C), device=inputs.device, requires_grad=True)
        scaling_factor1 = activation_r / (2. ** a_bit - 1.)
        b = scaling_factor1.repeat(N, 1).unsqueeze(0).repeat(B, 1, 1)
        if half_wave == 'A':
            ori_x = 0.5 * (inputs.abs() - (inputs - activation_r).abs() + activation_r)
        elif half_wave == 'Q':
            ori_x = 0.5 * ((-inputs + lw_1).abs() - (inputs - rw_1).abs() + lw_1 + rw_1)
        # quantization
        x = ori_x.detach().clone()
        x.div_(b).round_().mul_(b)
        return ori_x, x

class QModule(nn.Module):
    def __init__(self, w_bit=8, a_bit=8, half_wave='A', dim=192,**kwargs):
        super(QModule, self).__init__()
        # global group_n
        self.args = kwargs["args"]
        self.group_n = self.args.group_num
        if half_wave == 'A' or half_wave == 'Q':
            self._a_bit = a_bit
        else:
            self._a_bit = a_bit - 1
        self._w_bit = w_bit
        self._b_bit = 32
        self._half_wave = half_wave
        self.dim = dim

        self.init_range = 6.
        self.activation_range = torch.zeros(dim)
        self.groups_range1 = nn.Parameter(torch.zeros(self.group_n), requires_grad=True)
        self.weight_range = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)

        self._quantized = True
        self._tanh_weight = False
        self._fix_weight = False
        self._trainable_activation_range = True
        self._calibrate = False

        self.alpha_activ = nn.Parameter(torch.Tensor(self.group_n, dim), requires_grad=True)
        self.alpha_activ.data.fill_(0.01)
        self.sw = torch.Tensor(self.group_n, dim)
        self.mix_activ_mark1 = nn.ModuleList()

    def QuantGroup(self, input):
        B, N, C = input.shape
        outs_x = torch.zeros([1, B, N, C], requires_grad=True).to(input.device)
        outs_ori = torch.zeros([1, B, N, C], requires_grad=True).to(input.device)
        self.mix_activ_mark1 = nn.ModuleList()
        for group in self.groups_range1.data:
            self.mix_activ_mark1.append(Quant(range=group,dim=self.dim))
        self.sw = F.softmax(self.alpha_activ, dim=0)
        for i, branch in enumerate(self.mix_activ_mark1):
            ori_x, x = branch(input, half_wave=self._half_wave, a_bit=self._a_bit)
            if i == 0:
                outs_x, outs_ori = (x * self.sw[i]).unsqueeze(0).clone(), (ori_x * self.sw[i]).unsqueeze(0).clone()
            else:
                outs_x = torch.cat((outs_x, (x * self.sw[i]).unsqueeze(0)), 0)
                outs_ori = torch.cat((outs_ori, (ori_x * self.sw[i]).unsqueeze(0)), 0)
        activ = torch.sum(outs_x, dim=0)
        ori_activ = torch.sum(outs_ori, dim=0)
        return ori_activ, activ

    @property
    def w_bit(self):
        return self._w_bit

    @w_bit.setter
    def w_bit(self, w_bit):
        self._w_bit = w_bit

    @property
    def dim(self):
        return self._dim

    @dim.setter
    def dim(self, dim):
        self._dim = dim

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
        B, N, C = inputs.shape
        if self._quantized and self._a_bit > 0:
            if self._calibrate:
                inputs_calibrate = inputs.clone().transpose(0, 2)
                for pixel in range(C):
                    estimate_activation_range = min(self.init_range,
                                                    find_scale_by_percentile(inputs_calibrate[pixel]))
                    if pixel == 0:
                        activation_range2 = torch.tensor([estimate_activation_range])
                        continue
                    activation_range_tem = torch.tensor([estimate_activation_range])
                    activation_range2 = torch.cat((activation_range2, activation_range_tem), 0)

                activation_range2, groups_range = two_groups(activation_range2, mlp2=C, group_n=self.group_n)
                return inputs

            # print(activation_r, activation_r.grad)
            ori_x, x = self.QuantGroup(inputs)

            return STE.apply(ori_x, x)
            # return x
        else:
            return inputs

    def _quantize_weight(self, weight):
        #self._calibrate = True
        if self._tanh_weight:
            weight = weight.tanh()
            weight = weight / weight.abs().max()

        if self._quantized and self._w_bit > 0:
            #print('w_bit:', self._w_bit)
            threshold = self.weight_range.item()
            if threshold <= 0:
                threshold = weight.abs().max().item()
                self.weight_range.data[0] = threshold

            if self._calibrate:
                if self._w_bit < 5:
                    threshold = self._compute_threshold(weight.data.cpu().numpy(), self._w_bit)
                else:
                    threshold = weight.abs().max().item()
                self.weight_range.data[0] = threshold
                return weight
            #print('w', threshold)
            ori_w = weight

            if self._w_bit > 1:
                scaling_factor = threshold / (pow(2., self._w_bit - 1) - 1.)
                w = ori_w.clamp(-threshold, threshold)
                # w[w.abs() > threshold - threshold / 64.] = 0.
                w.div_(scaling_factor).round_().mul_(scaling_factor)
            else:
                #scaling_factor = threshold
                scaling_factor = ori_w.abs().mean()
                #print('w', scaling_factor)
                w = ori_w.clamp(-threshold, threshold)
                #print('before', w)
                w.sign_().mul_(scaling_factor)
                #print('after', w)
            # STE
            if self._fix_weight:
                # w = w.detach()
                return w.detach()
            else:
                # w = ori_w + w.detach() - ori_w.detach()
                return STE.apply(ori_w, w)
        else:
            return weight

    def _quantize_bias(self, bias):
        if bias is not None and self._quantized and self._b_bit > 0:
            if self._calibrate:
                return bias
            ori_b = bias
            threshold = ori_b.data.max().item() + 0.00001
            scaling_factor = threshold / (pow(2., self._b_bit - 1) - 1.)
            b = torch.clamp(ori_b.data, -threshold, threshold)
            b.div_(scaling_factor).round_().mul_(scaling_factor)
            # STE
            if self._fix_weight:
                return b.detach()
            else:
                # b = ori_b + b.detach() - ori_b.detach()
                return STE.apply(ori_b, b)
        else:
            return bias

    def _quantize(self, inputs, weight, bias):
        inputs = self._quantize_activation(inputs=inputs)
        weight = self._quantize_weight(weight=weight)
        # bias = self._quantize_bias(bias=bias)
        return inputs, weight, bias

    def forward(self, *inputs):
        raise NotImplementedError

    def extra_repr(self):
        return 'w_bit={}, a_bit={}, half_wave={}, tanh_weight={}'.format(
            self.w_bit if self.w_bit > 0 else -1, self.a_bit if self.a_bit > 0 else -1,
            self.half_wave, self._tanh_weight
        )


class STE(torch.autograd.Function):
    # for faster inference
    @staticmethod
    def forward(ctx, origin_inputs, wanted_inputs):
        return wanted_inputs.detach()

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


class QConv2d(QModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False,
                 w_bit=-1, a_bit=-1, half_wave=True):
        super(QConv2d, self).__init__(w_bit=w_bit, a_bit=a_bit, half_wave=half_wave)
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        inputs, weight, bias = self._quantize(inputs=inputs, weight=self.weight, bias=self.bias)
        return F.conv2d(inputs, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.w_bit > 0 or self.a_bit > 0:
            s += ', w_bit={}, a_bit={}'.format(self.w_bit, self.a_bit)
            s += ', half wave' if self.half_wave else ', full wave'
        return s.format(**self.__dict__)


class QLinear(QModule):
    def __init__(self, in_features, out_features, bias=True, w_bit=8, a_bit=8, half_wave='A', dim=192,**kwargs):
        super(QLinear, self).__init__(w_bit=w_bit, a_bit=a_bit, half_wave=half_wave, dim=in_features,**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.dim = in_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, inputs):
        inputs, weight, bias = self._quantize(inputs=inputs, weight=self.weight, bias=self.bias)
        return F.linear(inputs, weight=weight, bias=bias)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)
        if self.w_bit > 0 or self.a_bit > 0:
            s += ', w_bit={w_bit}, a_bit={a_bit}'.format(w_bit=self.w_bit, a_bit=self.a_bit)
            s += ', half wave' if self.half_wave else ', full wave'
        return s

def calibrate(model, loader, device):
    data_parallel_flag = False
    if hasattr(model, 'module'):
        data_parallel_flag = True
        model = model.module
    print('\n==> start calibrate')
    for name, module in model.named_modules():
        if isinstance(module, QModule):
            module.set_calibrate(calibrate=True)
    inputs, _ = next(iter(loader))
    # use 1 gpu to calibrate
    inputs = inputs.to(device, non_blocking=True)
    for i in range(7):
        inputs1, _ = next(iter(loader))
        inputs1 = inputs1.to(device, non_blocking=True)
        inputs = torch.cat((inputs, inputs1), 0)
    with torch.no_grad():
        model(inputs)
    for name, module in model.named_modules():
        if isinstance(module, QModule):
            module.set_calibrate(calibrate=False)
    print('==> end calibrate')
    if data_parallel_flag:
        model = nn.DataParallel(model)
    return model


def dorefa(model):
    print('\n==> set weight tanh')
    for name, module in model.named_modules():
        if isinstance(module, QModule):
            module.set_tanh(tanh=True)


def set_fix_weight(model, fix_weight=True):
    if fix_weight:
        print('\n==> set weight fixed')
    for name, module in model.named_modules():
        if isinstance(module, QModule):
            module.set_fix_weight(fix_weight=fix_weight)


def find_scale_by_percentile(x, percentile=0.9999):
    x_cpu = x.abs().flatten().detach().cpu().numpy()
    max_k = int(x_cpu.size * percentile)
    # print(max_k)
    return np.partition(x_cpu, max_k)[max_k]

from numpy import *
def two_groups(x, mlp2, group_n):
    # global group_n
    C = mlp2
    range_max = x.max()
    range_min = x.min()
    range_group = [range_min]
    range_div = range_max - range_min
    for m in range(group_n):
        range_group.append(range_min + range_div * (m+1)/group_n)
    # print(range_group)

    group_list = [[] for i in range(group_n)]
    mark = torch.zeros(C*group_n)
    # mark = torch.zeros(C*group_n)
    for k in range(C):
        for m in range(group_n):
            if x[k] >= range_group[m] and x[k] < range_group[m+1]:
                group_list[m].append(x[k])
                mark[m*C+k] = 1
            elif x[k] == range_group[-1]:
                group_list[-1].append(x[k])
                mark[(group_n-1) * C + k] = 1
    group_mean = []
    for m in range(group_n):
        # group_mean1 = mean(group_list[m])
        group_mean1 = max(group_list[m], default=0)
        # print(group_mean1)
        group_mean.append(group_mean1)
    # activation_range2[0] = 2
    # print(group_list[-1], range_group, group_mean)
    group_mean = torch.tensor(group_mean, requires_grad=True)
    # print(group_mean)
    for k in range(C):
        for m in range(group_n):
            if x[k] >= range_group[m] and x[k] < range_group[m+1]:
                x[k] = group_mean[m]
            elif x[k] == range_group[-1]:
                x[k] = group_mean[-1]
    #print(mark)
    # print(group_mean, group_mean.grad)
    # return x, mark, group_mean
    return x, group_mean

