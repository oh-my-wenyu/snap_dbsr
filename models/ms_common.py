import math

import torch
import torch.nn as nn

def default_conv(in_channels, out_channels, kernel_size, bias=True, groups=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, groups=groups)

def default_norm(n_feats):
    return nn.BatchNorm2d(n_feats)

def default_act():
    return nn.ReLU(True)

def empty_h(x, n_feats):
    '''
        create an empty hidden state

        input
            x:      B x T x 3 x H x W

        output
            h:      B x C x H/4 x W/4
    '''
    b = x.size(0)
    h, w = x.size()[-2:]
    return x.new_zeros((b, n_feats, h//4, w//4))

class Normalization(nn.Conv2d):
    """Normalize input tensor value with convolutional layer"""
    def __init__(self, mean=(0, 0, 0), std=(1, 1, 1)):
        super(Normalization, self).__init__(3, 3, kernel_size=1)
        tensor_mean = torch.Tensor(mean)
        tensor_inv_std = torch.Tensor(std).reciprocal()

        self.weight.data = torch.eye(3).mul(tensor_inv_std).view(3, 3, 1, 1)
        self.bias.data = torch.Tensor(-tensor_mean.mul(tensor_inv_std))

        for params in self.parameters():
            params.requires_grad = False

class BasicBlock(nn.Sequential):
    """Convolution layer + Activation layer"""
    def __init__(
        self, in_channels, out_channels, kernel_size, bias=True,
        conv=default_conv, norm=False, act=default_act):

        modules = []
        modules.append(
            conv(in_channels, out_channels, kernel_size, bias=bias))
        if norm: modules.append(norm(out_channels))
        if act: modules.append(act())

        super(BasicBlock, self).__init__(*modules)

class ResBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size, bias=True,
        conv=default_conv, norm=False, act=default_act):

        super(ResBlock, self).__init__()

        modules = []
        for i in range(2):
            modules.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if norm: modules.append(norm(n_feats))
            if act and i == 0: modules.append(act())

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

# ResDBlock--------------------------
#def esp_conv(in_channels, out_channels, kernel_size, dilation=[1,2,3,5],bias=True, groups=1):
#
#    split1 = nn.Conv2d(in_channels, out_channels//4, kernel_size, dilation=dilation[0], bias=bias,groups=groups)
#    split2 = nn.Conv2d(in_channels, out_channels//4, kernel_size, dilation=dilation[1], bias=bias,groups=groups)
#    split3 = nn.Conv2d(in_channels, out_channels//4, kernel_size, dilation=dilation[2], bias=bias,groups=groups)
#    split4 = nn.Conv2d(in_channels, out_channels//4, kernel_size, dilation=dilation[3], bias=bias,groups=groups)
#    sum1 = split1
#    sum2 = sum1 + split2
#    sum3 = sum2 + split3
#    sum4 = sum3 + split4
#    concate_sum = torch.cat([sum1, sum2, sum3, sum4], -1)
#    return concate_sum
#
#
#class ResDBlock(nn.Module):
#    def __init__(self, n_feats, kernel_size, bias=True, conv1=esp_conv, conv2=default_conv, act=default_act):
#        super(ResDBlock, self).__init__()
#
#        modules = []
#        modules.append(conv1(n_feats, n_feats, kernel_size, dilation=[1,2,3,5], bias=bias))
#        modules.append(act())
#        modules.append(conv2(n_feats, n_feats, kernel_size, bias=bias))
#        self.body = nn.Sequential(*modules)
#    def forward(self, x):
#        res = self.body(x)
#        res += x
#        return res


class ResDBlock(nn.Module):
    def __init__(self, n_feats, kernel_size, bias=True, groups=1, dilation=[1,2,3,5], conv2=default_conv, act=default_act):
        super(ResDBlock, self).__init__()
        self.split1 = nn.Conv2d(n_feats, n_feats // 4, kernel_size, dilation=dilation[0], padding=kernel_size // 2, bias=bias, groups=groups)
        self.split2 = nn.Conv2d(n_feats, n_feats // 4, kernel_size, dilation=dilation[1], padding=(dilation[1]*(kernel_size-1)) // 2, bias=bias, groups=groups)
        self.split3 = nn.Conv2d(n_feats, n_feats // 4, kernel_size, dilation=dilation[2], padding=(dilation[2]*(kernel_size-1)) // 2, bias=bias, groups=groups)
        self.split4 = nn.Conv2d(n_feats, n_feats // 4, kernel_size, dilation=dilation[3], padding=(dilation[3]*(kernel_size-1)) // 2, bias=bias, groups=groups)
        self.conv2 = conv2(n_feats,n_feats,kernel_size,bias=bias)
        self.act=act()

    def forward(self,x):
        sum1 = self.split1(x)
        sum2 = sum1 + self.split2(x)
        sum3 = sum2 + self.split3(x)
        sum4 = sum3 + self.split4(x)
        concate_sum = torch.cat([sum1, sum2, sum3, sum4], 1)
        concate_sum = self.act(concate_sum)
        out = self.conv2(concate_sum)
        out = out+x
        return out


class ResBlock_mobile(nn.Module):
    def __init__(
        self, n_feats, kernel_size, bias=True,
        conv=default_conv, norm=False, act=default_act, dropout=False):

        super(ResBlock_mobile, self).__init__()

        modules = []
        for i in range(2):
            modules.append(conv(n_feats, n_feats, kernel_size, bias=False, groups=n_feats))
            modules.append(conv(n_feats, n_feats, 1, bias=False))
            if dropout and i == 0: modules.append(nn.Dropout2d(dropout))
            if norm: modules.append(norm(n_feats))
            if act and i == 0: modules.append(act())

        self.body = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(
        self, scale, n_feats, bias=True,
        conv=default_conv, norm=False, act=False):

        modules = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                modules.append(conv(n_feats, 4 * n_feats, 3, bias))
                modules.append(nn.PixelShuffle(2))
                if norm: modules.append(norm(n_feats))
                if act: modules.append(act())
        elif scale == 3:
            modules.append(conv(n_feats, 9 * n_feats, 3, bias))
            modules.append(nn.PixelShuffle(3))
            if norm: modules.append(norm(n_feats))
            if act: modules.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*modules)

# Only support 1 / 2
class PixelSort(nn.Module):
    """The inverse operation of PixelShuffle
    Reduces the spatial resolution, increasing the number of channels.
    Currently, scale 0.5 is supported only.
    Later, torch.nn.functional.pixel_sort may be implemented.
    Reference:
        http://pytorch.org/docs/0.3.0/_modules/torch/nn/modules/pixelshuffle.html#PixelShuffle
        http://pytorch.org/docs/0.3.0/_modules/torch/nn/functional.html#pixel_shuffle
    """
    def __init__(self, upscale_factor=0.5):
        super(PixelSort, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, 2, 2, h // 2, w // 2)
        x = x.permute(0, 1, 5, 3, 2, 4).contiguous()
        x = x.view(b, 4 * c, h // 2, w // 2)

        return x

class Downsampler(nn.Sequential):
    def __init__(
        self, scale, n_feats, bias=True,
        conv=default_conv, norm=False, act=False):

        modules = []
        if scale == 0.5:
            modules.append(PixelSort())
            modules.append(conv(4 * n_feats, n_feats, 3, bias))
            if norm: modules.append(norm(n_feats))
            if act: modules.append(act())
        else:
            raise NotImplementedError

        super(Downsampler, self).__init__(*modules)


