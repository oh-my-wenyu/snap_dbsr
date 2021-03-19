import torch
import torch.nn as nn
from models.ms_resnet import ResNet

def default_conv1(in_channels, out_channels, kernel_size, bias=True, groups=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, groups=groups)

class conv_end(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, kernel_size=5, ratio=2):
        super(conv_end, self).__init__()

        modules = [
            default_conv1(in_channels, out_channels, kernel_size),
            #nn.PixelShuffle(ratio)
            nn.Upsample(scale_factor=ratio,mode='bilinear',align_corners=False)
        ]

        self.uppath = nn.Sequential(*modules)

    def forward(self, x):
        return self.uppath(x)

class DeblurNet(nn.Module):
    def __init__(self, n_feats=64, n_blocks=12, rgb_range=1, kernel_size=3, n_scales=3):
        super(DeblurNet, self).__init__()

        self.mean = rgb_range / 2
        self.n_scales = n_scales
        self.body_models = nn.ModuleList([
            ResNet(n_feats,n_blocks,rgb_range, kernel_size, 3, 3, mean_shift=False),
        ])
        for _ in range(1, self.n_scales):
            self.body_models.insert(0, ResNet(n_feats,n_blocks,rgb_range,kernel_size,6, 3, mean_shift=False))
        self.conv_end_models = nn.ModuleList([None])
        for _ in range(1, self.n_scales):
            self.conv_end_models += [conv_end(3,3)]

    def forward(self, input_pyramid):
        # with torch.no_grad():
        scales = range(self.n_scales-1, -1, -1)
        for s in scales:
            input_pyramid[s] = input_pyramid[s] - self.mean
        output_pyramid = [None] * self.n_scales
        input_s = input_pyramid[-1]
        for s in scales:    # [2, 1, 0]
            output_pyramid[s] = self.body_models[s](input_s)
            if s > 0:
                up_feat = self.conv_end_models[s](output_pyramid[s])
                input_s = torch.cat((input_pyramid[s-1], up_feat),1)
        for s in scales:
            output_pyramid[s] = output_pyramid[s] + self.mean

        return output_pyramid


