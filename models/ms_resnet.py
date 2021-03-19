import torch.nn as nn
import models.ms_common as common


class ResNet(nn.Module):
    def __init__(self, n_feats, n_resblocks, rgb_range, kernel_size, in_channels=3, out_channels=3, mean_shift=True):
        super(ResNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.n_feats = n_feats
        self.kernel_size = kernel_size
        self.n_resblocks = n_resblocks

        self.mean_shift = mean_shift
        self.rgb_range = rgb_range
        self.mean = self.rgb_range / 2

        modules = []
        modules.append(common.default_conv(self.in_channels, self.n_feats, self.kernel_size))
        for _ in range(self.n_resblocks):
            modules.append(common.ResBlock(self.n_feats, self.kernel_size))
        modules.append(common.default_conv(self.n_feats, self.out_channels, self.kernel_size))

        self.body = nn.Sequential(*modules)

    def forward(self, input):
        if self.mean_shift:
            input = input - self.mean

        output = self.body(input)

        if self.mean_shift:
            output = output + self.mean

        return output


