import torch
import torch.nn as nn
from models import mdcn_common as mdcn_common


class MDCB(nn.Module):
    def __init__(self,n_feats=128,d_feats=98, conv=mdcn_common.default_conv):
        super(MDCB, self).__init__()

        kernel_size_1 = 3
        kernel_size_2 = 5
        act = nn.ReLU(True)

        self.conv_3_1 = conv(n_feats, n_feats, kernel_size_1)
        self.conv_3_2 = conv(d_feats, d_feats, kernel_size_1)
        self.conv_5_1 = conv(n_feats, n_feats, kernel_size_2)
        self.conv_5_2 = conv(d_feats, d_feats, kernel_size_2)
        self.confusion_3 = nn.Conv2d(n_feats * 3, d_feats, 1, padding=0, bias=True)
        self.confusion_5 = nn.Conv2d(n_feats * 3, d_feats, 1, padding=0, bias=True)
        self.confusion_bottle = nn.Conv2d(n_feats * 3 + d_feats * 2, n_feats, 1, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([input_1, output_3_1, output_5_1], 1)
        input_2_3 = self.confusion_3(input_2)
        input_2_5 = self.confusion_5(input_2)

        output_3_2 = self.relu(self.conv_3_2(input_2_3))
        output_5_2 = self.relu(self.conv_5_2(input_2_5))
        input_3 = torch.cat([input_1, output_3_1, output_5_1, output_3_2, output_5_2], 1)
        output = self.confusion_bottle(input_3)
        output += x
        return output

class MCALayer(nn.Module):
    def __init__(self, n_feats, reduction=16):
        super(MCALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(n_feats, n_feats // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_feats // reduction, n_feats, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class DB(nn.Module):
    def __init__(self, n_feats=128,d_feats=96,n_blocks=12,conv=mdcn_common.default_conv):
        super(DB, self).__init__()


        self.fushion_down = nn.Conv2d(n_feats * (n_blocks - 1), d_feats, 1, padding=0, bias=True)
        self.channel_attention = MCALayer(d_feats)
        self.fushion_up = nn.Conv2d(d_feats, n_feats, 1, padding=0, bias=True)

    def forward(self, x):
        x = self.fushion_down(x)
        x = self.channel_attention(x)
        x = self.fushion_up(x)
        return x
    
class SRNet(nn.Module):
    def __init__(self,dn_feats=128,dd_feats=96,dn_blocks=12,n_feats=128, n_blocks=12, kernel_size=3, conv=mdcn_common.default_conv):
        super(SRNet, self).__init__()

        self.scale_idx = 0
        scale_up = [4]
        self.n_blocks = n_blocks
        # define head module
        modules_head = [conv(3, n_feats, kernel_size)]
        # define body module
        modules_body = nn.ModuleList()
        for i in range(n_blocks):
            modules_body.append(MDCB(n_feats=dn_feats,d_feats=dd_feats))
        # define distillation module
        modules_dist = nn.ModuleList()
        modules_dist.append(DB(n_feats=dn_feats,d_feats=dd_feats,n_blocks=dn_blocks))
        modules_transform = [conv(n_feats, n_feats, kernel_size)]
        self.upsample = nn.ModuleList([
            mdcn_common.Upsampler(
                conv, s, n_feats, act=True
            ) for s in scale_up
        ])
        modules_rebult = [conv(n_feats, 3, kernel_size)]
        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.dist = nn.Sequential(*modules_dist)
        self.transform = nn.Sequential(*modules_transform)
        self.rebult = nn.Sequential(*modules_rebult)

    def forward(self, x):
        x = self.head(x)
        front = x
        MDCB_out = []
        for i in range(self.n_blocks):
            x = self.body[i](x)
            if i != (self.n_blocks - 1):
                MDCB_out.append(x)

        hierarchical = torch.cat(MDCB_out, 1)
        hierarchical = self.dist(hierarchical)

        mix = front + hierarchical + x

        out = self.transform(mix)
        out = self.upsample[self.scale_idx](out)
        out = self.rebult(out)
        return out


