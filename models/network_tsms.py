import torch
import torch.nn as nn
from models.network_sr import SRNet
from models.network_deblur import DeblurNet


class TSMSNet(nn.Module):
    def __init__(self, n_feats_1=64, n_blocks_1=12, rgb_range_1=1, kernel_size_1=5, n_scales_1=3, dn_feats_2=128, dd_feats_2=96, dn_blocks_2=12, n_feats_2=128, n_blocks_2=12, kernel_size_2=3):
        super(TSMSNet, self).__init__()
        # deblur ----------------------------------------------------------------
        self.deblur = DeblurNet(n_feats_1, n_blocks_1, rgb_range_1, kernel_size_1, n_scales_1)
        # srmodel ---------------------------------------------------------------
        self.sr = SRNet(dn_feats_2, dd_feats_2, dn_blocks_2, n_feats_2, n_blocks_2, kernel_size_2)

    def forward(self,input_pyramid):
        # with torch.no_grad():
     
        lr_deblur = self.deblur(input_pyramid)
        lr_deblur = lr_deblur[0] #
        hr = self.sr(lr_deblur) #

        return hr
