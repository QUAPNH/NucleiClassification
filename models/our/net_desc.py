import math
from collections import OrderedDict
from .layers import CombinationModule
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .net_utils import (DenseBlock, Net, ResidualBlock, TFSamepaddingLayer,
                        UpSample2x)
from .utils import crop_op, crop_to_shape
from .scale_attention_layer import *
from .sff import *


####
class SRENet(Net):
    def __init__(self, heads, input_ch=3, nr_types=None, freeze=False, mode='fast'):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4

        assert mode == 'original' or mode == 'fast', \
            'Unknown mode `%s` for SRENet %s. Only support `original` or `fast`.' % mode

        module_list = [
            ("/", nn.Conv2d(input_ch, 64, 7, stride=1, padding=0, bias=False)),
            ("bn", nn.BatchNorm2d(64, eps=1e-5)),
            ("relu", nn.ReLU(inplace=True)),
        ]
        if mode == 'fast':  # prepend the padding for `fast` mode
            module_list = [("pad", TFSamepaddingLayer(ksize=7, stride=1))] + module_list

        self.conv0 = nn.Sequential(OrderedDict(module_list))
        self.d0 = ResidualBlock(64, [1, 3, 1], [64, 64, 256], 3, stride=1)
        self.d1 = ResidualBlock(256, [1, 3, 1], [128, 128, 512], 4, stride=2)
        self.d2 = ResidualBlock(512, [1, 3, 1], [256, 256, 1024], 6, stride=2)
        self.d3 = ResidualBlock(1024, [1, 3, 1], [512, 512, 2048], 3, stride=2)


        # TODO: pytorch still require the channel eventhough its ignored
        self.weights_init()
        self.dec_c4 = CombinationModule(2048, 1024, batch_norm=True)
        self.dec_c3 = CombinationModule(1024, 512, batch_norm=True)
        self.dec_c2 = CombinationModule(512, 256, batch_norm=True)
        
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(256, 256,
                                             kernel_size=7, padding=7 // 2, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, classes,
                                             kernel_size=7, padding=7 // 2, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(256, 256,
                                             kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256, classes, kernel_size=1, stride=1,
                                             padding=1 // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)        

        
        def down_stage(in_chan, out_chan):
            return nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_chan),
                nn.PReLU()
            )

        self.down4 = down_stage(256,64)
        self.down3 = down_stage(512,64)
        self.down2 = down_stage(1024,64)

        self.cx = sff(192)

        def create_decoder_branch(out_ch=2):
            module_list = [
                ("att", scale_atten_convblock(576,24)),
                ("conv", nn.Conv2d(24, out_ch, 1, stride=1, padding=0, bias=True),),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([("u0", u0), ])
            )
            return decoder


        if nr_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("np", create_decoder_branch(2)),
                        ("hv", create_decoder_branch(2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("tp", create_decoder_branch(nr_types)),
                        ("np", create_decoder_branch(2)),
                        ("hv", create_decoder_branch(2)),
                    ]
                )
            )

    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, imgs):

        imgs = imgs / 255.0  # to 0-1 range to match XY

        if self.training:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0, self.freeze)
            with torch.set_grad_enabled(not self.freeze):
                d1 = self.d1(d0)
                d2 = self.d2(d1)
                d3 = self.d3(d2)
            c4_combine = self.dec_c4(d3, d2)
            d = [d0, d1, d2, d3]
        else:
            d0 = self.conv0(imgs)
            d0 = self.d0(d0)
            d1 = self.d1(d0)
            d2 = self.d2(d1)
            d3 = self.d3(d2)
            c4_combine = self.dec_c4(d3, d2)
            d = [d0, d1, d2, d3]

        c3_combine = self.dec_c3(c4_combine, d1)
        c2_combine = self.dec_c2(c3_combine, d0)

        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)

        dec_dict['hm'] = torch.sigmoid(dec_dict['hm'])

        out_dict = OrderedDict()


        a2 = F.interpolate(self.down2(c4_combine), size=[256,256], mode='bilinear', align_corners=False)
        a3 = F.interpolate(self.down3(c3_combine), size=[256,256], mode='bilinear', align_corners=False)
        a4 = self.down4(c2_combine)

        s4=torch.cat([a2,a3,a4],dim=1)

        ret1,ret2,mix = self.cx(s4)

        result = torch.cat([ret1, ret2, mix], dim=1)

        for branch_name, branch_desc in self.decoder.items():
            u0 = branch_desc[0](result)
            out_dict[branch_name] = u0

        return out_dict, dec_dict


####
def create_model(heads, mode=None, **kwargs):
    if mode not in ['original', 'fast']:
        assert "Unknown Model Mode %s" % mode
    return SRENet(heads, mode=mode, **kwargs)

