"""
[1] https://github.com/yiskw713/DualAttention_for_Segmentation/blob/master/model/danet.py
[2] https://arxiv.org/pdf/1809.02983.pdf

@misc{fu2019dual,
      title={Dual Attention Network for Scene Segmentation}, 
      author={Jun Fu and Jing Liu and Haijie Tian and Yong Li and Yongjun Bao and Zhiwei Fang and Hanqing Lu},
      year={2019},
      eprint={1809.02983},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""

import torch.nn as nn
from .drnnet import drn_d_22, drn_d_38
import torch
import torch.nn.functional as F


class PositionAttentionModule(nn.Module):
    ''' self-attention '''

    def __init__(self, in_channels):        
        super().__init__()
        self.query_conv = nn.Conv2d(
            in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : feature maps from feature extractor. (N, C, H, W)
        outputs :
            feature maps weighted by attention along spatial dimensions
        """

        N, C, H, W = x.shape
        query = self.query_conv(x).view(
            N, -1, H*W).permute(0, 2, 1)  # (N, H*W, C')
        key = self.key_conv(x).view(N, -1, H*W)  # (N, C', H*W)

        # caluculate correlation
        energy = torch.bmm(query, key)    # (N, H*W, H*W)
        # spatial normalize
        attention = self.softmax(energy)

        value = self.value_conv(x).view(N, -1, H*W)    # (N, C, H*W)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(N, C, H, W)
        out = self.gamma*out + x
        return out


class ChannelAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : feature maps from feature extractor. (N, C, H, W)
        outputs :
            feature maps weighted by attention along a channel dimension
        """

        N, C, H, W = x.shape
        query = x.view(N, C, -1)    # (N, C, H*W)
        key = x.view(N, C, -1).permute(0, 2, 1)    # (N, H*W, C)

        # calculate correlation
        energy = torch.bmm(query, key)    # (N, C, C)
        energy = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy)

        value = x.view(N, C, -1)

        out = torch.bmm(attention, value)
        out = out.view(N, C, H, W)
        out = self.gamma*out + x
        return out


class DANet(nn.Module):
    def __init__(self, config, inter_channel=512):
        super().__init__()

        # set a base model
        if config.model == 'drn_d_22':
            print('Dilated ResNet D 22 wil be used as a base model')
            self.base = drn_d_22(pretrained=True)
            # remove the last layer (out_conv)
            self.base = nn.Sequential(
                *list(self.base.children())[:-1])
        elif config.model == 'drn_d_38':
            print('Dilated ResNet D 38 wil be used as a base model')
            self.base = drn_d_38(pretrained=True)
            # remove the last layer (out_conv)
            self.base = nn.Sequential(
                *list(self.base.children())[:-1])
        else:
            print('There is no option you choose as a base model.')
            print('Instead, Dilated ResNet D 22 wil be used as a base model')
            self.base = drn_d_22(pretrained=True)
            # remove the last layer (out_conv)
            self.base = nn.Sequential(
                *list(self.base.children())[:-1])

        # convolution before attention modules
        self.conv2pam = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU()
        )
        self.conv2cam = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU()
        )

        # attention modules
        self.pam = PositionAttentionModule(in_channels=inter_channel)
        self.cam = ChannelAttentionModule()

        # convolution after attention modules
        self.pam2conv = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU())
        self.cam2conv = nn.Sequential(
            nn.Conv2d(inter_channel, inter_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU())

        # output layers for each attention module and sum features.
        self.conv_pam_out = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channel, config.n_classes, 1)
        )
        self.conv_cam_out = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channel, config.n_classes, 1)
        )
        self.conv_out = nn.Sequential(
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channel, config.n_classes, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.base(x)

        # outputs from attention modules
        pam_out = self.conv2pam(x)
        pam_out = self.pam(pam_out)
        pam_out = self.pam2conv(pam_out)

        cam_out = self.conv2cam(x)
        cam_out = self.cam(cam_out)
        cam_out = self.cam2conv(cam_out)

        # segmentation result
        outputs = []
        feats_sum = pam_out + cam_out
        outputs.append(self.conv_out(feats_sum))
        outputs.append(self.conv_pam_out(pam_out))
        outputs.append(self.conv_cam_out(cam_out))

        return outputs