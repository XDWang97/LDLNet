import torch
import torch.nn as nn
from net.smt import smt_s
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np
import cv2


class MVCA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.Linear1 = nn.Conv2d(dim, dim, 1)
        self.DWConv5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')
        self.DWConv7 = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, padding_mode='reflect')
        self.DWConv9 = nn.Conv2d(dim, dim, kernel_size=9, padding=4, groups=dim, padding_mode='reflect')
        self.Linear2 = nn.Conv2d(dim, dim, 1)
        self.norm2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        identity = x
        x = self.Linear1(x)
        x1 = self.DWConv5(x)
        x2 = self.DWConv7(x)
        x3 = self.DWConv9(x)
        out = x1+x2+x3
        out = self.Linear2(out) + identity
        return out


class MVCABlocks(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.ModuleList([MVCA(dim=dim,)for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class IIB(nn.Module):
    def __init__(self):
        super(IIB, self).__init__()
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(64)

        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1b   = nn.BatchNorm2d(64)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2b   = nn.BatchNorm2d(64)
        self.conv3b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3b   = nn.BatchNorm2d(64)
        self.conv4b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4b   = nn.BatchNorm2d(64)

        self.conv1d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1d   = nn.BatchNorm2d(64)
        self.conv2d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2d   = nn.BatchNorm2d(64)
        self.conv3d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3d   = nn.BatchNorm2d(64)
        self.conv4d = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4d   = nn.BatchNorm2d(64)

    def forward(self, out1):
        out1 = F.relu(self.bn1(self.conv1(out1)), inplace=True)
        out2 = F.max_pool2d(out1, kernel_size=2, stride=2)
        out2 = F.relu(self.bn2(self.conv2(out2)), inplace=True)
        out3 = F.max_pool2d(out2, kernel_size=2, stride=2)
        out3 = F.relu(self.bn3(self.conv3(out3)), inplace=True)
        out4 = F.max_pool2d(out3, kernel_size=2, stride=2)
        out4 = F.relu(self.bn4(self.conv4(out4)), inplace=True)

        out1b = F.relu(self.bn1b(self.conv1b(out1)), inplace=True)
        out2b = F.relu(self.bn2b(self.conv2b(out2)), inplace=True)
        out3b = F.relu(self.bn3b(self.conv3b(out3)), inplace=True)
        out4b = F.relu(self.bn4b(self.conv4b(out4)), inplace=True)

        out1d = F.relu(self.bn1d(self.conv1d(out1)), inplace=True)
        out2d = F.relu(self.bn2d(self.conv2d(out2)), inplace=True)
        out3d = F.relu(self.bn3d(self.conv3d(out3)), inplace=True)
        out4d = F.relu(self.bn4d(self.conv4d(out4)), inplace=True)
        return (out4b, out3b, out2b, out1b), (out4d, out3d, out2d, out1d)


import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelAttentionModule(nn.Module):
    def __init__(self):
        super(PixelAttentionModule, self).__init__()

    def forward(self, x):

        saliency_map = torch.sigmoid(x.view(x.size(0), x.size(1), -1))  # 使用 sigmoid 替代 softmax
        saliency_map = saliency_map.view(x.size())  # Reshape back to original size
        return x * saliency_map  

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        avg_pool = self.global_avg_pool(x).view(batch_size, channels)
        attention = F.relu(self.fc1(avg_pool))
        attention = torch.sigmoid(self.fc2(attention)).view(batch_size, channels, 1, 1)
        return x * attention 

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.pixel_attention = PixelAttentionModule()
        self.channel_attention = ChannelAttentionModule(in_channels)

    def forward(self, x):
        pixel_attended = self.pixel_attention(x)
        channel_attended = self.channel_attention(pixel_attended)
        return channel_attended

class BB(nn.Module):
    def __init__(self):
        super(BB, self).__init__()
        self.attention_module = AttentionModule(in_channels=64)  # 添加注意力模块
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

    def forward(self, input1, input2=[0, 0, 0, 0]):
        # Step 1: Decode with Attention
        """
        torch.Size([4, 64, 22, 22])
        torch.Size([4, 64, 44, 44])
        torch.Size([4, 64, 88, 88])
        torch.Size([4, 64, 176, 176])
        """
        out0 = F.relu(self.bn0(self.conv0(input1[0] + input2[0])), inplace=True)
        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')

        out0_attended = self.attention_module(out0)

        out1 = F.relu(self.bn1(self.conv1(input1[1] + input2[1]+out0_attended)), inplace=True)
        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')

        out1_attended = self.attention_module(out1)

        out2 = F.relu(self.bn2(self.conv2(input1[2] + input2[2]+out1_attended)), inplace=True)
        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')

        out2_attended = self.attention_module(out2)

        out3 = F.relu(self.bn3(self.conv3(input1[3] + input2[3]+out2_attended)), inplace=True)

        return out3


class DB(nn.Module):
    def __init__(self):
        super(DB, self).__init__()
        self.conv0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn0   = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)
        self.mvca = MVCABlocks(dim=64, depth=1)

    def forward(self, input1, input2=[0,0,0,0]):
        out0 = F.relu(self.bn0(self.conv0(input1[0]+input2[0])), inplace=True)
        out0 = F.interpolate(out0, size=input1[1].size()[2:], mode='bilinear')
        out0 = self.mvca(out0)
        out1 = F.relu(self.bn1(self.conv1(input1[1]+input2[1]+out0)), inplace=True)
        out1 = F.interpolate(out1, size=input1[2].size()[2:], mode='bilinear')
        out1 = self.mvca(out1)
        out2 = F.relu(self.bn2(self.conv2(input1[2]+input2[2]+out1)), inplace=True)
        out2 = F.interpolate(out2, size=input1[3].size()[2:], mode='bilinear')
        out2 = self.mvca(out2)
        out3 = F.relu(self.bn3(self.conv3(input1[3]+input2[3]+out2)), inplace=True)
        return out3

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.backbone = smt_s()  # [64, 128, 320, 512]
        path = '/home/ubuntu/wxd/cod/code/LDLNet/models/smt_small.pth'
        self.backbone.load_state_dict(torch.load(path)['model'])
        print(f"loading pre_model ${path}")
        
        self.conv5b   = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4b   = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3b   = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2b   = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.conv5d   = nn.Sequential(nn.Conv2d(512, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv4d   = nn.Sequential(nn.Conv2d(256, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3d   = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv2d   = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.bb = BB()
        self.db = DB()
        self.iib = IIB()

        self.linearb  = nn.Conv2d(64, 1, kernel_size=3, padding =1)
        self.lineard  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.linear   = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 1, kernel_size=3, padding=1))

        
    def forward(self, x,shape=None):
        out2, out3, out4, out5 = self.backbone(x) # 64,128,320,512

        out2b, out3b, out4b, out5b   = self.conv2b(out2), self.conv3b(out3), self.conv4b(out4), self.conv5b(out5)
        out2d, out3d, out4d, out5d   = self.conv2d(out2), self.conv3d(out3), self.conv4d(out4), self.conv5d(out5)

        final_outb1 = self.bb([out5b, out4b, out3b, out2b])
        final_outd1 = self.db([out5d, out4d, out3d, out2d])
        final_out1 = torch.cat([final_outb1, final_outd1], dim=1)
        final_encoder_outb_1, final_encoder_outd_1 = self.iib(final_out1)

        final_outb2 = self.bb([out5b, out4b, out3b, out2b], final_encoder_outb_1)
        final_outd2 = self.db([out5d, out4d, out3d, out2d], final_encoder_outd_1)
        final_out2  = torch.cat([final_outb2, final_outd2], dim=1)
        final_encoder_outb_2, final_encoder_outd_2 = self.iib(final_out2)

        final_outb3 = self.bb([out5b, out4b, out3b, out2b], final_encoder_outb_2)
        final_outd3 = self.db([out5d, out4d, out3d, out2d], final_encoder_outd_2)
        final_out3 = torch.cat([final_outb3, final_outd3], dim=1)


        if shape is None:
            shape = x.size()[2:]
        final_out1  = F.interpolate(self.linear(final_out1),   size=shape, mode='bilinear')
        final_outb1 = F.interpolate(self.linearb(final_outb1), size=shape, mode='bilinear')
        final_outd1 = F.interpolate(self.lineard(final_outd1), size=shape, mode='bilinear')


        final_out2  = F.interpolate(self.linear(final_out2),   size=shape, mode='bilinear')
        final_outb2 = F.interpolate(self.linearb(final_outb2), size=shape, mode='bilinear')
        final_outd2 = F.interpolate(self.lineard(final_outd2), size=shape, mode='bilinear')

        final_out3  = F.interpolate(self.linear(final_out3),   size=shape, mode='bilinear')
        final_outb3 = F.interpolate(self.linearb(final_outb3), size=shape, mode='bilinear')
        final_outd3 = F.interpolate(self.lineard(final_outd3), size=shape, mode='bilinear')
        
        return final_outb1, final_outd1, final_out1, final_outb2, final_outd2, final_out2, final_outb3, final_outd3, final_out3
