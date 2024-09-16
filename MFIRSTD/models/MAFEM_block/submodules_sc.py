#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
from MFIRSTD.models.MAFEM_block.ms_deform_attn_sc import MSDeformAttn
from torch.autograd import Variable

def conv(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True,act=nn.LeakyReLU(0.1,inplace=True)):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        act
    )

def upconv(in_channels, out_channels,act=nn.LeakyReLU(0.1,inplace=True)):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        act
    )

class DeformableAttnBlock(nn.Module):
    def __init__(self,n_heads=4,n_levels=5,n_points=4,d_model=192,n_sequence=5):
        super().__init__()
        self.n_levels = n_levels
        self.defor_attn = MSDeformAttn(d_model=d_model,n_levels=5,n_heads=n_heads,n_points=n_points)
        self.feed_forward = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.emb_qk = nn.Conv2d(n_sequence*d_model+4*(n_sequence-1), n_sequence*d_model, kernel_size=3, padding=1)
        self.emb_v = nn.Conv2d(n_sequence*d_model, n_sequence*d_model, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU(0.1,inplace=True)
        self.feedforward = nn.Sequential(
            nn.Conv2d(2*d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
            )
        self.act = nn.LeakyReLU(0.1,inplace=True)
        
    def get_reference_points(self,spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    def get_valid_ratio(self,mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    def preprocess(self,srcs):
        bs,t,c,h,w = srcs.shape
        masks = [torch.zeros((bs,h,w)).bool().to(srcs.device) for _ in range(t)]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        spatial_shapes = []
        for lv1 in range(t):
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs.device)
        return spatial_shapes,valid_ratios
    def forward(self,frame,srcframe,flow_forward,flow_backward):
        b,t,c,h,w = frame.shape
        # bs,t,c,h,w = frame.shape
        flow_forward01 = flow_forward[:, 0]
        flow_forward12 = flow_forward[:, 1]
        flow_backward32 = flow_backward[:, 2]
        flow_backward43 = flow_backward[:, 3]
        flow_forward02 = flow_forward01 + warp(flow_forward12, flow_forward01)
        flow_backward42 = flow_backward43 + warp(flow_backward32, flow_backward43)
        warp_fea02 = warp(frame[:,0],flow_forward02)
        warp_fea12 = warp(frame[:,1],flow_forward12)
        warp_fea32 = warp(frame[:,3],flow_backward32)
        warp_fea42 = warp(frame[:, 4], flow_backward42)

        qureys = self.act(self.emb_qk(torch.cat([warp_fea02,warp_fea12,frame[:,2],warp_fea32,warp_fea42,flow_forward.reshape(b,-1,h,w),flow_backward.reshape(b,-1,h,w)],1))).reshape(b,t,c,h,w)
        value = self.act(self.emb_v(frame.reshape(b,t*c,h,w)).reshape(b,t,c,h,w))
        
        spatial_shapes,valid_ratios = self.preprocess(value)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = self.get_reference_points(spatial_shapes,valid_ratios,device=value.device)
        
        output = self.defor_attn(qureys,reference_points,value,spatial_shapes,level_start_index,None,flow_forward,flow_backward)
        output = self.feed_forward(output)
        output = output.reshape(b,t,c,h,w) + frame
        
        tseq_encoder_0 = torch.cat([output.reshape(b*t,c,h,w),srcframe.reshape(b*t,c,h,w)],1)
        output = output.reshape(b*t,c,h,w) + self.feedforward(tseq_encoder_0)
        return output.reshape(b,t,c,h,w),srcframe

def warp(x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).reshape(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).reshape(-1, 1).repeat(1, W)
        xx = xx.reshape(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.reshape(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        grid = grid.to(x.device)
        vgrid = Variable(grid) + flo
        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid, padding_mode='border',align_corners=True)

        return output

